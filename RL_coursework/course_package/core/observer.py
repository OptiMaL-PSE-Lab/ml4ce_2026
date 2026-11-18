"""Turn measurements into the observation vector of a policy."""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import NDArray

from course_package.core.measurements import Measurements
from course_package.core.params import HouseParams
from course_package.core.tracking import (
    DEFAULT_ERROR_WINDOW,
    ErrorState,
    TrackingErrorFilter,
)


@dataclass(frozen=True)
class FeatureScales:
    """Divisors of the features, fitted by ``ScaleFitter`` on the training split of
    ``10_prosumers_diverse`` (``scripts/observer/fit_feature_scales.py``).

    ``price_ref`` (price features), ``net_load_ref`` (net exogenous load) and
    ``temp_ref`` (setpoint - outdoor temperature) are the 0.99 quantile of the
    absolute raw quantity; ``hours_ref`` and ``setpoint_horizon`` are the forecast
    horizon; ``ghi_ref`` is the STC irradiance; ``clip`` bounds outliers.
    """

    price_ref: float = 0.417
    net_load_ref: float = 1.88
    hours_ref: float = 36.0
    setpoint_horizon: float = 36.0
    temp_ref: float = 21.5
    ghi_ref: float = 1000.0
    clip: float = 5.0


def time_to_change(vectors: NDArray) -> NDArray:
    """Time to the next change in the setpoint schedule, by comparing each k to
    k-1 for k >= 2. The first two leads are ignored, as they are always known to the agent.

    Parameters
    ----------
    vectors : NDArray
        Setpoint schedule vectors of shape ``(..., T, H)``.

    Returns
    -------
    NDArray
        Time to the next change in the setpoint schedule, shape ``(..., T)``
    """
    changed = vectors[..., 2:] != vectors[..., 1:-1]
    lead = np.argmax(changed, axis=-1) + 2
    return np.where(changed.any(axis=-1), lead, vectors.shape[-1]).astype(np.float32)


def _shared(m: Measurements, value) -> NDArray:
    return np.full(m.num_houses, value, dtype=np.float32)


def _usable(o: "Observer") -> NDArray:
    return o.params.ev.ev_battery_capacity - o.params.ev.ev_soc_min


def _setpoint_step(m: Measurements, o: "Observer") -> NDArray:
    horizon = m.setpoint.shape[-1]
    lead = o.setpoint_lead.astype(int)
    after = np.take_along_axis(m.setpoint, np.minimum(lead, horizon - 1)[:, None], 1)[
        :, 0
    ]
    return np.where(lead < horizon, (after - m.setpoint[:, 1]) / o.delta_temp, 0.0)


def _min_price_before_departure(m: Measurements, o: "Observer") -> NDArray:
    horizon = m.price.shape[0]
    n = np.clip(m.time_to_departure, 1, horizon).astype(int)
    in_window = np.arange(horizon)[None, :] < n[:, None]
    lowest = np.where(in_window, m.price[None, :], np.inf).min(axis=1)
    return m.ev_availability * lowest / o.scales.price_ref


FEATURES: dict[str, Callable[[Measurements, "Observer"], NDArray]] = {
    "hour_sin": lambda m, o: _shared(m, np.sin(2 * np.pi * m.hour / 24)),
    "hour_cos": lambda m, o: _shared(m, np.cos(2 * np.pi * m.hour / 24)),
    "is_weekend": lambda m, o: _shared(m, float(m.dayofweek >= 5)),
    "p_imp": lambda m, o: _shared(m, m.price[0] / o.scales.price_ref),
    "p_exp": lambda m, o: _shared(m, m.export_price[0] / o.scales.price_ref),
    "p_min_12h": lambda m, o: _shared(m, m.price[1:13].min() / o.scales.price_ref),
    "p_max_12h": lambda m, o: _shared(m, m.price[1:13].max() / o.scales.price_ref),
    "p_trend_3h": lambda m, o: _shared(
        m, (m.price[1:4].mean() - m.price[0]) / o.scales.price_ref
    ),
    "net_exo_load": lambda m, o: (m.load[:, 0] - m.pv[:, 0]) / o.scales.net_load_ref,
    "net_exo_load_mean_5h": lambda m, o: (
        (m.load[:, 1:6] - m.pv[:, 1:6]).mean(axis=1) / o.scales.net_load_ref
    ),
    "error": lambda m, o: (m.setpoint[:, 1] - m.indoor_temp) / o.delta_temp,
    "error_int": lambda m, o: o.errors.integral / (o.delta_temp * o.window),
    "error_der": lambda m, o: o.errors.derivative / o.delta_temp,
    "dText": lambda m, o: (m.setpoint[:, 1] - m.external_temp[0]) / o.scales.temp_ref,
    "dText_mean_6h": lambda m, o: (
        (m.setpoint[:, 2:8].mean(axis=1) - m.external_temp[1:7].mean())
        / o.scales.temp_ref
    ),
    "solar_ghi": lambda m, o: _shared(m, m.solar_ghi[0] / o.scales.ghi_ref),
    "solar_ghi_mean_6h": lambda m, o: _shared(
        m, m.solar_ghi[1:7].mean() / o.scales.ghi_ref
    ),
    "sp_step": _setpoint_step,
    "t_sp_change": lambda m, o: (
        np.minimum(o.setpoint_lead, o.scales.setpoint_horizon)
        / o.scales.setpoint_horizon
    ),
    "a_th_prev": lambda m, o: m.previous_thermal_action,
    "ev_avail": lambda m, o: m.ev_availability,
    "ev_soc_frac": lambda m, o: (m.ev_soc - o.params.ev.ev_soc_min) / _usable(o),
    "ev_soc_req_frac": lambda m, o: (
        m.ev_availability
        * np.clip((m.required_soc - o.params.ev.ev_soc_min) / _usable(o), 0.0, 1.2)
    ),
    "ev_time_to_dep": lambda m, o: (
        m.ev_availability
        * np.minimum(m.time_to_departure, m.price.shape[0])
        / o.scales.hours_ref
    ),
    "ev_charge_hours_left": lambda m, o: (
        m.ev_availability
        * np.maximum(m.required_soc - m.ev_soc, 0.0)
        / o.params.ev.ev_b_in_max
        / o.scales.hours_ref
    ),
    "p_min_before_dep": _min_price_before_departure,
}
"""Features that change every step: ``name -> f(measurements, observer)``."""


STATIC_FEATURES = {  # name: (getter, low, high) = design range of the population
    "pv_installed_capacity": (lambda p: p.pv.pv_installed_capacity, 2.0, 6.0),
    "ev_usable_capacity": (
        lambda p: p.ev.ev_battery_capacity - p.ev.ev_soc_min,
        24.0,
        76.0,
    ),
    "ev_b_in_max": (lambda p: p.ev.ev_b_in_max, 4.0, 11.0),
    "ev_b_out_max": (lambda p: p.ev.ev_b_out_max, 3.0, 10.0),
    "ev_rt_eff": (
        lambda p: p.ev.ev_ch_efficiency * p.ev.ev_disch_efficiency,
        0.81,
        0.93,
    ),
    "hp_rated_power": (lambda p: p.thermal.hp_rated_power, 3.0, 8.0),
    "hp_cop": (lambda p: p.thermal.hp_cop, 2.8, 4.5),
    "building_thermal_capacitance": (
        lambda p: p.thermal.building_thermal_capacitance,
        1.9,
        2.1,
    ),
    "building_ua_coeff": (
        lambda p: 1.0 / p.thermal.building_thermal_resistance,
        0.095,
        0.11,
    ),
    "building_solar_gain": (lambda p: p.thermal.building_solar_gain, 0.0018, 0.00222),
}
"""House parameters, scaled from their population range to ``[-1, 1]``."""

_SCHEDULE_FEATURES = frozenset(("sp_step", "t_sp_change"))


@dataclass(frozen=True)
class ObserverSpec:
    """Everything that defines the observations of a policy; save it with the policy.

    Attributes
    ----------
    names : tuple[str, ...]
        Features, in observation order (keys of ``FEATURES`` or
        ``STATIC_FEATURES``).
    delta_temp : float
        Half-width of the comfort band (°C), the scale of the error features.
    scales : FeatureScales
        Scales of the other features.
    window : int, default=DEFAULT_ERROR_WINDOW
        Steps in the integral of the tracking error.
    """

    names: tuple[str, ...]
    delta_temp: float
    scales: FeatureScales = field(default_factory=FeatureScales)
    window: int = DEFAULT_ERROR_WINDOW

    def __post_init__(self):
        object.__setattr__(self, "names", tuple(self.names))
        unknown = sorted(set(self.names) - set(FEATURES) - set(STATIC_FEATURES))
        if unknown:
            raise ValueError(
                f"Requested observations {unknown} are not implemented; available: "
                f"{sorted(set(FEATURES) | set(STATIC_FEATURES))}"
            )
        if self.delta_temp <= 0 or self.window <= 0:
            raise ValueError("delta_temp and window must be positive.")

    @property
    def size(self) -> int:
        return len(self.names)

    @classmethod
    def from_env_config(cls, config: dict) -> "ObserverSpec":
        """Read the spec from a marl_der environment configuration."""
        return cls(
            names=tuple(n for n, f in config["observations"].items() if f["active"]),
            delta_temp=float(config["cost"]["delta_temp"]),
            scales=FeatureScales(**config.get("feature_scales", {})),
            window=int(config.get("error_window", DEFAULT_ERROR_WINDOW)),
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict (see ``from_dict``)."""
        return {**asdict(self), "names": list(self.names)}

    @classmethod
    def from_dict(cls, data: dict) -> "ObserverSpec":
        return cls(**{**data, "scales": FeatureScales(**data.get("scales", {}))})


class Observer:
    """Turn the ``Measurements`` of ``N`` houses into their ``(N, F)`` observations.

    Keeps the tracking-error memory, so call ``observe`` once per step, in order.
    The house parameters are fixed, so their features are computed once.

    Parameters
    ----------
    spec : ObserverSpec
        Features and scales of the observations.
    params : HouseParams
        Parameters of the ``N`` houses, in the order of the measurements.
    """

    def __init__(self, spec: ObserverSpec, params: HouseParams) -> None:
        self.spec = spec
        self.params = params
        self.scales = spec.scales
        self.delta_temp = spec.delta_temp
        self.window = spec.window
        self.tracking = TrackingErrorFilter(spec.window)
        self._static = {
            name: np.atleast_1d((get(params) - (low + high) / 2) / ((high - low) / 2))
            for name, (get, low, high) in STATIC_FEATURES.items()
            if name in spec.names
        }
        self._needs_schedule = bool(_SCHEDULE_FEATURES & set(spec.names))
        self.errors: ErrorState | None = None
        self.setpoint_lead: NDArray | None = None

    def reset(self) -> None:
        """Forget the tracking-error memory (also done when ``step`` goes back)."""
        self.tracking.reset()

    def observe(self, m: Measurements) -> NDArray:
        """Return the observations of ``m``, clipped to ``±scales.clip``."""
        self.errors = self.tracking.update(m.step, m.tracking_error)
        if self._needs_schedule:
            self.setpoint_lead = time_to_change(m.setpoint)
        columns = [
            self._static[name] if name in self._static else FEATURES[name](m, self)
            for name in self.spec.names
        ]
        clip = self.scales.clip
        return np.clip(np.stack(columns, axis=-1), -clip, clip).astype(np.float32)
