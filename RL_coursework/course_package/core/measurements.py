"""Information accessible to a house at decision time."""

from collections.abc import Mapping
from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import ArrayLike, NDArray

FORECAST_HORIZON = 36  # hours; length of the information vector
LOOKBACK_PERIOD = 24  # hours
FLOW_LAG = 1  # hours; PV or domestic demand readings are only known after ``FLOW_LAG`` hours have passed

PRICE_STREAMS = ("import_cost", "export_price")  # import_cost includes the carbon cost
FORECAST_STREAMS = ("external_temp", "solar_ghi", "pv_generation", "domestic_load")

READINGS = (
    "hour",
    "dayofweek",
    "month",
    "indoor_temp",
    "ev_soc",
    "ev_availability",
    "ev_time_to_departure",
    "ev_required_soc_departure",
)
"""Device readings used, named as in the HP2CDT measurement schema."""

SHARED_FORECASTS = ("import_cost", "export_price", "external_temp", "solar_ghi")
"""Information vectors common to all houses, shape ``(H,)``."""

HOUSE_FORECASTS = ("indoor_temp_setpoint", "domestic_load", "pv_generation")
"""Information vectors of each house, shape ``(N, H)``."""

_HOUSE_FIELDS = frozenset(
    ("indoor_temp", "previous_thermal_action", "ev_soc", "ev_availability")
    + ("time_to_departure", "required_soc", "setpoint", "load", "pv")
)
"""Fields with one entry per house (an ``(N,)`` and an ``(H,)`` array cannot be
told apart by shape when ``N == H``)."""


@dataclass(frozen=True)
class Measurements:
    """Information of ``N`` houses at one decision step.

    Information vectors hold leads ``k = 0 .. H - 1`` hours ahead, ``k = 0`` being
    the current hour; ``setpoint[:, 1]`` is the target of the current action.

    Attributes
    ----------
    step : int
        Decision step since the start of the episode (environment) or of the
        operation (edge); it orders the updates of the tracking-error memory.
    hour, dayofweek, month : int
        Local time of the decision.
    indoor_temp : NDArray
        Indoor temperature (°C), ``(N,)``.
    previous_thermal_action : NDArray
        Normalized thermal command of the previous step (0 at the first), ``(N,)``.
    ev_soc, ev_availability, time_to_departure, required_soc : NDArray
        EV state of charge (kWh), plugged-in flag, hours to the declared departure
        and the SoC declared for it (kWh, 0 while away), ``(N,)``.
    setpoint, load, pv : NDArray
        Thermostat schedule (°C), domestic load and PV generation (kW), ``(N, H)``.
    price, export_price, external_temp, solar_ghi : NDArray
        Import price with carbon cost and export price (per kWh), outdoor
        temperature (°C) and global horizontal irradiance (W/m²), ``(H,)``.
    """

    step: int
    hour: int
    dayofweek: int
    month: int
    indoor_temp: NDArray
    previous_thermal_action: NDArray
    ev_soc: NDArray
    ev_availability: NDArray
    time_to_departure: NDArray
    required_soc: NDArray
    setpoint: NDArray
    load: NDArray
    pv: NDArray
    price: NDArray
    export_price: NDArray
    external_temp: NDArray
    solar_ghi: NDArray

    @classmethod
    def from_readings(
        cls,
        step: int,
        readings: Mapping[str, ArrayLike],
        forecasts: Mapping[str, ArrayLike],
        previous_thermal_action: ArrayLike,
    ) -> "Measurements":
        """Build the measurements of one step.

        Parameters
        ----------
        step : int
            Decision step (see ``Measurements.step``).
        readings : Mapping[str, ArrayLike]
            Device readings named as in ``READINGS``, one value per house; extra
            readings are ignored.
        forecasts : Mapping[str, ArrayLike]
            Information vectors named as in ``SHARED_FORECASTS``, shape ``(H,)``,
            and ``HOUSE_FORECASTS``, shape ``(N, H)``.
        previous_thermal_action : ArrayLike
            Thermal command of the previous step, one value per house.

        Raises
        ------
        KeyError
            If a reading or a forecast is missing.
        """
        missing = [name for name in READINGS if name not in readings] + [
            name for name in SHARED_FORECASTS + HOUSE_FORECASTS if name not in forecasts
        ]
        if missing:
            raise KeyError(f"Missing readings or forecasts: {missing}")

        def house(name: str) -> NDArray:
            return np.atleast_1d(np.asarray(readings[name], dtype=np.float32))

        def clock(name: str) -> int:
            return int(np.asarray(readings[name]).reshape(-1)[0])

        def vector(name: str, ndim: int) -> NDArray:
            value = np.asarray(forecasts[name], dtype=np.float32)
            return value[None, :] if ndim == 2 and value.ndim == 1 else value

        availability = house("ev_availability")
        return cls(
            step=int(step),
            hour=clock("hour"),
            dayofweek=clock("dayofweek"),
            month=clock("month"),
            indoor_temp=house("indoor_temp"),
            previous_thermal_action=np.atleast_1d(
                np.asarray(previous_thermal_action, dtype=np.float32)
            ),
            ev_soc=house("ev_soc"),
            ev_availability=availability,
            time_to_departure=house("ev_time_to_departure"),
            required_soc=house("ev_required_soc_departure") * availability,
            setpoint=vector("indoor_temp_setpoint", 2),
            load=vector("domestic_load", 2),
            pv=vector("pv_generation", 2),
            price=vector("import_cost", 1),
            export_price=vector("export_price", 1),
            external_temp=vector("external_temp", 1),
            solar_ghi=vector("solar_ghi", 1),
        )

    @property
    def num_houses(self) -> int:
        return self.indoor_temp.shape[0]

    @property
    def tracking_error(self) -> NDArray:
        """Return ``setpoint - temperature`` for the target of the current action."""
        return self.setpoint[:, 1] - self.indoor_temp

    def agent(self, agent_idx: int) -> "Measurements":
        """Return the measurements of one house (``N = 1``)."""
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        for name in _HOUSE_FIELDS:
            values[name] = values[name][agent_idx : agent_idx + 1]
        return Measurements(**values)


def build_price_forecast(
    price: NDArray,
    known_horizon: NDArray,
    horizon: int = FORECAST_HORIZON,
    lookback_period: int = LOOKBACK_PERIOD,
) -> NDArray:
    """Return price vectors. If k < known_horizon, the published price is used.
    If k >= known_horizon, it repeates the price of the same hour of k - lookback_period.

    Parameters
    ----------
    price : NDArray
        Hourly price series of shape ``(T,)``.
    known_horizon : NDArray
        Number of published hourly prices known at each hour, shape ``(T,)``.
    horizon : int, default=FORECAST_HORIZON
        Length each price vector must have.
    lookback_period : int, default=LOOKBACK_PERIOD
        Number of hours to look back to retrieve prices not yet available.

    Returns
    -------
    NDArray
        A 2D array of shape ``(T, horizon)``.

    Note
    ----
        ``price`` and ``known_horizon`` are read from Agile's published price file,
        which is updated daily around 16:00 local time.
    """
    price = np.asarray(price, dtype=np.float32)
    known_horizon = np.asarray(known_horizon, dtype=np.int64)
    if price.shape != known_horizon.shape or price.ndim != 1:
        raise ValueError("price and known_horizon must be 1-D arrays of one length.")
    if known_horizon.min() < 1:
        raise ValueError("At least the current price must be known.")
    t = np.arange(price.size)[:, None]
    lead = np.arange(horizon)[None, :]
    hours_past_window = np.maximum(lead - (known_horizon[:, None] - 1), 0)
    days_back = -(-hours_past_window // lookback_period)
    source = np.clip(t + lead - lookback_period * days_back, 0, price.size - 1)
    return price[source]


def build_schedule_vectors(
    schedule: NDArray, horizon: int = FORECAST_HORIZON
) -> NDArray:
    """Return the vectors of a schedule known in advance, e.g. the thermostat program.

    Parameters
    ----------
    schedule : NDArray
        Hourly series of shape ``(T,)``.
    horizon : int, default=FORECAST_HORIZON
        Length each vector must have.

    Returns
    -------
    NDArray
        Read-only array of shape ``(T, horizon)``. Leads past the end of the
        series repeat its last value.
    """
    schedule = np.asarray(schedule, dtype=np.float32)
    if schedule.ndim != 1 or schedule.size == 0:
        raise ValueError("schedule must be a non-empty 1-D array.")
    padded = np.concatenate([schedule, np.repeat(schedule[-1:], horizon - 1)])
    return np.lib.stride_tricks.sliding_window_view(padded, horizon)
