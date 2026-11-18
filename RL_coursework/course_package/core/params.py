"""Static parameters of the households, one array entry per house."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

HOUSE_SPECS = (
    "building_thermal_capacitance",
    "building_thermal_resistance",
    "building_solar_gain",
    "hp_rated_power",
    "hp_cop",
    "ev_b_in_max",
    "ev_b_out_max",
    "ev_battery_capacity",
    "ev_ch_efficiency",
    "ev_consumption",
    "ev_disch_efficiency",
    "ev_battery_soc_min_fraction",
    "pv_installed_capacity",
)
"""Device specifications of a house, named as in the HP2CDT setup files."""


@dataclass(frozen=True)
class BaseParams:
    """Base class for parameters."""

    def agent(self, agent_idx: int):
        """Return the parameters for a specific agent."""
        for field in self.__dataclass_fields__:
            value = getattr(self, field)[agent_idx]
            yield field, value

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EVParams(BaseParams):
    """Parameters for the electric vehicle (EV) model.

    Attributes:
    -----------
        ch_efficiency: Charging efficiency of the EV battery [-].
        disch_efficiency: Discharging efficiency of the EV battery [-].
        b_in_max: Maximum charging power of the EV battery [kW].
        b_out_max: Maximum discharging power of the EV battery [kW].
        capacity: Gross capacity of the EV battery [kWh].
        soc_min_fraction: Energy reserve between gross and usable capacity to prevent
            degradation, controlled by car's software [-].
        consumption: Energy consumption of the EV per unit distance [kWh/100km].
        soc_min: Minimum state of charge (SOC) of the EV battery [kWh].
    """

    ev_b_in_max: NDArray
    ev_b_out_max: NDArray
    ev_battery_capacity: NDArray
    ev_ch_efficiency: NDArray
    ev_consumption: NDArray
    ev_disch_efficiency: NDArray
    soc_min_fraction: NDArray

    def __post_init__(self):
        unitary_fields = ["ev_ch_efficiency", "ev_disch_efficiency", "soc_min_fraction"]
        positive_fields = [
            "ev_b_in_max",
            "ev_b_out_max",
            "ev_battery_capacity",
            "ev_consumption",
        ]
        for field in positive_fields:
            if np.any(getattr(self, field) < 0):
                raise ValueError(f"EV {field} must be non-negative.")
        for field in unitary_fields:
            if np.any(getattr(self, field) <= 0) or np.any(getattr(self, field) > 1):
                raise ValueError(f"EV {field} must be in (0, 1].")
        if np.any(self.ev_soc_min > self.ev_battery_capacity):
            raise ValueError("EV soc_min must be less than or equal to capacity.")

    @cached_property
    def ev_soc_min(self) -> NDArray:
        """Compute the minimum state of charge [kWh]."""
        return self.ev_battery_capacity * self.soc_min_fraction


@dataclass(frozen=True)
class ThermalModelParams(BaseParams):
    """Parameters for the thermal model."""

    building_thermal_capacitance: NDArray
    building_thermal_resistance: NDArray
    building_solar_gain: NDArray
    hp_rated_power: NDArray
    hp_cop: NDArray

    def __post_init__(self):
        if np.any(self.building_thermal_capacitance <= 0):
            raise ValueError("Thermal capacitance must be positive.")
        if np.any(self.building_thermal_resistance <= 0):
            raise ValueError("Thermal resistance must be positive.")
        if np.any(self.hp_rated_power <= 0):
            raise ValueError("Heat pump rated power must be positive.")
        if np.any(self.hp_cop <= 0):
            raise ValueError("Heat pump COP must be positive.")


@dataclass(frozen=True)
class PVParams(BaseParams):
    """Parameters for the photovoltaic (PV) model."""

    pv_installed_capacity: NDArray

    def __post_init__(self):
        if np.any(self.pv_installed_capacity < 0):
            raise ValueError("PV installed capacity must be positive.")


@dataclass(frozen=True)
class HouseParams:
    """Parameters characterizing each household."""

    thermal: ThermalModelParams
    ev: EVParams
    pv: PVParams

    @classmethod
    def from_config(cls, config: dict) -> "HouseParams":
        """Create a HouseParams object from a config dictionary.

        ``config`` maps each house to a dict whose ``"specs"`` hold the device
        specifications of the HP2CDT setup files.
        """
        params = cls.read_config(config)
        thermal_params = ThermalModelParams(
            building_thermal_capacitance=params["building_thermal_capacitance"],
            building_thermal_resistance=params["building_thermal_resistance"],
            building_solar_gain=params["building_solar_gain"],
            hp_rated_power=params["hp_rated_power"],
            hp_cop=params["hp_cop"],
        )
        ev_params = EVParams(
            ev_b_in_max=params["ev_b_in_max"],
            ev_b_out_max=params["ev_b_out_max"],
            ev_battery_capacity=params["ev_battery_capacity"],
            ev_ch_efficiency=params["ev_ch_efficiency"],
            ev_consumption=params["ev_consumption"],
            ev_disch_efficiency=params["ev_disch_efficiency"],
            soc_min_fraction=params["ev_battery_soc_min_fraction"],
        )
        pv_params = PVParams(pv_installed_capacity=params["pv_installed_capacity"])
        return HouseParams(thermal=thermal_params, ev=ev_params, pv=pv_params)

    @classmethod
    def from_specs(cls, specs: Mapping[str, float]) -> "HouseParams":
        """Create the parameters of one house from its device specifications, named
        as in ``HOUSE_SPECS``; other entries are ignored.

        Raises
        ------
        KeyError
            If a specification is missing.
        """
        missing = [name for name in HOUSE_SPECS if name not in specs]
        if missing:
            raise KeyError(f"Missing device specs: {missing}")
        return cls.from_config({"house": {"specs": {k: specs[k] for k in HOUSE_SPECS}}})

    @staticmethod
    def read_config(config: dict) -> dict:
        param_names = next(iter(config.values())).get("specs", {}).keys()
        param_values = {name: [] for name in param_names}
        for prosumer_id, prosumer_config in config.items():
            specs = prosumer_config.get("specs", {})
            if not specs:
                raise ValueError(f"Missing 'specs' in `{prosumer_id}` config.")
            for name in param_names:
                param_values[name].append(specs.get(name))
        for name in param_names:
            param_values[name] = np.array(param_values[name], dtype=np.float32)
        return param_values

    def agent(self, agent_idx: int):
        """Return the parameters for a specific agent."""
        return HouseParams(
            thermal=ThermalModelParams(**dict(self.thermal.agent(agent_idx))),
            ev=EVParams(**dict(self.ev.agent(agent_idx))),
            pv=PVParams(**dict(self.pv.agent(agent_idx))),
        )
