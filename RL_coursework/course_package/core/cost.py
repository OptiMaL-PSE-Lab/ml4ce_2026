"""Cost or reward of a step."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from course_package.core.params import BaseParams, HouseParams

EV_HASSLE_FACTOR: float = 1.5
EV_BUFFER_STRESS_FACTOR: float = 0.1


@dataclass(frozen=True)
class CostBreakdown(BaseParams):
    """Breakdown of the cost components."""

    grid: NDArray
    battery_degradation: NDArray
    comfort: NDArray
    ev_unmet: NDArray
    exogenous: NDArray
    total: NDArray


@dataclass(frozen=True)
class CostParameters:
    """Parameters to compute cost and related KPIs"""

    delta_temp: float = field(default=1.0)
    k_battery_degradation: float = field(default=0.0)
    k_comfort: float = field(default=0.0)
    k_ev_unmet: float = field(default=0.0)
    k_ev_buffer: float = field(default=0.0)
    price_saved_energy: float = field(default=0.0)
    reward_scale_factor: float = field(default=1.0)
    subtract_exogenous: int = field(default=1)

    def __post_init__(self):
        if self.delta_temp <= 0:
            raise ValueError("delta_temp must be positive.")
        if self.k_battery_degradation < 0:
            raise ValueError("k_battery_degradation must be non-negative.")
        if self.k_comfort < 0:
            raise ValueError("k_comfort must be non-negative.")
        if self.k_ev_unmet < 0:
            raise ValueError("k_ev_unmet must be non-negative.")
        if self.k_ev_buffer < 0:
            raise ValueError("k_ev_buffer must be non-negative.")
        if self.price_saved_energy < 0:
            raise ValueError("price_saved_energy must be non-negative.")
        if self.reward_scale_factor <= 0:
            raise ValueError("reward_scale_factor must be positive.")
        if not isinstance(
            self.subtract_exogenous, int
        ) or self.subtract_exogenous not in (0, 1):
            raise ValueError(
                "subtract_exogenous must be an integer of 1=True or 0=False."
            )

    @classmethod
    def from_config(cls, config: dict) -> "CostParameters":
        """Create a CostParameters object from a config dictionary."""
        k_ev_unmet = config["price_public_ev_charge"] * EV_HASSLE_FACTOR
        k_ev_buffer = k_ev_unmet * EV_BUFFER_STRESS_FACTOR
        return CostParameters(
            k_battery_degradation=config["k_battery_degradation"],
            k_comfort=config["k_comfort"],
            k_ev_unmet=k_ev_unmet,
            k_ev_buffer=k_ev_buffer,
            price_saved_energy=config.get("price_saved_energy", 0.0),
            reward_scale_factor=config["reward_scale_factor"],
            delta_temp=config["delta_temp"],
            subtract_exogenous=int(config["subtract_exogenous"]),
        )


def compute_net_import(
    domestic_load: NDArray,
    pv_generation: NDArray,
    b_in: NDArray,
    b_out: NDArray,
    heating_load: NDArray,
    cooling_load: NDArray,
    p: HouseParams,
) -> NDArray:
    return (
        domestic_load
        - pv_generation
        + b_in / p.ev.ev_ch_efficiency
        - b_out * p.ev.ev_disch_efficiency
        + (heating_load + cooling_load) / p.thermal.hp_cop
    )


def grid_cost(
    net_import: NDArray,
    price_import: float,
    price_export: float,
    delta_t: float = 1.0,
) -> NDArray:
    """Compute the grid cost [MU]."""

    return (
        price_import * np.maximum(net_import, 0.0)
        - price_export * np.maximum(-net_import, 0.0)
    ) * delta_t


def comfort_violation(
    indoor_temp_next: NDArray, setpoint_next: NDArray, delta_temp: float
) -> NDArray:
    """Compute the comfort violation [K] resulting from the action applied at time ``t``."""
    return np.maximum(np.abs(indoor_temp_next - setpoint_next) - delta_temp, 0.0)


def ev_unmet_demand(
    required_soc: NDArray, soc_next: NDArray, time_to_departure: NDArray
) -> NDArray:
    """Compute the EV unmet demand [kWh]. Non-zero only if the EV is departing at the next timestep."""
    return np.maximum(required_soc - soc_next, 0.0) * (time_to_departure == 1.0).astype(
        float
    )


def cost_function(
    domestic_load: NDArray,
    pv_generation: NDArray,
    b_in: NDArray,
    b_out: NDArray,
    heating_load: NDArray,
    cooling_load: NDArray,
    indoor_temp_next: NDArray,
    setpoint_next: NDArray,
    required_soc: NDArray,
    soc_next: NDArray,
    time_to_departure: NDArray,
    price_import: float,
    price_export: float,
    house_param: HouseParams,
    cost_param: CostParameters,
    delta_t: float = 1.0,
) -> CostBreakdown:
    """Compute the cost breakdown."""
    net_import = compute_net_import(
        domestic_load,
        pv_generation,
        b_in,
        b_out,
        heating_load,
        cooling_load,
        house_param,
    )
    exogenous_import = domestic_load - pv_generation
    grid = grid_cost(net_import, price_import, price_export, delta_t)
    exogenous = grid_cost(exogenous_import, price_import, price_export, delta_t)
    battery_degradation = cost_param.k_battery_degradation * (b_in + b_out)
    comfort = (
        cost_param.k_comfort
        * comfort_violation(indoor_temp_next, setpoint_next, cost_param.delta_temp)
        * delta_t
    )
    ev_unmet = cost_param.k_ev_unmet * ev_unmet_demand(
        required_soc, soc_next, time_to_departure
    )

    total = grid + battery_degradation + comfort + ev_unmet

    return CostBreakdown(
        grid=grid,
        battery_degradation=battery_degradation,
        comfort=comfort,
        ev_unmet=ev_unmet,
        exogenous=exogenous,
        total=total,
    )


def reward(breakdown: CostBreakdown, cost_param: CostParameters) -> NDArray:
    """Return the reward of a step: the scaled negative cost, net of the exogenous
    cost when ``subtract_exogenous`` is set."""
    return -cost_param.reward_scale_factor * (
        breakdown.total - breakdown.exogenous * cost_param.subtract_exogenous
    )
