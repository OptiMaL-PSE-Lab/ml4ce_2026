"""
MARL environment for distributed energy resource (DER) optimization
in households with HVAC and EV charging systems.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from numpy.typing import NDArray

from course_package.core import HouseParams, Measurements, Observer, ObserverSpec
from course_package.core.cost import (
    CostBreakdown,
    CostParameters,
    compute_net_import,
    cost_function,
    reward,
)
from course_package.core.measurements import (
    FLOW_LAG,
    FORECAST_HORIZON,
    HOUSE_FORECASTS,
    PRICE_STREAMS,
    SHARED_FORECASTS,
    build_price_forecast,
    build_schedule_vectors,
)
from course_package.environment.constants import (
    DEFAULT_EV_INITIAL_SOC_RANGE,
    DEFAULT_LOCAL_TZ,
)
from course_package.environment.forecasting import (
    read_forecast_file,
    resolve_forecast_path,
)

# pylint: disable=no-member
# TODO: omit whatever starting from eff from template

MultiAgentData = dict[str, dict[str, dict[str, NDArray]]]


WEATHER_COLUMNS = ("external_temp", "solar_ghi")
SCENARIO_KEYS = {
    "start_date",
    "end_date",
    "episode_len",
    "local_tz",
    "episode_starts",
    "episode_sampling",
    "episode_sampling_seed",
    "ev_initial_soc_range",
}
PROSUMER_COLUMNS = (
    "domestic_load",
    "pv_generation",
    "indoor_temp_setpoint",
    "ev_availability",
    "ev_trip_demand",
    "ev_time_to_departure",
    "ev_required_soc_departure",
)


def _require_columns(data: pd.DataFrame, columns: tuple[str, ...], source) -> None:
    """Raise if ``data`` lacks any of ``columns``."""
    missing = [column for column in columns if column not in data]
    if missing:
        raise ValueError(
            f"{source} lacks {missing}. Regenerate the dataset with "
            "data/run_data_generation.py."
        )


class DERMultiAgentEnv(gym.Env):
    """Multi-agent DER optimization environment.

    Parameters
    ----------
    env_config : dict[str, Any]
        Environment configuration.
    seed : Optional[int], default=None
        Seed for reproducible episode sampling.

    Attributes
    ----------
    action_space : spaces.Dict
        Joint agent/device/action space for all prosumers.
    observation_space : spaces.Dict
        Joint agent/device/observation space for all prosumers.
    """

    def __init__(self, env_config: dict[str, Any], seed: int | None = None) -> None:
        """Initialize the multi-agent DER environment.

        Parameters
        ----------
        env_config : dict[str, Any]
            Environment configuration.
        seed : int | None, default=None
            Seed for reproducible episode sampling.
        """
        super().__init__()

        self._parse_common_data(env_config)
        self._parse_prosumer_data(env_config)

        self._load_forecasting_data(env_config["data"]["forecasting"])

        self._setup_episode_boundaries()
        if self.episode_sampling not in {"random", "sequential"}:
            raise ValueError("episode_sampling must be either 'random' or 'sequential'")
        self._episode_start_cursor = 0

        self.action_space = self._construct_action_space(
            schema=self.action_space_schema, agent_ids=self.prosumer_ids, seed=seed
        )
        self.action_devices = {
            action: device
            for device, actions in self.action_space_schema.items()
            for action in actions
        }
        self.num_actions = sum(len(v) for v in self.action_space_schema.values())

        self.observer = Observer(self.observer_spec, self.param)
        clip, n_features = self.observer_spec.scales.clip, self.observer_spec.size
        self.observation_space = spaces.Dict(
            OrderedDict(
                (agent_id, spaces.Box(-clip, clip, (n_features,), np.float32))
                for agent_id in self.prosumer_ids
            ),
            seed=seed,
        )

        self.reset(seed=seed)

    @staticmethod
    def _construct_composite_space(
        schema: dict[str, list],
        bounds: dict,
    ) -> spaces.Dict[str, spaces.Dict[str, spaces.Box]]:
        """Construct a composite space from a dictionary of spaces.

        Parameters
        ----------
        schema : dict[str, list]
            spaces.Dictionary mapping keys to lists of spaces.
        bounds : dict[str, tuple[NDArray, NDArray]]
            spaces.Dictionary mapping keys to observation/action bounds.

        Returns
        -------
        spaces.Dict[str, spaces.Dict[str, spaces.Box]]
            spaces.Dictionary mapping keys to composite spaces.
        """
        composite_space = OrderedDict()
        for group, values in schema.items():
            device_space = OrderedDict()
            for value in values:
                low, high = bounds[value]
                device_space[value] = spaces.Box(
                    low=np.asarray([low], dtype=np.float32),
                    high=np.asarray([high], dtype=np.float32),
                    dtype=np.float32,
                )
            composite_space[group] = spaces.Dict(device_space)
        return spaces.Dict(composite_space)

    @staticmethod
    def _construct_action_space(
        schema: dict[str, list], agent_ids: tuple[str, ...], seed: int | None
    ) -> spaces.Dict:
        """Construct a composite action space from a mapping of device -> action keys."""
        bounds = {name: (-1.0, 1.0) for names in schema.values() for name in names}
        agent_spaces = OrderedDict(
            (
                agent_id,
                DERMultiAgentEnv._construct_composite_space(schema, bounds),
            )
            for agent_id in agent_ids
        )
        return spaces.Dict(agent_spaces, seed=seed)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, NDArray], dict]:
        """
        Reset the environment and sample a new episode boundary.

        Parameters
        ----------
        seed : int | None, default=None
            Optional reset seed.
        options : dict | None, default=None
            Optional reset options, including ``episode_start``.

        Returns
        -------
        tuple[dict[str,NDArray], dict]
            Initial observations and info dictionary.
        """
        super().reset(seed=seed)

        episode_start = (options or {}).get("episode_start")

        if episode_start is None:
            if self.episode_sampling == "sequential":
                panel_idx = self._episode_start_cursor % len(self.valid_episode_starts)
                episode_start = int(self.valid_episode_starts[panel_idx])
                self._episode_start_cursor += 1
                if self.episode_sampling_seed is not None:
                    super().reset(seed=int(self.episode_sampling_seed) + panel_idx)
            else:
                episode_start = int(self.np_random.choice(self.valid_episode_starts))
        else:
            episode_start = int(episode_start)
            assert episode_start in self.valid_episode_starts, (
                f"Invalid episode_start {episode_start}."
            )

        self.current_episode_start = episode_start
        self.current_step = 0
        self.current_step_global = self.current_episode_start + self.current_step
        self.terminateds = {"__all__": False}

        # Actions
        self.actions_norm = np.zeros(
            (self.num_prosumers, self.episode_len, self.num_actions),
            dtype=np.float32,
        )
        self.last_action = np.zeros(
            (self.num_prosumers, self.num_actions), dtype=np.float32
        )
        self.last_action_fresh = np.zeros(self.num_prosumers, dtype=bool)
        self.pros_import = np.zeros(
            (self.num_prosumers, self.episode_len), dtype=np.float32
        )
        self.district_import = np.zeros((self.episode_len), dtype=np.float32)
        self.thermal_load = np.zeros_like(self.pros_import)
        self.heating_load = np.zeros_like(self.pros_import)
        self.cooling_load = np.zeros_like(self.pros_import)
        self.b_in = np.zeros_like(self.pros_import)
        self.b_out = np.zeros_like(self.pros_import)

        # States: battery
        self.ev_soc = np.zeros(
            (self.num_prosumers, self.episode_len + 1), dtype=np.float32
        )
        self.soc_infeas_flag = np.zeros_like(self.pros_import).astype(bool)
        self.soc_violation = np.zeros_like(self.pros_import)
        self.ev_charge_override = np.zeros_like(self.pros_import, dtype=bool)

        # States: temperature
        self.indoor_temp = np.zeros(
            (self.num_prosumers, self.episode_len + 1), dtype=np.float32
        )
        self.indoor_temp[:, 0] = np.array(
            [
                self.np_random.uniform(
                    low=self.indoor_temp_setpoint[i, self.current_episode_start]
                    - self.cost_param.delta_temp,
                    high=self.indoor_temp_setpoint[i, self.current_episode_start]
                    + self.cost_param.delta_temp,
                )
                for i in range(self.num_prosumers)
            ],
            dtype=np.float32,
        )
        self.temp_infeas_flag = np.zeros_like(self.soc_infeas_flag)
        self.ev_soc[:, 0] = self._sample_initial_ev_soc()
        self.hvac_mode = np.zeros_like(self.pros_import).astype(np.int8)

        self._update_hvac_mode()
        self.observer.reset()
        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: MultiAgentData | NDArray
    ) -> tuple[dict[str, NDArray], NDArray, bool, bool, dict]:
        """Advance the environment using nested or legacy matrix actions."""
        reward, terminated, truncated, info = self.advance_one_simulation_step(action)
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the environment.

        Raises
        ------
        NotImplementedError
            Always raised because rendering is not implemented.
        """
        raise (
            NotImplementedError("Render method not implemented for DERMultiAgentEnv.")
        )

    def advance_one_simulation_step(
        self,
        action: MultiAgentData | NDArray | None,
        action_mask: NDArray | None = None,
    ) -> tuple[NDArray, bool, bool, dict]:
        """Apply the action and advance the environment.

        Parameters
        ----------
        action : MultiAgentData | NDArray | None
            Nested action dictionary for all prosumers. Joint action arrays remain
            supported for internal callers.

        Returns
        -------
        tuple[NDArray, bool, bool, dict]
            Reward vector, termination flag, truncation flag and info.
        """
        # TODO: decouple advance_one_simulation_step to step real-time evolution counter or even step_counter. What is a step in asynchronous settings?
        if action is None:
            action_array = self.last_action
        else:
            self._validate_action(action)
            action_array = self._decode_action(action)

        self.last_action = action_array
        if action_mask is None:
            self.last_action_fresh = np.ones(self.num_prosumers, dtype=bool)
        else:
            if action_mask.shape != (self.num_prosumers,):
                raise ValueError(
                    f"Action mask shape {action_mask.shape} does not match expected shape ({self.num_prosumers},)."
                )
            self.last_action_fresh = action_mask.astype(bool, copy=False)

        self._translate_action(action_array)
        self._compute_energy_balance()
        self._update_temperature()
        self._update_battery_soc()
        reward, breakdown = self._compute_reward()

        self.current_step += 1
        self.current_step_global = self.current_episode_start + self.current_step
        truncated = self.current_step >= self.episode_len
        if not truncated:
            self._update_hvac_mode()
        terminated = False
        applied_step = self.current_step - 1

        a_b = self.actions_norm[:, applied_step, 1]
        applied_battery_power = (
            self.b_in[:, applied_step] - self.b_out[:, applied_step]
        ).copy()
        commanded = (
            np.maximum(a_b, 0) * self.param.ev.ev_b_in_max
            - np.maximum(-a_b, 0) * self.param.ev.ev_b_out_max
        )
        info = {
            "episode_start_idx": self.current_episode_start,
            "commanded_action": self.actions_norm[:, applied_step].copy(),
            "applied_thermal_load": self.thermal_load[:, applied_step].copy(),
            "applied_battery_power": applied_battery_power,
            "ev_forced_charge": self.ev_charge_override[:, applied_step].copy(),
            "ev_correction": applied_battery_power - commanded,
            "current_step": self.current_step,
            "current_step_global": self.current_step_global,
            "action_fresh": self.last_action_fresh.copy(),
            "applied_step": applied_step,
            "cost": breakdown.to_dict(),
        }
        info["cost"].pop("exogenous")

        return reward, terminated, truncated, info

    def get_measurements(self) -> Measurements:
        """Return what the houses know at the current step.

        Built from the device readings, the information vectors of the current
        hour and the previous thermal command, as an HP2CDT edge builds them.
        """
        t, g = self.current_step, min(self.current_step_global, self.history_len - 1)
        forecasts = {name: self.forecasts[name][g] for name in SHARED_FORECASTS} | {
            name: self.forecasts[name][:, g] for name in HOUSE_FORECASTS
        }
        previous_action = (
            self.actions_norm[:, t - 1, 0]
            if t > 0
            else np.zeros(self.num_prosumers, np.float32)
        )
        return Measurements.from_readings(
            t, self.get_readings(), forecasts, previous_action
        )

    def get_readings(self) -> dict[str, NDArray]:
        """Return the current device readings of all houses, one value per house."""
        t, g = self.current_step, min(self.current_step_global, self.history_len - 1)
        last = max(g - FLOW_LAG, 0)

        def shared(x):
            return np.full(self.num_prosumers, x, np.float32)

        def applied(x):  # metered over the previous step, 0 before the first
            return x[:, t - 1] if t > 0 else np.zeros(self.num_prosumers, np.float32)

        return {
            "applied_ev_charge": applied(self.b_in),
            "applied_ev_discharge": applied(self.b_out),
            "applied_heating_load": applied(self.heating_load),
            "applied_cooling_load": applied(self.cooling_load),
            "ev_soc": self.ev_soc[:, t],
            "ev_availability": self.ev_availability[:, g],
            "ev_time_to_departure": self.ev_time_to_departure[:, g],
            "ev_required_soc_departure": self._get_required_ev_soc(t),
            "indoor_temp": self.indoor_temp[:, t],
            "indoor_temp_setpoint": self.indoor_temp_setpoint[
                :, min(g + 1, self.history_len - 1)
            ],
            "hour": shared(self._local_time_index[g].hour),
            "dayofweek": shared(self._local_time_index[g].dayofweek),
            "month": shared(self._local_time_index[g].month),
            "import_cost": shared(self.import_cost[g]),  # published tariff of this hour
            "export_price": shared(self.export_price[g]),
            "domestic_load": self.domestic_load[:, last],
            "pv_generation": self.pv_generation[:, last],
            "external_temp": shared(self.external_temp[g]),
            "solar_ghi": shared(self.solar_ghi[last]),
        }

    def _check_comfort(self) -> NDArray:
        """Check whether current indoor temperatures remain within comfort bounds."""
        current_indoor_temp = self.indoor_temp[:, self.current_step]
        current_setpoint = self.indoor_temp_setpoint[:, self.current_step_global]

        p = self.cost_param
        lower_bound = current_setpoint - p.delta_temp
        upper_bound = current_setpoint + p.delta_temp

        in_comfort = (current_indoor_temp >= lower_bound) & (
            current_indoor_temp <= upper_bound
        )
        return in_comfort

    def _check_ev_soc(self) -> tuple[NDArray, NDArray]:
        """Check whether current EV state of charge can satisfy required demand."""
        current_soc = self.ev_soc[:, self.current_step]

        required_soc = self._get_required_ev_soc(self.current_step)
        time_to_departure = self.ev_time_to_departure[:, self.current_step_global]
        max_charge_after_step = self.param.ev.ev_b_in_max * np.maximum(
            time_to_departure - 1.0, 0.0
        )
        soc_sufficient = current_soc + max_charge_after_step >= required_soc
        return soc_sufficient, required_soc

    def _compute_ev_requirement(self) -> NDArray:
        """Compute minimum charging required to satisfy EV constraints."""
        soc_sufficient_flag, required_soc = self._check_ev_soc()
        current_soc = self.ev_soc[:, self.current_step]
        req_b_in = np.clip(
            required_soc - current_soc,
            0.0,
            self.param.ev.ev_b_in_max
            * self.ev_availability[:, self.current_step_global],
        )
        req_b_in = np.where(soc_sufficient_flag, 0.0, req_b_in)
        return req_b_in

    def _compute_thermal_load_bounds(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        p = self.param.thermal

        current_indoor_temp, next_setpoint = self._get_temp_context(self.current_step)
        external_temp = self.external_temp[self.current_step_global]
        solar_ghi = self.solar_ghi[self.current_step_global]

        lower_bound = next_setpoint - self.cost_param.delta_temp
        upper_bound = next_setpoint + self.cost_param.delta_temp

        too_cold = current_indoor_temp < lower_bound
        too_hot = current_indoor_temp > upper_bound
        comfort = ~(too_cold | too_hot)

        def load_to_target(target_temp):
            return (
                p.building_thermal_capacitance * (target_temp - current_indoor_temp)
                - (external_temp - current_indoor_temp) / p.building_thermal_resistance
                - solar_ghi * p.building_solar_gain
            )

        q_lower = load_to_target(lower_bound)
        q_upper = load_to_target(upper_bound)

        heating_load_lb = np.where(too_cold, q_lower, 0.0)
        heating_load_ub = np.where(too_cold | comfort, q_upper, 0.0)

        cooling_load_lb = np.where(too_hot, q_upper, 0.0)
        cooling_load_ub = np.where(too_hot | comfort, q_lower, 0.0)

        heating_load_lb = np.clip(heating_load_lb, 0.0, p.hp_rated_power)
        heating_load_ub = np.clip(heating_load_ub, 0.0, p.hp_rated_power)
        cooling_load_lb = np.clip(cooling_load_lb, -p.hp_rated_power, 0.0)
        cooling_load_ub = np.clip(cooling_load_ub, -p.hp_rated_power, 0.0)

        return heating_load_lb, cooling_load_lb, heating_load_ub, cooling_load_ub

    def _compute_energy_balance(self) -> None:
        """Update per-prosumer and district electricity import balances."""
        self.pros_import[:, self.current_step] = compute_net_import(
            domestic_load=self.domestic_load[:, self.current_step_global],
            pv_generation=self.pv_generation[:, self.current_step_global],
            b_in=self.b_in[:, self.current_step],
            b_out=self.b_out[:, self.current_step],
            heating_load=self.heating_load[:, self.current_step],
            cooling_load=self.cooling_load[:, self.current_step],
            p=self.param,
        )
        self.district_import[self.current_step] = np.sum(
            self.pros_import[:, self.current_step]
        )

    def _compute_reward(self) -> tuple[NDArray, dict[str, CostBreakdown]]:
        """Compute the weighted per-prosumer reward."""
        t = self.current_step
        t_global = self.current_step_global
        t_global_next = min(t_global + 1, self.history_len - 1)
        breakdown = cost_function(
            domestic_load=self.domestic_load[:, t_global],
            pv_generation=self.pv_generation[:, t_global],
            b_in=self.b_in[:, t],
            b_out=self.b_out[:, t],
            heating_load=self.heating_load[:, t],
            cooling_load=self.cooling_load[:, t],
            indoor_temp_next=self.indoor_temp[:, t + 1],
            setpoint_next=self.indoor_temp_setpoint[:, t_global_next],
            required_soc=self.ev_required_soc_departure[:, t_global],
            soc_next=self.ev_soc[:, t + 1],
            time_to_departure=self.ev_time_to_departure[:, t_global],
            price_import=self.import_cost[t_global],
            price_export=self.export_price[t_global],
            house_param=self.param,
            cost_param=self.cost_param,
            delta_t=self.dt["control_interval_seconds"] / 3600.0,
        )

        return reward(breakdown, self.cost_param).astype(np.float32), breakdown

    def _get_obs(self) -> dict[str, NDArray]:
        features = self.observer.observe(self.get_measurements())
        return OrderedDict(
            (agent_id, features[i]) for i, agent_id in enumerate(self.prosumer_ids)
        )

    def _get_required_ev_soc(self, timestep: int) -> NDArray:
        """Return the SoC required at the next departure, zero while away."""
        idx = min(
            self.current_episode_start + timestep,
            self.ev_required_soc_departure.shape[-1] - 1,
        )
        return self.ev_required_soc_departure[:, idx] * self.ev_availability[:, idx]

    def _sample_initial_ev_soc(self) -> NDArray:
        """Draw the initial EV SoC uniformly in ``ev_initial_soc_range`` of capacity.

        The draw is raised to ``soc_min`` and to the least SoC from which the first
        departure is still reachable at full charging power.
        """
        low, high = self.ev_initial_soc_range
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(
                f"ev_initial_soc_range must satisfy 0 <= low <= high <= 1; "
                f"got {(low, high)}."
            )

        p = self.param.ev
        soc = self.np_random.uniform(low, high, self.num_prosumers)
        soc = soc * p.ev_battery_capacity
        soc = np.maximum(soc, p.ev_soc_min)
        time_to_departure = self.ev_time_to_departure[:, self.current_step_global]
        reachable_from = (
            self._get_required_ev_soc(0) - p.ev_b_in_max * time_to_departure
        )
        soc = np.maximum(soc, reachable_from)
        return np.minimum(soc, p.ev_battery_capacity).astype(np.float32)

    def _get_temp_context(self, timestep: int) -> tuple[NDArray, NDArray]:
        """Return current indoor temperature and next setpoint context."""
        current_indoor_temp = self.indoor_temp[:, timestep]
        next_timestep_global = min(
            self.current_episode_start + timestep + 1,
            self.indoor_temp_setpoint.shape[-1] - 1,
        )
        next_setpoint = self.indoor_temp_setpoint[:, next_timestep_global]
        return current_indoor_temp, next_setpoint

    def _parse_common_data(self, env_config: dict[str, Any]) -> None:
        """Parse and load common pricing/weather data and observation flags."""
        scenarios = env_config["scenarios"]
        if unknown := set(scenarios) - SCENARIO_KEYS:
            raise ValueError(f"Unknown scenario keys: {sorted(unknown)}")
        self.start_date = pd.to_datetime(scenarios["start_date"], utc=True)
        self.end_date = pd.to_datetime(scenarios["end_date"], utc=True)
        self.episode_len = int(scenarios["episode_len"])
        self.episode_starts = scenarios.get("episode_starts")
        self.episode_sampling = scenarios.get("episode_sampling", "random")
        self.episode_sampling_seed = scenarios.get("episode_sampling_seed")
        self.ev_initial_soc_range = scenarios.get(
            "ev_initial_soc_range", DEFAULT_EV_INITIAL_SOC_RANGE
        )

        self.cost_param = CostParameters.from_config(env_config.get("cost", {}))
        # TODO improve safety when parsin dt
        self.dt = env_config.get("dt", {"control_interval_seconds": 3600.0})

        self.action_space_schema = env_config["action_space_schema"]
        self.measurements_schema = env_config["measurements_schema"]

        self.root_dir = Path(env_config["root_dir"])
        if not self.root_dir.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            self.root_dir = project_root / self.root_dir

        self.observer_spec = ObserverSpec.from_env_config(env_config)
        self.observation_names = list(self.observer_spec.names)

        pricing_path = self.root_dir / env_config["data"]["pricing"]
        pricing = pd.read_csv(pricing_path)
        pricing["datetime"] = pd.to_datetime(pricing["datetime"])
        self.datetime = pricing["datetime"].values
        pricing.set_index("datetime", inplace=True)
        self._build_price_information(pricing, self.start_date, self.end_date)
        pricing = pricing.loc[self.start_date : self.end_date]
        self._time_index = pricing.index
        self._local_time_index = self._time_index.tz_convert(
            scenarios.get("local_tz", DEFAULT_LOCAL_TZ)
        )

        weather_path = self.root_dir / env_config["data"]["weather"]
        weather = pd.read_csv(weather_path)
        weather["datetime"] = pd.to_datetime(weather["datetime"])
        weather.set_index("datetime", inplace=True)
        weather = weather.loc[self.start_date : self.end_date]
        _require_columns(weather, WEATHER_COLUMNS, weather_path)
        self.external_temp = weather["external_temp"].to_numpy(np.float32)
        self.solar_ghi = weather["solar_ghi"].to_numpy(np.float32)

    def _parse_prosumer_data(self, env_config: dict[str, Any]) -> None:
        """Parse prosumer parameters and time-series arrays from configuration."""
        prosumers_config = env_config.get("prosumers", {})

        self.prosumer_ids = tuple(prosumers_config)
        self.num_prosumers = len(self.prosumer_ids)
        self.param = HouseParams.from_config(prosumers_config)

        data_file_arrays = {}
        setpoint_vectors = []
        for prosumer_id in self.prosumer_ids:
            data_file_path = prosumers_config[prosumer_id]["data_file"]
            data_file_path = self.root_dir / data_file_path
            df = pd.read_csv(data_file_path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            _require_columns(df, PROSUMER_COLUMNS, data_file_path)
            # Schedules are vectorized before slicing, so leads past end_date are real.
            window = df.index.slice_indexer(self.start_date, self.end_date)
            setpoint = df["indoor_temp_setpoint"].to_numpy()
            setpoint_vectors.append(build_schedule_vectors(setpoint)[window])
            df = df.loc[self.start_date : self.end_date]
            for column_name in PROSUMER_COLUMNS:
                data_file_arrays.setdefault(column_name, []).append(
                    df[column_name].values
                )

        columns = {}
        for column_name, prosumer_data_list in data_file_arrays.items():
            try:
                columns[column_name] = np.array(prosumer_data_list, dtype=np.float32)
            except ValueError as e:
                raise ValueError(
                    f"Error converting data for column '{column_name}' to numpy array. "
                    f"Prosumer data list shapes: {[data.shape for data in prosumer_data_list]}. "
                    "Check that all prosumer data files have the same number of timesteps and that the column exists in all files."
                ) from e

        self.domestic_load = columns["domestic_load"]
        self.pv_generation = columns["pv_generation"]
        self.indoor_temp_setpoint = columns["indoor_temp_setpoint"]
        self.ev_availability = columns["ev_availability"]
        self.ev_trip_demand = columns["ev_trip_demand"]
        self.ev_time_to_departure = columns["ev_time_to_departure"]
        self.ev_required_soc_departure = columns["ev_required_soc_departure"]
        self.forecasts["indoor_temp_setpoint"] = np.stack(setpoint_vectors)

    def _build_price_information(
        self, pricing: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> None:
        """Build the price vectors from the tariff data."""
        _require_columns(pricing, (*PRICE_STREAMS, "price_horizon"), "Pricing data")
        window = pricing.index.slice_indexer(start_date, end_date)
        known = pricing["price_horizon"].to_numpy()
        self.price_horizon = known[window].astype(np.float32)
        self.import_cost = pricing["import_cost"].to_numpy(np.float32)[window]
        self.export_price = pricing["export_price"].to_numpy(np.float32)[window]
        self.forecasts = {
            stream: build_price_forecast(
                pricing[stream].to_numpy(), known, FORECAST_HORIZON
            )[window]
            for stream in PRICE_STREAMS
        }

    def _load_forecasting_data(self, forecasting_data: str | Path | None) -> None:
        """Load the cross-fitted forecast vectors of external temperature, PV and load.

        Parameters
        ----------
        forecasting_data : str | Path | None
            File written by ``course_package.environment.forecasting`` containing forecasts of
            weather variables and domestic load.
        """
        vectors, self.forecast_metadata = read_forecast_file(
            resolve_forecast_path(self.root_dir, forecasting_data),
            self._time_index,
            self.prosumer_ids,
            FORECAST_HORIZON,
        )
        self.forecasts.update(vectors)

    def forecast_vector(self, variable: str, agent_idx: int | None = None) -> NDArray:
        """Return the information vector of a variable at the current step.

        Parameters
        ----------
        variable : str
            ``import_cost`` (import price plus carbon cost), ``export_price``,
            ``indoor_temp_setpoint``, ``external_temp``, ``solar_ghi``,
            ``pv_generation`` or ``domestic_load``.
        agent_idx : int | None, default=None
            Prosumer index; required for the per-prosumer streams.
        """
        idx = min(self.current_step_global, self.history_len - 1)
        vectors = self.forecasts[variable]
        return vectors[idx] if vectors.ndim == 2 else vectors[agent_idx, idx]

    @property
    def forecast_horizon(self) -> int:
        """Return the number of leads of every information vector."""
        return FORECAST_HORIZON

    def known_price_hours(self) -> int:
        """Return how many hourly prices are published at the current step."""
        idx = min(self.current_step_global, self.history_len - 1)
        return int(self.price_horizon[idx])

    def _setup_episode_boundaries(self) -> None:
        """Pre-compute non-overlapping fixed-length episode start indices."""
        self.history_len = len(self.import_cost)

        assert self.history_len >= self.episode_len, (
            f"Data length ({self.history_len} timesteps) must be >= "
            f"episode_len ({self.episode_len} timesteps). "
            f"Please expand date range in env_config or reduce episode_len."
        )
        valid_episode_starts = np.arange(
            0,
            self.history_len - self.episode_len + 1,
            self.episode_len,
            dtype=np.int64,
        )
        if self.episode_starts is not None:
            requested_starts = np.array(self.episode_starts, dtype=np.int64)
            if requested_starts.ndim != 1 or requested_starts.size == 0:
                raise ValueError("episode_starts must be a non-empty 1D sequence.")
            if len(set(requested_starts.tolist())) != requested_starts.size:
                raise ValueError("episode_starts must not contain duplicates.")

            invalid_starts = sorted(
                set(requested_starts.tolist()) - set(valid_episode_starts.tolist())
            )
            if invalid_starts:
                raise ValueError(
                    "Configured episode_starts must be valid non-overlapping "
                    f"episode boundaries. Invalid starts: {invalid_starts[:10]}"
                )
            valid_episode_starts = requested_starts

        self.valid_episode_starts = valid_episode_starts
        self.num_possible_episodes = len(self.valid_episode_starts)

    def _translate_action(self, action: NDArray) -> None:
        """Translate normalized actions into thermal and battery control variables."""
        self.actions_norm[:, self.current_step] = action
        thermal_action = action[:, 0]
        battery_action = action[:, 1]

        heating_load, cooling_load = self._translate_action_thermal(thermal_action)
        b_in, b_out = self._translate_action_battery(battery_action)

        self.heating_load[:, self.current_step] = heating_load
        self.cooling_load[:, self.current_step] = cooling_load
        self.thermal_load[:, self.current_step] = heating_load - cooling_load
        self.b_in[:, self.current_step] = b_in
        self.b_out[:, self.current_step] = b_out

    def _decode_action(self, action: MultiAgentData | NDArray) -> NDArray:
        """Convert nested actions to the internal thermal/battery action matrix."""
        if isinstance(action, np.ndarray):
            return action.astype(np.float32, copy=False)

        thermal_device = self.action_devices["action_thermal"]
        battery_device = self.action_devices["action_ev"]
        action_array = np.zeros(
            (self.num_prosumers, self.num_actions), dtype=np.float32
        )
        for agent_idx, agent_id in enumerate(self.prosumer_ids):
            agent_action = action[agent_id]
            action_array[agent_idx, 0] = agent_action[thermal_device]["action_thermal"][
                0
            ]
            action_array[agent_idx, 1] = agent_action[battery_device]["action_ev"][0]
        return action_array

    def encode_action(self, action: NDArray) -> MultiAgentData:
        """Convert an internal thermal/battery action matrix to nested actions.

        Inverse of ``_decode_action``. Controllers such as the expert emit actions
        in the internal ``[thermal, battery]`` column order, which differs from the
        schema order used by flattened action spaces.
        """
        action = np.asarray(action, dtype=np.float32).reshape(
            self.num_prosumers, self.num_actions
        )
        thermal_device = self.action_devices["action_thermal"]
        battery_device = self.action_devices["action_ev"]
        nested_action = {}
        for agent_idx, agent_id in enumerate(self.prosumer_ids):
            agent_action = {}
            agent_action.setdefault(thermal_device, {})["action_thermal"] = action[
                agent_idx, 0:1
            ]
            agent_action.setdefault(battery_device, {})["action_ev"] = action[
                agent_idx, 1:2
            ]
            nested_action[agent_id] = agent_action
        return nested_action

    def _translate_action_battery(self, action: NDArray) -> tuple[NDArray, NDArray]:
        """Battery action × rated power, then the shield (feasibility and mobility)."""
        p = self.param.ev
        t, g = self.current_step, self.current_step_global
        soc, plugged = self.ev_soc[:, t], self.ev_availability[:, g]

        b_in_raw = np.maximum(action, 0.0) * p.ev_b_in_max
        b_out_raw = np.maximum(-action, 0.0) * p.ev_b_out_max

        b_in_lb = self._compute_ev_requirement()
        b_in_ub = np.maximum(
            np.minimum(p.ev_b_in_max * plugged, p.ev_battery_capacity - soc), b_in_lb
        )
        b_out_ub = np.where(
            b_in_lb > 0.0,
            0.0,
            np.minimum(p.ev_b_out_max * plugged, np.maximum(soc - p.ev_soc_min, 0.0)),
        )
        self.ev_charge_override[:, t] = b_in_lb > 0.0
        return np.clip(b_in_raw, b_in_lb, b_in_ub), np.clip(b_out_raw, 0.0, b_out_ub)

    def _translate_action_thermal(self, action: NDArray) -> tuple[NDArray, NDArray]:
        """Map normalized thermal actions using HVAC mode gating constraints."""
        _, _, hub, cub = self._compute_thermal_load_bounds()

        u_heat = np.maximum(action, 0.0)
        u_cool = np.maximum(-1 * action, 0.0)
        u_heat = np.where(self.hvac_mode[:, self.current_step] == -1, 0.0, u_heat)
        u_cool = np.where(self.hvac_mode[:, self.current_step] == +1, 0.0, u_cool)

        hp_rated_power = self.param.thermal.hp_rated_power
        heating_load = u_heat * hp_rated_power
        cooling_load = u_cool * hp_rated_power

        heating_load = np.minimum(heating_load, hub)
        cooling_load = np.minimum(cooling_load, -1 * cub)

        return heating_load, cooling_load

    def _update_battery_soc(self) -> None:
        """Update EV state of charge and record trip energy below ``soc_min``.

        The SoC is clipped at ``soc_min``; ``soc_violation`` keeps the energy the
        clip supplied (a trip the battery could not cover).
        """
        p = self.param.ev
        next_soc = (
            self.ev_soc[:, self.current_step]
            + self.b_in[:, self.current_step]
            - self.b_out[:, self.current_step]
            - self.ev_trip_demand[:, self.current_step_global]
        )
        self.ev_soc[:, self.current_step + 1] = np.clip(
            next_soc, p.ev_soc_min, p.ev_battery_capacity
        )
        self.soc_violation[:, self.current_step] = np.maximum(
            0.0, p.ev_soc_min - next_soc
        )

    def _update_temperature(self) -> None:
        """Update indoor temperatures using thermal dynamics."""
        current_indoor_temp = self.indoor_temp[:, self.current_step]
        external_temp = self.external_temp[self.current_step_global]
        solar_ghi = self.solar_ghi[self.current_step_global]

        p = self.param.thermal
        temp_change = (
            self.heating_load[:, self.current_step]
            - self.cooling_load[:, self.current_step]
            + (external_temp - current_indoor_temp) / p.building_thermal_resistance
            + solar_ghi * p.building_solar_gain
        ) / p.building_thermal_capacitance

        self.indoor_temp[:, self.current_step + 1] = current_indoor_temp + temp_change

    def _update_hvac_mode(self) -> None:
        """Allow only heating below the comfort band and only cooling above it."""
        current_indoor_temp, next_setpoint = self._get_temp_context(self.current_step)

        hvac_mode = np.zeros(self.num_prosumers, dtype=np.int8)
        hvac_mode = np.where(
            current_indoor_temp < next_setpoint - self.cost_param.delta_temp,
            +1,
            hvac_mode,
        )
        hvac_mode = np.where(
            current_indoor_temp > next_setpoint + self.cost_param.delta_temp,
            -1,
            hvac_mode,
        )
        self.hvac_mode[:, self.current_step] = hvac_mode

    def _validate_action(self, action: MultiAgentData | NDArray) -> None:
        """Validate nested actions or the shape of legacy action matrices."""
        if not isinstance(action, np.ndarray):
            if not self.action_space.contains(action):
                raise ValueError("Action does not match the configured action space.")
            return

        expected_shape = (self.num_prosumers, self.num_actions)
        if action.shape != expected_shape:
            raise ValueError(
                f"Action shape {action.shape} does not match expected shape "
                f"{expected_shape}."
            )
