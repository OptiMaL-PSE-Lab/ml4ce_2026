"""Construction of per-prosumer expert controllers (thermal PID + battery MPC)."""

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from course_package.environment import DERMultiAgentEnv
from course_package.environment.forecasting import FORECAST_HORIZON
from course_package.expert.mpc import (
    BatteryCommand,
    BatteryMPC,
    MPCParameters,
    initialize_mpc_param,
)
from course_package.expert.pid import PIDParams, TemperaturePID, ThermalCommand


@dataclass(frozen=True)
class ExpertConfig:
    """Configuration for the per-prosumer expert controllers.

    Attributes
    ----------
    pid_gain_path : str | Path
        PID gains file, or directory containing one ``<prosumer_id>.npz`` per agent.
    horizon : int, default=FORECAST_HORIZON
        MPC prediction horizon in hours.
    information : Literal["perfect", "forecast"], default="forecast"
        Information mode of the battery MPC.
    """

    pid_gain_path: str | Path
    horizon: int = FORECAST_HORIZON
    information: Literal["perfect", "forecast"] = "forecast"


class Expert:
    """Expert controller of one prosumer: thermal PID and battery MPC.

    Parameters
    ----------
    thermal : TemperaturePID
        Thermal PID controller.
    battery : BatteryMPC
        EV battery MPC controller.
    """

    def __init__(self, thermal: TemperaturePID, battery: BatteryMPC) -> None:
        self.thermal = thermal
        self.battery = battery

    @classmethod
    def from_params(
        cls,
        mpc_param: MPCParameters,
        pid_param: PIDParams,
        gain_path: str | Path,
        information: Literal["perfect", "forecast"] = "forecast",
    ) -> "Expert":
        """Build the PID and MPC from their parameters and a PID gains file.

        Parameters
        ----------
        mpc_param : MPCParameters
            Battery MPC parameters.
        pid_param : PIDParams
            PID configuration values.
        gain_path : str | Path
            Path to the PID gains ``.npz`` file.
        information : Literal["perfect", "forecast"], default="forecast"
            Information mode of both controllers.

        Returns
        -------
        Expert
            Expert with a freshly built PID and MPC.
        """
        return cls(
            thermal=TemperaturePID(
                pid_param, gain_dir=str(gain_path), information=information
            ),
            battery=BatteryMPC(mpc_param, information=information),
        )

    def __call__(
        self,
        env: DERMultiAgentEnv,
        agent_idx: int = 0,
        *,
        thermal: ThermalCommand | None = None,
        battery: BatteryCommand | None = None,
    ) -> NDArray:
        """Return the clipped internal ``[thermal, battery]`` action."""
        return self.act(env, agent_idx=agent_idx, thermal=thermal, battery=battery)

    def act(
        self,
        env: DERMultiAgentEnv,
        agent_idx: int = 0,
        *,
        thermal: ThermalCommand | None = None,
        battery: BatteryCommand | None = None,
    ) -> NDArray:
        """Return the clipped internal ``[thermal, battery]`` action.

        Parameters
        ----------
        env : DERMultiAgentEnv
            Environment at the decision step.
        agent_idx : int, default=0
            Prosumer index.
        thermal, battery : ThermalCommand | BatteryCommand | None, default=None
            Modifications of the PID and the MPC for this step, e.g. set by a
            supervisory policy; ``None`` applies the expert unchanged.
        """
        battery_action = self.battery_action(env, agent_idx, battery)
        thermal_action = self.thermal_action(env, agent_idx, thermal)
        return np.array([thermal_action, battery_action], dtype=np.float32)

    def thermal_action(
        self,
        env: DERMultiAgentEnv,
        agent_idx: int = 0,
        command: ThermalCommand | None = None,
    ) -> float:
        """Return the clipped normalized thermal action."""
        return float(np.clip(self.thermal(env, agent_idx, command), -1.0, 1.0))

    def battery_action(
        self,
        env: DERMultiAgentEnv,
        agent_idx: int = 0,
        command: BatteryCommand | None = None,
    ) -> float:
        """Return the clipped normalized battery action."""
        return float(np.clip(self.battery(env, agent_idx, command), -1.0, 1.0))

    def reset(self, seed: int | None = None) -> None:
        """Start a new episode: forget the PID's tracking-error memory.

        The PID also restarts it by itself at step 0; the MPC keeps no state.
        """
        self.thermal.reset()


def build_experts(env: DERMultiAgentEnv, config: ExpertConfig) -> list[Expert]:
    """Build one independent PID + MPC expert per prosumer of a raw environment.

    Controller parameters are indexed by the prosumer's position in ``env``, so
    configs restricted with ``select_prosumers`` are supported. Warns if a PID
    integrates the error over another window than the policy observations, since
    a policy cloned from it then cannot reproduce its integral action exactly.
    """
    experts = []
    for agent_idx, agent_id in enumerate(env.prosumer_ids):
        gain_path = Path(config.pid_gain_path)
        if gain_path.is_dir():
            gain_path = gain_path / f"{agent_id}.npz"
        experts.append(
            Expert.from_params(
                initialize_mpc_param(env, agent_idx=agent_idx, N=config.horizon),
                PIDParams(),
                gain_path,
                information=config.information,
            )
        )
        if experts[-1].thermal.window != env.observer_spec.window:
            warnings.warn(
                f"PID of {agent_id} integrates over {experts[-1].thermal.window} "
                f"steps, the observations over {env.observer_spec.window}.",
                stacklevel=2,
            )
    return experts


def extend_expert_env_config(
    split_config: dict, env_config: dict, horizon: int
) -> dict:
    """Extend data access beyond the final episode so MPC horizons stay in range."""
    if horizon <= 0:
        raise ValueError("Expert horizon must be positive.")

    split_end = datetime.fromisoformat(split_config["scenarios"]["end_date"])
    data_end = datetime.fromisoformat(env_config["scenarios"]["end_date"])
    expert_end = split_end + timedelta(hours=horizon)
    if expert_end > data_end:
        raise ValueError("Environment data does not cover the expert MPC horizon.")

    expert_env_config = dict(split_config)
    expert_env_config["scenarios"]["end_date"] = expert_end.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return expert_env_config
