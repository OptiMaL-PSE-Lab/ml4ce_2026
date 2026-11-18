"""Implementation of a PID controller for temperature regulation in a DER environment."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from course_package.core import DEFAULT_ERROR_WINDOW, TrackingErrorFilter

if TYPE_CHECKING:
    from course_package.environment import DERMultiAgentEnv


@dataclass
class PIDParams:
    """PID configuration parameters.

    Attributes
    ----------
    n_episodes_avg : int
        Number of episodes used when tuning gains.
    w_magnitude : float
        Weight for action magnitude in the tuning objective.
    w_change : float
        Weight for action change in the tuning objective.
    """

    n_episodes_avg: int = 10
    w_magnitude: float = 0.01
    w_change: float = 0.01
    seed: int = 42
    random_search_iters: int = 500
    tuning_objective: Literal["environment_return", "tracking"] = "environment_return"


@dataclass(frozen=True)
class ThermalCommand:
    """Per-step modifications of the PID set by a supervisory policy.

    The defaults leave the PID unchanged.

    Attributes
    ----------
    setpoint_offset : float, default=0.0
        Offset (°C) added to the setpoint at this step. The PID remembers the
        offsets of the last ``window`` steps in its integral.
    gain_scale : tuple[float, float, float], default=(1.0, 1.0, 1.0)
        Multipliers of the gains ``(Kp, Ki, Kd)``.
    """

    setpoint_offset: float = 0.0
    gain_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


def random_search(
    objective_function: Callable,
    bounds: NDArray,
    mode: Literal["normal", "log"] = "normal",
    iter: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray, float]:
    """Run random search over bounded parameters.

    Parameters
    ----------
    objective_function : Callable
        Objective function to minimize.
    bounds : NDArray
        Parameter bounds shaped ``(n_params, 2)``.
    mode : Literal["normal", "log"], default="normal"
        Sampling distribution type.
    iter : int, default=100
        Number of random samples.

    Returns
    -------
    tuple[NDArray, float]
        Best parameter vector and objective value.
    """
    if iter <= 0:
        raise ValueError("iter must be positive.")
    rng = np.random.default_rng() if rng is None else rng

    if mode == "log":
        lb = np.where(bounds[:, 0] <= 0.0, 1e-8, bounds[:, 0])
        sample_param = 10 ** rng.uniform(
            np.log10(lb), np.log10(bounds[:, 1]), size=(iter, bounds.shape[0])
        )
    elif mode == "normal":
        sample_param = rng.uniform(
            bounds[:, 0], bounds[:, 1], size=(iter, bounds.shape[0])
        )

    obj_values = list(map(objective_function, sample_param))
    idx_best_obj = np.argmin(obj_values)
    return (
        sample_param[idx_best_obj, :],
        obj_values[idx_best_obj],
    )


class TemperaturePID:
    """PID controller for temperature regulation.

    Its only state is the tracking-error memory of each prosumer it controls (a
    ``TrackingErrorFilter`` over the integral window the gains were tuned with),
    fed from ``env.get_measurements()``. Call it once per step, in order; a new
    episode (step 0) restarts the memory.

    Parameters
    ----------
    param : PIDParams
        PID configuration values.
    log : bool, default=False
        Whether to write optimization logs.
    gain_dir : Optional[str], default=None
        Optional path to pre-tuned gains.
    information : Literal["perfect", "forecast"], default="perfect"
        Information mode used by the expert interface.

    Attributes
    ----------
    Ks : NDArray
        PID gains.
    gain_dir : Optional[str]
        Path the gains were loaded from, or ``None`` for default gains.
    memory : Optional[int]
        Integral window the loaded gains were tuned with, or ``None`` if unknown
        (then ``DEFAULT_ERROR_WINDOW`` is used).
    n_episodes_avg : int
        Number of episodes to average for objective evaluation.
    w_magnitude : float
        Weight for magnitude term in objective function.
    w_change : float
        Weight for change term in objective function.
    information : str
        Information mode used by the expert interface.
    _log : bool
        Whether to write optimization logs.
    """

    def __init__(
        self,
        param: PIDParams,
        log: bool = False,
        gain_dir: str | None = None,
        information: Literal["perfect", "forecast"] = "perfect",
    ) -> None:
        """Initialize PID controller state and optional gain loading.

        Parameters
        ----------
        param : PIDParams
            PID configuration values.
        log : bool, default=False
            Whether to write optimization logs.
        gain_dir : Optional[str], default=None
            Optional path to pre-tuned gains.
        information : Literal["perfect", "forecast"], default="perfect"
            Information mode used by the expert interface.
        """
        self.n_episodes_avg = param.n_episodes_avg
        self.w_magnitude = param.w_magnitude
        self.w_change = param.w_change
        self.tuning_objective = param.tuning_objective
        self.information = information
        self._log = log
        self.gain_dir = gain_dir
        self.memory: int | None = None
        if gain_dir is None:
            self.Ks = np.array([0.1, 0.01, 0.001])
        else:
            self.load_gains(gain_dir)
        self._tracking: dict[int, TrackingErrorFilter] = {}

    @property
    def window(self) -> int:
        """Return the integral window of the tracking-error memory (steps)."""
        return DEFAULT_ERROR_WINDOW if self.memory is None else int(self.memory)

    def __call__(
        self,
        env: "DERMultiAgentEnv",
        agent_idx: int = 0,
        command: ThermalCommand | None = None,
    ) -> float:
        """Return the computed thermal action (see ``compute_action``)."""
        return self.compute_action(env, agent_idx=agent_idx, command=command)

    def compute_action(
        self,
        env: "DERMultiAgentEnv",
        agent_idx: int = 0,
        command: ThermalCommand | None = None,
    ) -> float:
        """Compute normalized PID thermal action.

        Parameters
        ----------
        env : DERMultiAgentEnv
            Environment containing current thermal context.
        agent_idx : int, default=0
            Agent index.
        command : ThermalCommand | None, default=None
            Setpoint offsets and gain multipliers of this step; ``None`` applies
            the PID unchanged.

        Returns
        -------
        float
            PID action value.
        """
        state = self.error_state(env, agent_idx, command)
        return np.float32(self.gains(command) @ state)

    def error_state(
        self,
        env: "DERMultiAgentEnv",
        agent_idx: int = 0,
        command: ThermalCommand | None = None,
    ) -> NDArray:
        """Return ``[error, integral, derivative]`` after the command's offset.

        Errors are ``setpoint - temperature``. The offset shifts the error and the
        integral; the derivative ignores it, so a change of offset causes no
        derivative kick.
        """
        m = env.get_measurements()
        offset = 0.0 if command is None else command.setpoint_offset
        if agent_idx not in self._tracking:
            self._tracking[agent_idx] = TrackingErrorFilter(self.window)
        state = self._tracking[agent_idx].update(
            m.step, m.tracking_error[agent_idx : agent_idx + 1], offset
        )
        return np.array(
            [state.error[0], state.integral[0], state.derivative[0]], dtype=np.float64
        )

    def reset(self) -> None:
        """Forget the tracking-error memory (also done when a new episode starts)."""
        self._tracking.clear()

    def gains(self, command: ThermalCommand | None = None) -> NDArray:
        """Return the gains ``[Kp, Ki, Kd]`` after the command's multipliers."""
        if command is None:
            return np.asarray(self.Ks, dtype=np.float64)
        return np.asarray(self.Ks, dtype=np.float64) * np.asarray(command.gain_scale)

    def load_gains(self, path: str) -> None:
        """Load PID gains from a file.

        Parameters
        ----------
        path : str
            Path to ``.npz`` gains file.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"PID gains file not found: {path}")
        Ks = np.load(path)
        self.Ks = np.array([Ks["Kp"], Ks["Ki"], Ks["Kd"]])
        if "metadata_json" in Ks.files:
            self.memory = json.loads(str(Ks["metadata_json"])).get("memory")

    @property
    def log(self) -> bool:
        """Flag to write optimization logs.

        Returns
        -------
        bool
            Whether logging is enabled.
        """
        return self._log

    @log.setter
    def log(self, value: bool) -> None:
        """Enable or disable objective logging to file."""
        self._log = value
        if value:
            self.log_path = Path.cwd() / "PID_log.log"
            self.log_f = open(self.log_path, "w+", encoding="utf-8")
            self.log_f.write("error,umag,ucha,obj_fcn\n")

    def objective_function(
        self, Ks: NDArray, env: "DERMultiAgentEnv", episode_starts: list
    ) -> float:
        """Evaluate PID gains on sampled episodes.

        Parameters
        ----------
        Ks : NDArray
            Candidate gains ``[Kp, Ki, Kd]``.
        env : DERMultiAgentEnv
            Environment used for evaluation.
        episode_starts : list
            Episode start indices for rollouts.

        Returns
        -------
        float
            Scalar objective to minimize.
        """
        self.Ks = Ks
        error_list = []
        action_list = []
        action_change_list = []
        episode_returns = []
        for t_start in episode_starts:
            env.reset(seed=int(t_start), options={"episode_start": int(t_start)})
            done = False
            episode_actions = []
            episode_return = 0.0
            while not done:
                a = np.clip(
                    [self.compute_action(env, i) for i in range(env.num_prosumers)],
                    -1.0,
                    1.0,
                )
                action = np.zeros((env.num_prosumers, env.num_actions), np.float32)
                action[:, 0] = a
                _, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_return += float(np.sum(reward))

                action_list.extend(a)
                episode_actions.append(a)
                setpoint_idx = min(
                    env.current_step_global,
                    env.indoor_temp_setpoint.shape[-1] - 1,
                )
                error_list.append(
                    (
                        env.indoor_temp[:, env.current_step]
                        - env.indoor_temp_setpoint[:, setpoint_idx]
                    )
                    ** 2
                )
            action_change_list.extend(
                np.abs(np.diff(episode_actions, axis=0)).ravel().tolist()
            )
            episode_returns.append(episode_return)
        error = np.mean(error_list)
        u_mag = np.mean(np.abs(action_list))
        u_change = float(np.mean(action_change_list)) if action_change_list else 0.0
        if self.tuning_objective == "environment_return":
            obj = -float(np.mean(episode_returns))
        else:
            obj = error + self.w_magnitude * u_mag + self.w_change * u_change
        if self.log:
            self.log_f.write(f"{error},{u_mag},{u_change},{obj}\n")

        return obj

    def update_temperature(
        self,
        env: "DERMultiAgentEnv",
        heat_load: float,
        cool_load: float,
        agent_idx: int = 0,
    ) -> NDArray:
        """Return indoor temperature after applying heat and cool loads.

        TODO: Extract this and unify with the environment's temperature update logic.

        Parameters
        ----------
        env : DERMultiAgentEnv
            Environment containing current thermal context.
        heat_load : float
            Heating load.
        cool_load : float
            Cooling load.
        agent_idx : int, default=0
            Agent index.

        Returns
        -------
        NDArray
            Updated indoor temperature value.
        """
        T = env.indoor_temp[agent_idx, env.current_step]
        Text = env.external_temp[env.current_step_global]
        solar_ghi = env.solar_ghi[env.current_step_global]

        p = env.param.thermal
        temp_change = (
            heat_load
            - cool_load
            + (Text - T) / p.thermal_resistance
            + solar_ghi * p.solar_gain
        ) / p.thermal_capacitance

        return T + temp_change


class VelocityPID:
    """Velocity form of a ``TemperaturePID``, for gains that change every step.

    ``u[t] = u[t-1] + K[t] (x[t] - x[t-1])`` with ``x = [error, integral,
    derivative]``. A change of gains only scales the next increment, so the
    output does not jump (bumpless). With constant gains the increments telescope
    to the position form ``u[t] = K x[t]``, so it reproduces the PID.

    The controller keeps ``u[t-1]`` and ``x[t-1]``: call it exactly once per step.
    It restarts from the position form at the first step of every episode.

    Parameters
    ----------
    pid : TemperaturePID
        PID providing the base gains and the error state.
    """

    def __init__(self, pid: TemperaturePID) -> None:
        self.pid = pid
        self.reset()

    def __call__(
        self,
        env: "DERMultiAgentEnv",
        agent_idx: int = 0,
        command: ThermalCommand | None = None,
    ) -> float:
        """Return the normalized thermal action and advance the controller."""
        state = self.pid.error_state(env, agent_idx, command)
        gains = self.pid.gains(command)
        if self._output is None or env.current_step == 0:
            output = float(gains @ state)
        else:
            output = self._output + float(gains @ (state - self._state))
        self._output, self._state = output, state
        return np.float32(output)

    def reset(self) -> None:
        """Forget the previous output and error state."""
        self._output: float | None = None
        self._state: NDArray | None = None


def tune_gains(
    param: PIDParams,
    env: "DERMultiAgentEnv",
    bounds: NDArray,
) -> NDArray:
    """Tune PID gains using random search followed by local optimization.

    Parameters
    ----------
    param : PIDParams
        PID tuning parameters.
    env : DERMultiAgentEnv
        Environment used for evaluation rollouts.
    bounds : NDArray
        Gain bounds shaped ``(3, 2)``.

    Returns
    -------
    NDArray
        Optimized gains ``[Kp, Ki, Kd]``.

    Raises
    ------
    ValueError
        If local optimization fails.
    """

    if param.n_episodes_avg > len(env.valid_episode_starts):
        raise ValueError("n_episodes_avg exceeds the available episode count.")
    rng = np.random.default_rng(param.seed)
    pid = TemperaturePID(param)
    episode_starts = rng.choice(
        env.valid_episode_starts,
        size=param.n_episodes_avg,
        replace=False,
    ).astype(int)
    Ks_rs, _ = random_search(
        lambda p: pid.objective_function(p, env, episode_starts),
        bounds,
        mode="log",
        iter=param.random_search_iters,
        rng=rng,
    )
    print(f"Random search best gains: Kp={Ks_rs[0]}, Ki={Ks_rs[1]}, Kd={Ks_rs[2]}")

    pid.log = True
    result = minimize(
        pid.objective_function,
        args=(env, episode_starts),
        x0=Ks_rs,
        bounds=bounds,
        method="Powell",
    )
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed: " + result.message)
