"""Array interface of the DER environment used throughout the coursework."""

from typing import Any

import numpy as np
from gymnasium import Wrapper
from gymnasium.spaces import Box
from numpy.typing import NDArray

from course_package.environment.environment import DERMultiAgentEnv

_NO_COPY_MESSAGE = (
    "Environments cannot be pickled or copied: a policy must not keep a reference "
    "to its environment, and every simulated step must go through env.step() so it "
    "is counted in the budget."
)


class _NonCopyableDERMultiAgentEnv(DERMultiAgentEnv):
    """``DERMultiAgentEnv`` that refuses to be pickled or (deep)copied."""

    def __getstate__(self) -> dict:
        raise TypeError(_NO_COPY_MESSAGE)


class MARLEnv(Wrapper):
    """``DERMultiAgentEnv`` with array observations, actions and rewards.

    Row ``i`` of every array belongs to household ``agent_ids[i]``.

    - Observations: ``(num_agents, num_features)``, scaled features clipped to
      ``±observer_spec.scales.clip``.
    - Actions: ``(num_agents, 2)`` in ``[-1, 1]``, columns ``[thermal, battery]``.
    - Rewards: ``(num_agents,)``.
    - ``step`` returns ``done`` in the ``terminated`` slot (the inner environment
      ends episodes by truncation) and ``False`` as ``truncated``.

    Attributes of the inner environment (``episode_len``, ``valid_episode_starts``,
    ``current_step``, ...) are available on the wrapper. The environment cannot be
    pickled or copied.

    Parameters
    ----------
    env_config : dict[str, Any]
        Environment configuration (see ``data/datasets/training/env_config.json``).
    """

    def __init__(self, env_config: dict[str, Any]) -> None:
        super().__init__(_NonCopyableDERMultiAgentEnv(env_config))
        raw = self.env
        self.num_agents = raw.num_prosumers
        self.agent_ids = raw.prosumer_ids
        self.active_observations = list(raw.observer_spec.names)
        clip = raw.observer_spec.scales.clip
        self.observation_space = Box(
            -clip, clip, (self.num_agents, raw.observer_spec.size), np.float32
        )
        self.action_space = Box(
            -1.0, 1.0, (self.num_agents, raw.num_actions), np.float32
        )
        # Number of env.step() calls made on this env, i.e. the steps billed to a
        # training budget.
        self.step_count = 0
        self._done = True

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray, dict]:
        """Start an episode and return the ``(num_agents, num_features)`` observations."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._done = False
        return self._stack(obs), info

    def step(self, actions: NDArray) -> tuple[NDArray, NDArray, bool, bool, dict]:
        """Apply the joint ``(num_agents, 2)`` action and advance one hour."""
        if self._done:
            raise RuntimeError("Episode has finished. Call reset() before step().")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != self.action_space.shape:
            raise ValueError(
                f"actions must have shape {self.action_space.shape}, got {actions.shape}"
            )
        actions = np.clip(np.nan_to_num(actions, nan=0.0), -1.0, 1.0)
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        self.step_count += 1
        self._done = bool(terminated or truncated)
        return self._stack(obs), rewards, self._done, False, info

    @staticmethod
    def _stack(obs: dict[str, NDArray]) -> NDArray:
        """Stack per-household observations in household order."""
        return np.stack(list(obs.values())).astype(np.float32, copy=False)

    def __getstate__(self) -> dict:
        raise TypeError(_NO_COPY_MESSAGE)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "env"), name)
