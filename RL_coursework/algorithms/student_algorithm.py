"""Group XX: <member names>. Implementation of my own RL algorithm."""

from dataclasses import dataclass

import numpy as np

from course_package import MARLEnv
from course_package.utils.helpers import (
    BaseAlgorithmConfig,
    set_seed,
)


class SharedPolicy:
    """
    The single policy shared by all agents.

    It maps ONE agent's local observation to that agent's action. The same
    parameters are used for every agent: ``act`` is called once per agent, so
    agents behave differently only because their observations differ.

    The policy object is the ONLY thing carried from one stage to the next: store in
    it anything your algorithm needs later (e.g. a copy of earlier parameters). It
    must be picklable and must not keep a reference to the environment.
    """

    def __init__(self, obs_dim: int, action_dim: int):
        # YOUR POLICY PARAMETERS GO HERE (e.g. numpy arrays or a torch.nn.Module)
        self.weights = np.zeros((action_dim, obs_dim), dtype=np.float32)

    def act(self, local_obs: np.ndarray) -> np.ndarray:
        """
        Deterministic action used for evaluation and grading.

        ``local_obs`` has shape (obs_dim,). Return an array of shape
        (action_dim,) with values in [-1, 1].
        """
        return np.tanh(self.weights @ local_obs)


@dataclass
class AlgorithmConfig(BaseAlgorithmConfig):
    """Hyperparameter specification for Custom Algorithm."""

    hyperparameter_1: float = 0.5
    hyperparameter_2: int = 10
    group_identifier: str = "GROUP_NAME"


def train(
    env: MARLEnv,
    config: AlgorithmConfig,
    seed: int = 0,
    policy: SharedPolicy | None = None,
) -> tuple[SharedPolicy, dict[str, str]]:
    """
    Train the single policy shared by all agents on ONE stage of the curriculum.

    `env` contains only the household of this stage. It returns observations of
    shape (num_agents, num_features): row i is the local observation of agent i.
    Apply the same policy to every row.

    `policy` is the policy returned by the previous stage, or None at the first
    stage (`config.stage == 0`), where you create it. Adapt it to the new household
    without forgetting the previous ones, within `config.max_steps` environment
    steps and `config.max_time` seconds.
    """
    set_seed(seed)
    if policy is None:
        policy = SharedPolicy(env.observation_space.shape[1], env.action_space.shape[1])

    # YOUR CODE GOES HERE

    return policy, {}
