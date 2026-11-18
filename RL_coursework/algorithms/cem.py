"""Noisy Cross-Entropy Method (CEM) for a single shared policy.

The elite variance is inflated by a decaying noise term so the search does not
collapse when only a few elites are available.
"""

from dataclasses import dataclass
from time import time

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from course_package import MARLEnv
from course_package.utils.helpers import (
    BaseAlgorithmConfig,
    CSVLogger,
    evaluate_marl_policy,
    save_training_plots,
    set_seed,
    setup_model_saving,
)


@dataclass
class AlgorithmConfig(BaseAlgorithmConfig):
    """Hyperparameters for the Cross-Entropy Method."""

    pop_size: int = 16
    elite_frac: float = 0.25
    init_std: float = 0.1  # search spread when the policy is created (first stage)
    extra_noise: float = 1.0
    num_eval_episodes: int = 2
    group_identifier: str = "CEM"


class PolicyNetwork(torch.nn.Module):
    """Single policy shared by all agents, with Gaussian outputs."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.output = torch.nn.Linear(64, action_dim * 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.output(x)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        return torch.tanh(mean), log_std

    @torch.no_grad()
    def act(self, local_obs: NDArray) -> NDArray:
        """Deterministic (mean) action ``(action_dim,)`` for one agent's observation ``(obs_dim,)``."""
        mean, _ = self(torch.as_tensor(local_obs, dtype=torch.float32))
        return mean.numpy()


def _policy_vector(policy: PolicyNetwork) -> torch.Tensor:
    """Flatten a policy network into a single parameter vector."""
    with torch.no_grad():
        return torch.nn.utils.parameters_to_vector(policy.parameters()).detach().clone()


def extra_noise_std(generation: int, num_generations: int, initial_std: float) -> float:
    """Noise std added after ``generation`` (noisy CEM, Szita & Lőrincz 2006); linear decay to 0."""
    return initial_std * max(0.0, 1.0 - (generation + 1) / num_generations)


def train(
    env: MARLEnv,
    config: AlgorithmConfig,
    seed: int = 0,
    policy: PolicyNetwork | None = None,
) -> tuple[PolicyNetwork, dict[str, str]]:
    """
    Train ONE shared policy (used by every agent) with the Cross-Entropy Method.

    ``policy`` (from a previous stage) is the centre of the initial search
    distribution; when ``None`` a new policy is created.
    """
    set_seed(seed)
    if policy is None:
        agent_policy = PolicyNetwork(
            env.observation_space.shape[1], env.action_space.shape[1]
        )
        initial_std = config.init_std
    else:
        agent_policy = policy
        initial_std = config.init_std / 2
    pbar = tqdm(
        desc=f"Running {config.group_identifier} RL training loop",
        total=config.max_steps,
    )
    save_folder = setup_model_saving(f"{config.group_identifier}_stage{config.stage}")

    pop_size = config.pop_size
    elite_size = max(1, int(pop_size * config.elite_frac))

    mean = _policy_vector(agent_policy)
    std = torch.full_like(mean, initial_std)

    save_paths = {"shared_policy": save_folder / "shared_policy.pt"}
    logger = CSVLogger(save_folder / "shared_policy.csv")

    start_time = time()
    step_count = 0
    generation = 0
    candidate_cost = env.episode_len * config.num_eval_episodes
    num_generations = max(1, config.max_steps // (pop_size * candidate_cost))

    def budget_remaining() -> bool:
        pbar.update(step_count - pbar.n)
        return (
            step_count + candidate_cost <= config.max_steps
            and time() - start_time < config.max_time
        )

    while budget_remaining():
        samples = []
        fitness = []

        for _ in range(pop_size):
            if not budget_remaining():
                break

            sample = mean + std * torch.randn_like(mean)
            torch.nn.utils.vector_to_parameters(sample, agent_policy.parameters())

            returns = evaluate_marl_policy(
                env,
                agent_policy,
                num_episodes=config.num_eval_episodes,
                seed=seed
                + generation * config.num_eval_episodes,  # same episodes per generation
            )
            samples.append(sample)
            fitness.append(float(returns.mean(axis=0).sum()))
            step_count += env.episode_len * config.num_eval_episodes

        if not samples:
            break

        samples = torch.stack(samples)
        fitness = np.asarray(fitness, dtype=np.float32)

        elite_indices = fitness.argsort()[::-1][:elite_size].copy()
        elite_params = samples[elite_indices]
        mean = elite_params.mean(dim=0)
        noise = extra_noise_std(
            generation, num_generations, config.extra_noise * initial_std
        )
        elite_var = elite_params.var(dim=0, unbiased=False)
        std = torch.sqrt(elite_var + noise**2) + 1e-8

        torch.nn.utils.vector_to_parameters(mean, agent_policy.parameters())
        torch.save(agent_policy.state_dict(), save_paths["shared_policy"])

        logger.log(
            step=step_count,
            episode=generation,
            agent_id="shared_policy",
            train_return=float(fitness.mean()),
            eval_return=float(fitness[elite_indices].mean()),
            policy_loss=np.nan,
            value_loss=np.nan,
            entropy=np.nan,
        )
        generation += 1

    logger.close()
    torch.nn.utils.vector_to_parameters(mean, agent_policy.parameters())
    pbar.close()

    save_training_plots(save_folder, title=config.group_identifier)
    return agent_policy, save_paths
