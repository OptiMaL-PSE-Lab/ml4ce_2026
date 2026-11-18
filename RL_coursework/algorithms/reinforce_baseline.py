"""REINFORCE with a value-function baseline for a single shared policy."""

from copy import deepcopy
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
    compute_returns,
    evaluate_marl_policy,
    save_training_plots,
    set_seed,
    setup_model_saving,
)


@dataclass
class AlgorithmConfig(BaseAlgorithmConfig):
    """Hyperparameters for REINFORCE with a learned state-value baseline."""

    init_std: float = 0.1
    entropy_weight: float = 0.01
    gradient_clip: float = 1.0
    lr_policy: float = 1e-3
    lr_value: float = 1e-3
    train_batch_size: int = 504
    num_eval_episodes: int = 1
    eval_seed_offset: int = 10_000  # fixed evaluation episodes: seed + offset (+ i)
    group_identifier: str = "REINFORCE"


class PolicyNetwork(torch.nn.Module):
    """
    Single policy shared by all agents, with Gaussian outputs.
    """

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.output = torch.nn.Linear(256, action_dim * 2)

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


class ValueNetwork(torch.nn.Module):
    """State-value baseline evaluated on one agent's local observation."""

    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReinforceAgent:
    """Shared policy plus its value baseline, carried together between stages.

    Only ``act`` (the policy's deterministic action) is used for evaluation; the
    value network is kept so that the next stage can continue training it.
    """

    def __init__(self, obs_dim: int, action_dim: int):
        self.policy = PolicyNetwork(obs_dim, action_dim)
        self.value = ValueNetwork(obs_dim)

    def act(self, local_obs: NDArray) -> NDArray:
        """Deterministic (mean) action of the policy for one agent's observation."""
        return self.policy.act(local_obs)


def action_distribution(
    policy_net: PolicyNetwork, obs: torch.Tensor
) -> torch.distributions.Normal:
    """Diagonal Gaussian of the shared policy for observations ``(..., obs_dim)``."""
    mean, log_std = policy_net(obs)
    log_std = -3.0 + 3.0 * torch.sigmoid(log_std)
    std = torch.exp(log_std)
    return torch.distributions.Normal(mean, std)


@torch.no_grad()
def choose_action(state: NDArray, policy_net: PolicyNetwork) -> NDArray:
    """Sample actions ``(..., action_dim)`` from the shared policy (no graph is kept)."""
    dist = action_distribution(policy_net, torch.from_numpy(state).float())
    return dist.sample().numpy()


def train(
    env: MARLEnv,
    config: AlgorithmConfig,
    seed: int = 0,
    policy: ReinforceAgent | None = None,
) -> tuple[ReinforceAgent, dict[str, str]]:
    """
    Train ONE shared policy (used by every agent) with REINFORCE + baseline.
    """
    set_seed(seed)
    if policy is None:
        agent = ReinforceAgent(
            env.observation_space.shape[1], env.action_space.shape[1]
        )
    else:
        agent = policy
    agent_policy, value_net = agent.policy, agent.value
    pbar = tqdm(
        desc=f"Running {config.group_identifier} RL training loop",
        total=config.max_steps,
    )
    save_folder = setup_model_saving(f"{config.group_identifier}_stage{config.stage}")

    policy_opt = torch.optim.Adam(agent_policy.parameters(), lr=config.lr_policy)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=config.lr_value)

    save_paths = {"shared_policy": save_folder / "shared_policy.pt"}
    logger = CSVLogger(save_folder / "shared_policy.csv")

    best_eval_return = -np.inf
    best_state_dict = deepcopy(agent_policy.state_dict())

    start_time = time()
    step_count = 0
    batch_idx = 0
    episode_idx = 0
    eval_cost = env.episode_len * config.num_eval_episodes

    def budget_remaining() -> bool:
        pbar.update(step_count - pbar.n)
        return (
            step_count + eval_cost < config.max_steps
            and time() - start_time < config.max_time
        )

    while budget_remaining():
        logprobs, entropies, values, returns = [], [], [], []
        episode_returns = []

        batch_size = 0
        while batch_size < config.train_batch_size and budget_remaining():
            obs, _ = env.reset(seed=seed + episode_idx)
            done = False
            episode_rewards = []

            while not done and budget_remaining():
                dist = action_distribution(agent_policy, torch.from_numpy(obs).float())
                actions = dist.sample().numpy()
                logprob = dist.log_prob(torch.from_numpy(actions).float()).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                value = value_net(torch.from_numpy(obs).float()).squeeze(-1)

                obs, rewards, done, _, _ = env.step(actions)
                episode_rewards.append(np.asarray(rewards, dtype=np.float32))

                batch_size += 1
                step_count += 1
                logprobs.append(logprob)
                entropies.append(entropy)
                values.append(value)

            rewards = np.stack(episode_rewards)  # (steps, num_agents)
            returns.append(
                np.stack(
                    [
                        compute_returns(rewards[:, a], gamma=config.gamma)
                        for a in range(rewards.shape[1])
                    ],
                    axis=1,
                )
            )
            episode_returns.append(float(rewards.sum()))
            episode_idx += 1

        returns = torch.as_tensor(np.concatenate(returns), dtype=torch.float32)
        returns = returns.flatten()
        values = torch.stack(values).flatten()
        logprobs = torch.stack(logprobs).flatten()
        entropies = torch.stack(entropies).flatten()

        advantages = returns - values.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = (
            -(advantages * logprobs).mean() - config.entropy_weight * entropies.mean()
        )
        value_loss = torch.mean((values - returns) ** 2)

        policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_policy.parameters(), config.gradient_clip)
        policy_opt.step()

        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        eval_returns = evaluate_marl_policy(
            env,
            agent_policy,
            num_episodes=config.num_eval_episodes,
            seed=seed + config.eval_seed_offset,
        ).mean(axis=0)
        eval_return = float(eval_returns.sum())
        train_return = float(np.mean(episode_returns))
        step_count += env.episode_len * config.num_eval_episodes

        if eval_return > best_eval_return:
            best_eval_return = eval_return
            best_state_dict = deepcopy(agent_policy.state_dict())
            torch.save(best_state_dict, save_paths["shared_policy"])

        logger.log(
            step=step_count,
            episode=batch_idx,
            agent_id="shared_policy",
            train_return=train_return,
            eval_return=eval_return,
            policy_loss=float(policy_loss.detach()),
            value_loss=float(value_loss.detach()),
            entropy=float(entropies.mean().detach()),
        )
        batch_idx += 1

    logger.close()
    agent_policy.load_state_dict(best_state_dict)
    pbar.close()
    save_training_plots(save_folder, title=config.group_identifier)
    return agent, save_paths
