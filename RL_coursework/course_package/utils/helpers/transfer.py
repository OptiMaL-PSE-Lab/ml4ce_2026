"""Training over a sequence of households (transfer / continual learning).

A curriculum trains ONE shared policy in stages. Stage ``k`` gives ``train`` an
environment with household ``k`` only, plus the policy returned by stage ``k - 1``
(``None`` at the first stage). Nothing but the policy object is carried between
stages: it is pickled and unpickled, as in grading, and must stay below
``MAX_POLICY_BYTES``. After every stage the policy is evaluated on every household
of a validation environment, which gives a matrix of mean episode returns.
"""

import pickle
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from course_package import Expert, MARLEnv, select_prosumers
from course_package.utils.helpers.helpers import (
    BaseAlgorithmConfig,
    _shared_policy_actions,
    expert_actions,
)

MAX_POLICY_BYTES = 50 * 1024**2
"""Largest pickled policy that may be carried from one stage to the next."""

STAGE_SEED_STRIDE = 10_000
"""Stage ``k`` is trained with seed ``seed + STAGE_SEED_STRIDE * k``."""

NUM_THREADS = 1
"""PyTorch threads during a curriculum, as in grading (small networks run faster on one)."""

TrainFn = Callable[[MARLEnv, BaseAlgorithmConfig, int, Any], tuple[Any, dict]]


class BudgetExceededError(RuntimeError):
    """A stage used more environment steps or time than its budget."""


class ZeroPolicy:
    """Reference policy that always returns the zero action (devices idle)."""

    def __init__(self, action_dim: int = 2) -> None:
        self.action_dim = action_dim

    def act(self, local_obs: NDArray) -> NDArray:
        return np.zeros(self.action_dim, dtype=np.float32)


@dataclass(frozen=True)
class StageReport:
    """Cost of one stage of a curriculum."""

    house: str
    steps: int
    seconds: float
    policy_bytes: int
    paths: dict


@dataclass(frozen=True)
class CurriculumResult:
    """Outcome of ``run_curriculum``.

    Attributes
    ----------
    policy
        Policy returned by the last stage.
    stage_houses : list[str]
        Household trained at each stage, in order.
    eval_houses : list[str]
        Households of the validation environment (columns of ``returns``).
    returns : NDArray
        ``(num_stages, num_eval_houses)`` mean episode return of every validation
        household after each stage.
    stages : list[StageReport]
        Steps, time and policy size of each stage.
    """

    policy: Any
    stage_houses: list[str]
    eval_houses: list[str]
    returns: NDArray
    stages: list[StageReport]


def episode_starts(env_config: dict) -> list[int]:
    """Return the weekly episode starts of ``env_config``, as the environment sees them."""
    return [int(s) for s in MARLEnv(env_config).valid_episode_starts]


def split_validation(env_config: dict, n_weeks: int = 8) -> tuple[dict, dict]:
    """Split the episodes of ``env_config`` into training and validation configs.

    The last ``n_weeks`` weekly episodes go to validation. The training environment
    cannot be reset to them (``scenarios.episode_starts``).
    """
    starts = episode_starts(env_config)
    if not 0 < n_weeks < len(starts):
        raise ValueError(f"n_weeks must be in [1, {len(starts) - 1}], got {n_weeks}")
    train_config, val_config = deepcopy(env_config), deepcopy(env_config)
    train_config["scenarios"]["episode_starts"] = starts[:-n_weeks]
    val_config["scenarios"]["episode_starts"] = starts[-n_weeks:]
    return train_config, val_config


def make_stage_env(env_config: dict, house: str) -> MARLEnv:
    """Return an environment with ``house`` only."""
    return MARLEnv(select_prosumers(env_config, [house]))


def _episode_returns(
    env: MARLEnv,
    joint_action: Callable[[NDArray], NDArray],
    seed: int,
    on_reset: Callable[[], None] = lambda: None,
) -> NDArray:
    """Run every episode of ``env`` once; ``(num_episodes, num_agents)`` returns."""
    starts = env.valid_episode_starts
    returns = np.zeros((len(starts), env.num_agents), dtype=np.float32)
    for i, start in enumerate(starts):
        obs, _ = env.reset(seed=seed + i, options={"episode_start": int(start)})
        on_reset()
        done = False
        while not done:
            obs, rewards, done, _, _ = env.step(joint_action(obs))
            returns[i] += rewards
    return returns


def evaluate_households(env: MARLEnv, policy: Any, seed: int = 1234) -> NDArray:
    """Run every episode of ``env`` once with the deterministic ``policy.act``.

    Households are independent, so one rollout gives the return of each of them.

    Returns
    -------
    NDArray
        ``(num_episodes, num_agents)`` episode returns.
    """
    return _episode_returns(env, lambda obs: _shared_policy_actions(env, policy, obs), seed)


def evaluate_expert_households(
    env: MARLEnv, experts: list[Expert], seed: int = 1234
) -> NDArray:
    """Run every episode of ``env`` once with the experts (same episodes as
    ``evaluate_households``).

    Returns
    -------
    NDArray
        ``(num_episodes, num_agents)`` episode returns.
    """

    def reset_experts() -> None:
        for expert in experts:
            expert.reset()

    return _episode_returns(env, lambda obs: expert_actions(env, experts), seed, reset_experts)


def check_stage_budget(
    steps: int, seconds: float, config: BaseAlgorithmConfig, house: str
) -> None:
    """Raise ``BudgetExceededError`` if a stage went over ``max_steps`` or ``max_time``."""
    if steps > config.max_steps:
        raise BudgetExceededError(
            f"Stage {config.stage} ({house}) used {steps} env steps; "
            f"the budget is {config.max_steps}."
        )
    if seconds > config.max_time:
        raise BudgetExceededError(
            f"Stage {config.stage} ({house}) took {seconds:.1f} s; "
            f"the budget is {config.max_time:.0f} s."
        )


def carry_policy(policy: Any, max_bytes: int = MAX_POLICY_BYTES) -> tuple[Any, int]:
    """Pickle and unpickle ``policy``, as between two graded stages.

    Returns the restored copy and its pickled size. Only what the policy object
    holds survives; environments cannot be pickled, so a policy that references
    one fails here.
    """
    try:
        blob = pickle.dumps(policy)
    except Exception as error:
        raise TypeError(
            "The policy returned by train() must be picklable and must not hold "
            f"a reference to the environment: {error}"
        ) from error
    if len(blob) > max_bytes:
        raise ValueError(
            f"The pickled policy is {len(blob) / 1e6:.1f} MB; "
            f"the limit is {max_bytes / 1e6:.0f} MB."
        )
    return pickle.loads(blob), len(blob)


def run_curriculum(
    train_fn: TrainFn,
    config: BaseAlgorithmConfig,
    train_config: dict,
    val_config: dict,
    seed: int = 0,
    houses: Sequence[str] | None = None,
    max_policy_bytes: int = MAX_POLICY_BYTES,
    eval_seed: int = 1234,
    verbose: bool = True,
) -> CurriculumResult:
    """Train one policy over a sequence of households, one household per stage.

    Parameters
    ----------
    train_fn : TrainFn
        ``train(env, config, seed, policy)`` returning ``(policy, paths)``.
    config : BaseAlgorithmConfig
        Hyperparameters; ``max_steps`` and ``max_time`` are per stage. Each stage
        receives a copy with ``stage`` set to its index.
    train_config, val_config : dict
        Environment configs for training and evaluation (see ``split_validation``).
    seed : int, default=0
        Seed of the first stage; stage ``k`` uses ``seed + STAGE_SEED_STRIDE * k``.
    houses : Sequence[str] | None, default=None
        Households to train on, in order. Defaults to the households of
        ``train_config``. Households of ``val_config`` that are not listed are
        evaluated without being trained on.
    max_policy_bytes : int, default=MAX_POLICY_BYTES
        Limit of the pickled policy carried between stages.
    eval_seed : int, default=1234
        Seed of the evaluation episodes.
    verbose : bool, default=True
        Print one line per stage.

    Raises
    ------
    BudgetExceededError
        If a stage uses more than ``config.max_steps`` steps or ``config.max_time``
        seconds.

    Notes
    -----
    PyTorch runs with ``NUM_THREADS`` threads during the curriculum, as in grading,
    so stage times match the graded ones.
    """
    houses = list(houses) if houses is not None else list(train_config["prosumers"])
    val_env = MARLEnv(val_config)
    eval_houses = list(val_env.agent_ids)
    if missing := sorted(set(houses) - set(eval_houses)):
        raise ValueError(f"Stage households missing from val_config: {missing}")

    threads = torch.get_num_threads()
    torch.set_num_threads(NUM_THREADS)
    try:
        policy, rows, reports = _run_stages(
            train_fn, config, train_config, val_env, houses, seed, max_policy_bytes,
            eval_seed, verbose,
        )
    finally:
        torch.set_num_threads(threads)

    return CurriculumResult(
        policy=policy,
        stage_houses=houses,
        eval_houses=eval_houses,
        returns=np.stack(rows),
        stages=reports,
    )


def _run_stages(
    train_fn: TrainFn,
    config: BaseAlgorithmConfig,
    train_config: dict,
    val_env: MARLEnv,
    houses: list[str],
    seed: int,
    max_policy_bytes: int,
    eval_seed: int,
    verbose: bool,
) -> tuple[Any, list[NDArray], list[StageReport]]:
    """Train and evaluate stage by stage; see ``run_curriculum``."""
    policy, rows, reports = None, [], []
    for k, house in enumerate(houses):
        env = make_stage_env(train_config, house)
        stage_config = replace(config, stage=k)
        start = perf_counter()
        policy, paths = train_fn(env, stage_config, seed + STAGE_SEED_STRIDE * k, policy)
        seconds = perf_counter() - start
        check_stage_budget(env.step_count, seconds, stage_config, house)
        policy, policy_bytes = carry_policy(policy, max_policy_bytes)

        rows.append(evaluate_households(val_env, policy, eval_seed).mean(axis=0))
        reports.append(StageReport(house, env.step_count, seconds, policy_bytes, paths))
        if verbose:
            print(
                f"Stage {k + 1}/{len(houses)} ({house}): {env.step_count} steps, "
                f"{seconds:.1f} s, policy {policy_bytes / 1e6:.2f} MB, "
                f"mean validation return {rows[-1].mean():.1f}"
            )
    return policy, rows, reports
