from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import (
    apply_publication_style,
)

apply_publication_style()


def _log_folder(log_folder: str | Path | Mapping[str, str | Path]) -> Path:
    """Normalize a log directory or the save-path mapping returned by train()."""
    if isinstance(log_folder, Mapping):
        if not log_folder:
            raise ValueError("The model path mapping is empty.")
        return Path(next(iter(log_folder.values()))).parent
    return Path(log_folder)


def _extract_csv_data(
    log_folder: str | Path | Mapping[str, str | Path],
) -> dict[str, pd.DataFrame]:
    """Extract training and evaluation returns from a CSV log file.

    Usage hint
    ----------
    python plot_logs.py path/to/log.csv"""

    folder = _log_folder(log_folder)
    data = {
        csv_path.stem: pd.read_csv(csv_path)
        for csv_path in sorted(folder.glob("*.csv"))
    }
    if not data:
        raise FileNotFoundError(f"No CSV training logs found in {folder}")
    return data


def _format_axes(ax: plt.Axes, title: str | None = None):
    ax.set_xlabel("Steps")
    ax.set_ylabel("Return")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    ax.grid(False)
    ax.figure.tight_layout()


def _plot_single_learning_curve(
    steps: pd.Series,
    train_return: pd.Series,
    eval_return: pd.Series | None,
    ax: plt.Axes,
    agent: str | None = None,
):
    if agent is None:
        label_train = "Train"
        label_eval = "Validation"
    else:
        label_train = f"{agent} (train)"
        label_eval = f"{agent} (valid)"
    ax.plot(steps, train_return, label=label_train)

    if eval_return is not None:
        color = ax.get_lines()[-1].get_color()
        ax.plot(steps, eval_return, color=color, linestyle="--", label=label_eval)


def plot_learning_curves(
    log_folder: str | Path | Mapping[str, str | Path],
    title: str | None = None,
    *,
    show: bool = True,
) -> plt.Figure:
    """Plot the training and evaluation returns of the shared policy.

    The return is summed over all agents, so a single curve describes the run."""
    data = _extract_csv_data(log_folder)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, df in data.items():
        _plot_single_learning_curve(
            steps=df["step"],
            train_return=df["train_return"],
            eval_return=df["eval_return"] if "eval_return" in df.columns else None,
            ax=ax,
            agent=name if len(data) > 1 else None,
        )

    _format_axes(ax, title)
    if show:
        plt.show()
    return fig


def plot_reward_distribution(
    rewards,
    labels: list[str] | None = None,
) -> plt.Figure:
    """Violin plot of the episode returns of several methods on one axis.

    Each violin is the kernel density of one method's returns, with its median
    marked; the individual episodes are drawn on top as points.

    Arguments:
        rewards : [n_episodes] or [n_episodes, n_methods] array of episode returns.
        labels : one name per method (column of ``rewards``).
    """
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim == 1:
        rewards = rewards[:, None]
    n_methods = rewards.shape[1]
    if labels is not None and len(labels) != n_methods:
        raise ValueError(
            f"Got {len(labels)} labels for {n_methods} columns of rewards."
        )

    positions = np.arange(1, n_methods + 1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(max(5, 1.8 * n_methods), 4))

    parts = ax.violinplot(rewards, positions=positions, showmedians=True)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i % len(colors)])
        body.set_edgecolor("black")
        body.set_alpha(0.5)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("black")
        parts[key].set_linewidth(1.0)

    for i in range(n_methods):
        ax.scatter(
            np.full(rewards.shape[0], positions[i]),
            rewards[:, i],
            s=12,
            color="black",
            alpha=0.6,
            zorder=3,
        )

    ax.set_xticks(positions)
    if labels is not None:
        ax.set_xticklabels(labels)
    ax.set_ylabel("Global episode return")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-4, 4))
    ax.grid(False)
    fig.tight_layout()
    return fig
