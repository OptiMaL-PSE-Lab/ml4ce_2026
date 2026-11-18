"""Plot of the stage-by-household returns of a curriculum."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .style import apply_publication_style

apply_publication_style()

if TYPE_CHECKING:
    from course_package.utils.helpers.transfer import CurriculumResult


def plot_transfer_matrix(
    result: "CurriculumResult", title: str | None = None
) -> Figure:
    """Heatmap of the validation return of every household after every stage.

    Row ``k`` is the policy after stage ``k``; boxed cells are the household trained
    at that stage. Columns marked ``(not trained)`` were never trained on.
    """
    R = result.returns
    n_rows, n_cols = R.shape
    labels = [
        h if h in result.stage_houses else f"{h}\n(not trained)"
        for h in result.eval_houses
    ]

    fig, ax = plt.subplots(figsize=(1.3 * n_cols + 2.5, 0.6 * n_rows + 1.8))
    image = ax.imshow(R, cmap="Wistia", aspect="auto")
    ax.set_xticks(range(n_cols), labels)
    ax.set_yticks(range(n_rows), [f"after {h}" for h in result.stage_houses])
    ax.set_xlabel("Validation household")
    ax.set_ylabel("Stage")
    ax.grid(False)

    threshold = np.nanmean(R)
    for i in range(n_rows):
        for j in range(n_cols):
            color = "black" if R[i, j] > threshold else "white"
            ax.text(
                j,
                i,
                f"{R[i, j]:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    for k, house in enumerate(result.stage_houses):
        j = result.eval_houses.index(house)
        ax.add_patch(
            Rectangle(
                (j - 0.5, k - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2
            )
        )

    fig.colorbar(image, ax=ax, label="Mean episode return")
    ax.set_title(title or "Validation return after each stage")
    fig.tight_layout()
    plt.show()
    return fig
