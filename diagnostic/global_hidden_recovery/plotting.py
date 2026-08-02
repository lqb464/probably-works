from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def plot_margin_scatter(
    delta_global: Sequence[float],
    delta_hidden: Sequence[float],
    categories: Sequence[str],
    path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    delta_global = np.asarray(delta_global)
    delta_hidden = np.asarray(delta_hidden)

    colors = {
        "A": "#4c78a8",
        "B_hidden_recoverable_global_failure": "#54a24b",
        "C": "#9d755d",
        "D_harm": "#e45756",
        "tie": "#bab0ac",
    }

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for category, color in colors.items():
        mask = np.asarray([c == category for c in categories], dtype=bool)
        if mask.any():
            ax.scatter(
                delta_global[mask],
                delta_hidden[mask],
                s=8,
                alpha=0.55,
                c=color,
                label=category.replace("_hidden_recoverable_global_failure", ""),
                linewidths=0,
                rasterized=True,
            )

    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("delta_global")
    ax.set_ylabel("delta_hidden")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_recovery_funnel(
    path: Path,
    num_queries: int,
    fixed_pair: Mapping[str, object],
    top50: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    global_wrong = int(fixed_pair["global_failures"])
    global_correct = int(num_queries - global_wrong)
    hidden_harms = int(fixed_pair["category_D_harm"])
    hidden_agrees = int(fixed_pair["category_A"])

    inside_top50 = int(top50.get("num_global_failures_with_positive_in_topk", 0))
    outside_top50 = int(global_wrong - inside_top50)
    rescued = int(top50.get("num_rescued", 0))
    hidden_fails = int(inside_top50 - rescued)

    text = (
        f"All queries: {num_queries}\n"
        f"|-- Global correct: {global_correct}\n"
        f"|   |-- Hidden agrees: {hidden_agrees}\n"
        f"|   `-- Hidden harms: {hidden_harms}\n"
        f"`-- Global wrong: {global_wrong}\n"
        f"    |-- Positive outside Top-50: {outside_top50}\n"
        f"    `-- Positive inside Top-50: {inside_top50}\n"
        f"        |-- Hidden fails: {hidden_fails}\n"
        f"        `-- Hidden rescues: {rescued}\n"
    )
    path.write_text(text, encoding="utf-8")
