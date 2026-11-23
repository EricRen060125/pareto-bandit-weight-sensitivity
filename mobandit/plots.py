"""Figures for the feasibility check and the main experiment."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C

FW_COLOR, PARETO_COLOR = "#d64550", "#3b7dd8"
PAIRS = [("offense", "defense"), ("offense", "playmaking"), ("defense", "playmaking")]


def _save(fig, path, dpi=130):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# --- Feasibility --------------------------------------------------------------
def feasibility_plots(means, means_z, front, prefix):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (a, b) in zip(axes, PAIRS):
        ax.scatter(means[a][~front], means[b][~front], s=28, c="#9aa0a6",
                   alpha=0.7, label="dominated")
        ax.scatter(means[a][front], means[b][front], s=60, c=FW_COLOR,
                   edgecolor="k", linewidth=0.5, label="Pareto front")
        ax.set_xlabel(a); ax.set_ylabel(b); ax.set_title(f"{a} vs {b}")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Pairwise projections of per-player means (ground truth)")
    _save(fig, f"{prefix}_projections.png")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for mask, kw in [(~front, dict(c="#9aa0a6", alpha=0.6, label="dominated")),
                     (front, dict(c=FW_COLOR, s=55, edgecolor="k", label="Pareto front"))]:
        ax.scatter(means_z["offense"][mask], means_z["defense"][mask],
                   means_z["playmaking"][mask], **kw)
    ax.set_xlabel("offense (z)"); ax.set_ylabel("defense (z)"); ax.set_zlabel("playmaking (z)")
    ax.set_title("Three-objective space (standardized)")
    ax.legend(fontsize=8)
    _save(fig, f"{prefix}_3d.png")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    corr = means[C.OBJECTIVES].corr()
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu")
    ax.set_xticks(range(3)); ax.set_xticklabels(C.OBJECTIVES, rotation=30, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(C.OBJECTIVES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title("Correlation of objective means")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, f"{prefix}_corr.png")


def paper_projection_figure(means, front, path):
    """Pairwise projections with the Pareto front highlighted (paper figure)."""
    labels = {"offense": "Offense (efficiency)",
              "defense": "Defense (per 36 min)",
              "playmaking": "Playmaking (per 36 min)"}
    ticks = {"offense": [0.4, 0.5, 0.6, 0.7],
             "defense": [1, 2, 3, 4, 5, 6],
             "playmaking": [2.5, 5.0, 7.5, 10.0, 12.5]}
    with plt.rc_context({"font.size": 20}):
        _paper_projection(means, front, labels, ticks, path)


def _paper_projection(means, front, labels, ticks, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (a, b) in zip(axes, PAIRS):
        ax.scatter(means[a][~front], means[b][~front], s=22, c="#9aa0a6",
                   alpha=.55, linewidth=0, label="Dominated")
        ax.scatter(means[a][front], means[b][front], s=62, c=FW_COLOR,
                   edgecolor="k", linewidth=.5, zorder=3, label="Pareto front")
        ax.set_xlabel(labels[a]); ax.set_ylabel(labels[b])
        ax.set_xticks(ticks[a]); ax.set_yticks(ticks[b])
        ax.grid(alpha=.25)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, fontsize=16, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.08), frameon=False)
    _save(fig, path, dpi=200)


# --- Main experiment ----------------------------------------------------------
def experiment_plots(res, prefix):
    report, curves, matrix = res["report"], res["curves"], res["matrix"]
    scen_names, names = res["scen_names"], res["method_names"]
    is_fw = [n.startswith("assume") for n in names]
    colors = [FW_COLOR if f else PARETO_COLOR for f in is_fw]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(scen_names)))
    ax.set_xticklabels([f"true: {s}" for s in scen_names], rotation=20, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("Cost of guessing the preference wrong (lower is better)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="selection regret")
    _save(fig, f"{prefix}_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nm, f in zip(names, is_fw):
        ax.plot(np.sort(curves[nm]), label=nm, color=FW_COLOR if f else PARETO_COLOR,
                linestyle="--" if f else "-", linewidth=1.4 if f else 2)
    ax.set_xlabel("possible true preferences (sorted by regret)")
    ax.set_ylabel("selection regret")
    ax.set_title("Regret under unknown preferences: Fixed-Weight (dashed) vs Pareto (solid)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    _save(fig, f"{prefix}_curves.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, [report[n]["mean"] for n in names],
           yerr=[report[n]["sensitivity"] for n in names], color=colors, capsize=4)
    ax.set_ylabel("mean regret across preferences (± std)")
    ax.set_title("Mean regret and its spread under unknown preferences")
    plt.xticks(rotation=30, ha="right")
    _save(fig, f"{prefix}_meanregret.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, [report[n]["worst"] for n in names], color=colors)
    ax.set_ylabel("worst-case regret over preferences")
    ax.set_title("Worst case when the preference guess is wrong")
    plt.xticks(rotation=30, ha="right")
    _save(fig, f"{prefix}_worst.png")


def sem_plot(sem_curves, runs, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = list(range(2, runs + 1))
    for nm, c in sem_curves.items():
        f = nm.startswith("assume")
        ax.plot(xs, c, "--" if f else "-", color=FW_COLOR if f else PARETO_COLOR,
                label=nm, linewidth=1.6)
    ax.set_xlabel("number of repetitions")
    ax.set_ylabel("standard error of the mean regret")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    _save(fig, path, dpi=200)


def front_stability_plot(sizes, freq, path, top_n=30):
    """Left: front size across splits. Right: the top_n most frequent members."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].hist(sizes, bins=range(sizes.min(), sizes.max() + 2),
                 color=PARETO_COLOR, edgecolor="white")
    axes[0].set_xlabel("Players on the front")
    axes[0].set_ylabel("Number of splits")
    top = freq[freq > 0][:top_n]
    axes[1].barh(range(len(top)), top.values, color=FW_COLOR)
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels(top.index, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Fraction of splits on the front")
    _save(fig, path, dpi=200)
