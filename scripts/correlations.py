"""Step 2 - Correlations and Pareto front with the same per-game objective
definitions as the bandit environment.

By default the ground-truth half is used (partition=truth, seed 0), i.e. exactly
the data that defines the reference front in the main experiment.

Usage:
    python scripts/correlations.py --csv data/PlayerStatistics.csv
    python scripts/correlations.py --csv data/PlayerStatistics.csv --partition all
"""

import argparse
import itertools
from pathlib import Path

import pandas as pd
from scipy import stats

from mobandit import config as C
from mobandit.data import per_game_table, zscore
from mobandit.pareto import pareto_front
from mobandit.plots import paper_projection_figure


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--partition", choices=["truth", "all"], default="truth")
    ap.add_argument("--seed", type=int, default=C.EXPERIMENT_SEED)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    games = per_game_table(args.csv, seed=args.seed,
                           partition="truth" if args.partition == "truth" else None)
    means = games.groupby("player")[C.OBJECTIVES].mean()
    z = zscore(means)
    front = pd.Series(pareto_front(z), index=means.index)

    print(f"\nPlayers analysed: {len(means)}")
    print("\nPairwise correlations (per-game definitions, matching the environment):")
    for a, b in itertools.combinations(C.OBJECTIVES, 2):
        r, p = stats.pearsonr(means[a], means[b])
        rho, _ = stats.spearmanr(means[a], means[b])
        print(f"  {a:<11s} vs {b:<11s}  Pearson r={r:+.3f} (p={p:.3f})   Spearman rho={rho:+.3f}")

    print(f"\nPareto front: {int(front.sum())} non-dominated players out of {len(means)}")
    conceded = z[front][C.OBJECTIVES].idxmin(axis=1)
    print("Among front players, the objective conceded is:")
    for obj in C.OBJECTIVES:
        print(f"  {obj:<10s} : {int((conceded == obj).sum())}")

    front_players = means[front].assign(conceded=conceded).sort_values("conceded")
    print("\nFront players:\n" + front_players.round(3).to_string())

    tag = f"correlations_pergame_{args.partition}"
    means.assign(on_pareto_front=front).to_csv(out / f"{tag}.csv")
    paper_projection_figure(means, front, out / f"{tag}_projections.png")
    print(f"\nSaved {tag}.csv and {tag}_projections.png to {out}/")


if __name__ == "__main__":
    main()
