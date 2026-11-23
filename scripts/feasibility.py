"""Step 1 - Feasibility check: do offense, defense and playmaking really conflict?

The study only makes sense if the three objectives form a genuine trade-off.
This script tests that instead of assuming it, using three criteria:
  (1) objectives are distinct: no pair is strongly positively correlated;
  (2) the Pareto front is large enough to be a real identification task;
  (3) front players sacrifice different objectives ("pick two" variety).
Correlation is reported as auxiliary evidence; the front itself is the arbiter.

Usage:
    python scripts/feasibility.py --csv data/PlayerStatistics.csv
"""

import argparse
import itertools
from pathlib import Path

import pandas as pd
from scipy import stats

from mobandit import config as C
from mobandit.data import aggregate_ts_means, filter_players, load_games, split_truth_pool, zscore
from mobandit.pareto import pareto_front
from mobandit.plots import feasibility_plots


def correlation_report(means):
    print("\nPairwise correlation of ground-truth means (auxiliary evidence):")
    for a, b in itertools.combinations(C.OBJECTIVES, 2):
        r, p = stats.pearsonr(means[a], means[b])
        rho, _ = stats.spearmanr(means[a], means[b])
        tag = "conflict-leaning" if r < 0.1 else ("weak" if r < 0.4 else "aligned (bad sign)")
        print(f"    {a:<10} vs {b:<10}: Pearson r={r:+.2f} (p={p:.3f}), "
              f"Spearman rho={rho:+.2f}   -> {tag}")


def sacrificed_profile(means_z, front):
    """Among front players, which objective is each one's weakest?"""
    counts = means_z[front][C.OBJECTIVES].idxmin(axis=1).value_counts()
    print("\nAmong Pareto-front players, the objective sacrificed (weakest):")
    for obj in C.OBJECTIVES:
        print(f"    sacrifices {obj:<10}: {int(counts.get(obj, 0))} players")
    return int((counts > 0).sum())


def verdict(means, front, spread):
    n, n_front = len(means), int(front.sum())
    max_pos = max(stats.pearsonr(means[a], means[b])[0]
                  for a, b in itertools.combinations(C.OBJECTIVES, 2))
    print("\n" + "=" * 68 + "\nFEASIBILITY VERDICT\n" + "=" * 68)
    print(f"Players analysed             : {n}")
    print(f"Pareto front (non-dominated) : {n_front}  ({n_front / n:.0%} of pool)")
    print(f"Objectives sacrificed spread : {spread}/3 categories")
    print(f"Strongest + pairwise corr    : {max_pos:+.2f} (want < 0.50)")

    distinct, rich, spread_ok = max_pos < 0.50, n_front >= 12, spread >= 2
    if distinct and rich and spread_ok:
        print("\nVERDICT: VIABLE - a genuine three-objective trade-off is present.")
    elif n_front >= 6 and spread_ok:
        print("\nVERDICT: MARGINAL - a trade-off exists but one condition is weak.")
    else:
        print("\nVERDICT: WEAK - the objectives largely align or the front is tiny.")
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True, help="path to PlayerStatistics.csv")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--seed", type=int, default=C.FEASIBILITY_SEED)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    games = filter_players(load_games(args.csv))
    truth_games, pool_games = split_truth_pool(games, seed=args.seed)
    print(f"Split: {len(truth_games)} game rows -> ground truth, "
          f"{len(pool_games)} -> bandit pool (no overlap).")

    means = aggregate_ts_means(truth_games)
    means_z = zscore(means)
    correlation_report(means)
    front = pd.Series(pareto_front(means_z), index=means_z.index)
    verdict(means, front, sacrificed_profile(means_z, front))

    feasibility_plots(means, means_z, front, prefix=str(out / "feasibility"))
    means.assign(on_pareto_front=front).to_csv(out / "feasibility_ground_truth.csv")
    print(f"\nSaved plots and feasibility_ground_truth.csv to {out}/")


if __name__ == "__main__":
    main()
