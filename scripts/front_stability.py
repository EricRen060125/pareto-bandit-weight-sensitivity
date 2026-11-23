"""Step 5 - How stable is the reference Pareto front?

The ground truth is estimated from a random half of each player's games, so the
front depends on that split. This script repeats the split many times and
reports:
  1. the distribution of front sizes across splits
  2. each player's membership frequency (fraction of splits on the front)
  3. a core front: players on the front in at least a given fraction of splits

Usage:
    python scripts/front_stability.py --csv data/PlayerStatistics.csv --reps 200
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mobandit import config as C
from mobandit.data import per_game_table, split_truth_pool, zscore
from mobandit.pareto import pareto_front
from mobandit.plots import front_stability_plot


def front_for_seed(games, seed):
    """Split once under `seed`, compute ground-truth means, return the front."""
    truth, _ = split_truth_pool(games, seed=seed)
    means = truth.groupby("player")[C.OBJECTIVES].mean()
    return pd.Series(pareto_front(zscore(means)), index=means.index)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.5, help="core-front threshold")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # load and filter once; only the split is repeated
    games = per_game_table(args.csv, partition=None)
    print(f"Players: {games['player'].nunique()}   game rows: {len(games):,}\n")

    sizes, counts = [], None
    for s in range(args.reps):
        front = front_for_seed(games, seed=s)
        sizes.append(int(front.sum()))
        counts = front.astype(int) if counts is None else counts + front.astype(int)
        if (s + 1) % 25 == 0:
            print(f"  {s + 1}/{args.reps} splits done", flush=True)

    sizes = np.array(sizes)
    freq = (counts / args.reps).sort_values(ascending=False)

    lo, hi = np.percentile(sizes, [2.5, 97.5])
    print(f"\n== Front size across {args.reps} random splits ==")
    print(f"  mean   {sizes.mean():.1f}")
    print(f"  median {np.median(sizes):.0f}")
    print(f"  range  {sizes.min()} to {sizes.max()}")
    print(f"  sd     {sizes.std(ddof=1):.1f}")
    print(f"  95% of splits give between {lo:.0f} and {hi:.0f} players")

    core = freq[freq >= args.threshold]
    print("\n== Membership frequency ==")
    print(f"  appear in every split      : {int((freq == 1.0).sum())}")
    print(f"  appear in >= {args.threshold:.0%} of splits : {len(core)}")
    print(f"  appear at least once       : {int((freq > 0).sum())}")
    print(f"\n== Players appearing in at least {args.threshold:.0%} of splits ==")
    for name, f in core.items():
        print(f"  {name:28s} {f:.2f}")

    freq.to_csv(out / "front_stability_frequency.csv", header=["frequency"])
    front_stability_plot(sizes, freq, out / "front_stability.png")
    print(f"\nSaved front_stability.png and front_stability_frequency.csv to {out}/")


if __name__ == "__main__":
    main()
