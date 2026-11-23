"""Step 4 - Are 30 repetitions enough?

Trains every method R times and tracks how the standard error of the mean
regret evolves as runs accumulate. If it flattens well before the last run,
the number of repetitions is adequate.

Usage:
    python scripts/convergence.py --csv data/PlayerStatistics.csv --runs 30
"""

import argparse
from pathlib import Path

import numpy as np

from mobandit import config as C
from mobandit.env import build_env_from_csv
from mobandit.evaluation import eval_weight_grid, method_names, selection_regret, train_once
from mobandit.plots import sem_plot


def run(env, T, runs, seed):
    rng = np.random.default_rng(seed)
    grid = eval_weight_grid()
    names = method_names()
    per_run = {nm: [] for nm in names}
    for r in range(runs):
        sel = train_once(env, T, rng)
        for nm in names:
            per_run[nm].append(float(np.mean([selection_regret(env, sel[nm], w) for w in grid])))
        print(f"  run {r + 1}/{runs} done", flush=True)

    print("\n== Standard error of the mean as runs accumulate ==")
    print("  runs " + "".join(f"{nm:>18s}" for nm in names))
    sem_curves = {nm: [] for nm in names}
    for k in range(2, runs + 1):
        row = f"  {k:>4d} "
        for nm in names:
            vals = np.array(per_run[nm][:k])
            sem = vals.std(ddof=1) / np.sqrt(k)
            sem_curves[nm].append(sem)
            row += f"{sem:>18.5f}"
        print(row)

    print("\n== Final estimate after all runs ==")
    for nm in names:
        vals = np.array(per_run[nm])
        mean, sem = vals.mean(), vals.std(ddof=1) / np.sqrt(runs)
        print(f"  {nm:18s} mean={mean:.4f}  sem={sem:.5f}  "
              f"95% CI=[{mean - 1.96 * sem:.4f}, {mean + 1.96 * sem:.4f}]")
    return per_run, sem_curves


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--runs", type=int, default=C.N_REPEATS)
    ap.add_argument("--T", type=int, default=C.HORIZON)
    ap.add_argument("--seed", type=int, default=C.EXPERIMENT_SEED)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    env = build_env_from_csv(args.csv, seed=args.seed)
    _, sem_curves = run(env, args.T, args.runs, args.seed)
    path = out / f"convergence{args.runs}_sem.png"
    sem_plot(sem_curves, args.runs, path)
    print(f"\nFigure saved: {path}")


if __name__ == "__main__":
    main()
