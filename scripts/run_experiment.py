"""Step 3 - Main experiment: weight sensitivity of Fixed-Weight vs. Pareto bandits.

Reproduces the paper's main results (T = 100,000, 30 runs, seed 0):
    python scripts/run_experiment.py --csv data/PlayerStatistics.csv

Robustness check without the strongest outlier:
    python scripts/run_experiment.py --csv data/PlayerStatistics.csv \
        --exclude "Nikola Jokic" --out-prefix results30_noJokic

Fixed-Weight methods with Thompson Sampling or Explore-Then-Commit instead of UCB:
    python scripts/run_experiment.py --csv data/PlayerStatistics.csv --fw ts

Quick test run:
    python scripts/run_experiment.py --csv data/PlayerStatistics.csv --T 5000 --repeats 2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mobandit import config as C
from mobandit.algorithms import PARETO_METHODS
from mobandit.env import build_env_from_csv
from mobandit.evaluation import (describe_picks, eval_weight_grid,
                                 pareto_identification_f1, run_experiment)
from mobandit.plots import experiment_plots


def identification_report(env, T, repeats, rng):
    """F1 of each Pareto method's estimated front against the true front."""
    n_true = int(env.pareto_mask.sum())
    print(f"\n== Pareto-set identification (F1 against the true front, {repeats} runs) ==")
    rows = {}
    for name, fn in PARETO_METHODS.items():
        ms = [pareto_identification_f1(env, fn(env, T, rng)[2]) for _ in range(repeats)]
        f1 = np.array([m["f1"] for m in ms])
        rows[name] = dict(f1_mean=f1.mean(), f1_std=f1.std(),
                          precision=np.mean([m["precision"] for m in ms]),
                          recall=np.mean([m["recall"] for m in ms]),
                          n_identified=np.mean([m["n_est"] for m in ms]), n_true=n_true)
        r = rows[name]
        print(f"  {name:12s} F1={r['f1_mean']:.2f}+/-{r['f1_std']:.2f}  "
              f"precision={r['precision']:.2f}  recall={r['recall']:.2f}  "
              f"(identified {r['n_identified']:.0f} on average, true {n_true})")
    return rows


def print_results(env, res, repeats):
    print("\n== Who each method selects ==")
    for nm, who in describe_picks(env, res["picks"]):
        print(f"  {nm:18s} -> {who}")

    print("\n== Regret if the TRUE preference turns out to be... ==")
    print("  " + " " * 18 + "".join(f"{s:>13s}" for s in res["scen_names"]))
    for i, nm in enumerate(res["method_names"]):
        print(f"  {nm:18s}" + "".join(f"{v:>13.4f}" for v in res["matrix"][i]))

    report = res["report"]
    print(f"\n== Weight sensitivity over {len(eval_weight_grid())} possible "
          f"preferences ({repeats} runs) ==")
    for nm, s in report.items():
        print(f"  {nm:18s} mean={s['mean']:.4f}  worst={s['worst']:.4f}  "
              f"sensitivity(std)={s['sensitivity']:.4f}")

    fw = [report[n]["mean"] for n in report if n.startswith("assume")]
    if min(fw) > 0:
        print(f"\n  Spread across preset weights: {max(fw) / min(fw):.1f}x "
              f"(best {min(fw):.4f} vs worst {max(fw):.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--T", type=int, default=C.HORIZON)
    ap.add_argument("--repeats", type=int, default=C.N_REPEATS)
    ap.add_argument("--seed", type=int, default=C.EXPERIMENT_SEED)
    ap.add_argument("--fw", choices=["ucb", "ts", "etc"], default="ucb",
                    help="algorithm for the Fixed-Weight methods (paper tables: ucb)")
    ap.add_argument("--exclude", nargs="*", default=None, help="player names to drop")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    default_prefix = f"results{args.repeats}" + ("" if args.fw == "ucb" else f"_{args.fw}")
    prefix = str(out / (args.out_prefix or default_prefix))

    env = build_env_from_csv(args.csv, seed=args.seed, exclude_players=args.exclude)
    rng = np.random.default_rng(args.seed)
    # Order matters for exact reproducibility: the identification runs consume
    # the RNG before the main experiment, as in the original analysis.
    ident = identification_report(env, args.T, args.repeats, rng)
    res = run_experiment(env, args.T, rng, repeats=args.repeats, fw=args.fw)
    print_results(env, res, args.repeats)

    experiment_plots(res, prefix)
    pd.DataFrame(res["report"]).T.to_csv(f"{prefix}_sensitivity.csv")
    pd.DataFrame(res["matrix"], index=res["method_names"],
                 columns=[f"true: {s}" for s in res["scen_names"]]).to_csv(f"{prefix}_matrix.csv")
    with open(f"{prefix}_identification.json", "w") as f:
        json.dump(ident, f, indent=2, default=float)
    print(f"\nSaved figures and tables with prefix {prefix}_*")


if __name__ == "__main__":
    main()
