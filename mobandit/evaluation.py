"""Evaluation: Pareto-front identification and weight sensitivity.

Core idea of the paper: a Fixed-Weight method must commit to a weight before
training and then selects a single player. A Pareto method uses no weight and
returns the whole set of players it believes are non-dominated. Each selection
is then scored against a grid of possible "true but unknown" preferences.
A method's weight sensitivity is the spread (std) of its regret over that grid.
"""

from collections import Counter

import numpy as np

from . import config as C
from .algorithms import FIXED_WEIGHT_METHODS, PARETO_METHODS
from .pareto import pareto_front


def pareto_identification_f1(env, means):
    """Compare the front estimated from learned means with the true front.
    This measures whether a method recognizes the front, not how often it
    samples it (which would unfairly penalize methods that spread their pulls)."""
    est = pareto_front(means)
    true = env.pareto_mask
    tp = int(np.sum(est & true))
    fp = int(np.sum(est & ~true))
    fn = int(np.sum(~est & true))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(precision=prec, recall=rec, f1=f1, n_est=int(est.sum()))


def scalarized_regret(chosen, env, w):
    """Cumulative regret against the best arm under weight w."""
    scal = env.truth_means @ w
    return np.cumsum(scal.max() - scal[chosen])


def selection_regret(env, arms, w_eval):
    """Regret of a selection (one or more arms) under preference w_eval,
    relative to the best single arm under that preference. A Pareto method's
    selected front is scored by the average of its members."""
    scal = env.truth_means @ w_eval
    return float(scal.max() - scal[list(arms)].mean())


def eval_weight_grid(step=0.1, lo=0.1):
    """Weight vectors on the 3-simplex with every component >= lo (36 vectors
    with the defaults): the possible true preferences a decision maker might hold."""
    grid = []
    vals = np.round(np.arange(lo, 1.0, step), 3)
    for a in vals:
        for b in vals:
            c = round(1.0 - a - b, 3)
            if lo - 1e-9 <= c <= 1.0:
                grid.append(np.array([a, b, c]))
    return grid


def method_names():
    return [f"assume {k}" for k in C.WEIGHTS] + list(PARETO_METHODS)


def train_once(env, T, rng, fw="ucb"):
    """Train every method once and return what each one selects.
    `fw` picks the single-objective algorithm used by the Fixed-Weight methods
    ('ucb' in the paper's results; 'ts' and 'etc' are also available)."""
    fw_algo = FIXED_WEIGHT_METHODS[fw]
    sel = {}
    for label, w0 in C.WEIGHTS.items():
        _, _, means = fw_algo(env, w0, T, rng)
        sel[f"assume {label}"] = [int(np.argmax(means))]
    for label, fn in PARETO_METHODS.items():
        _, _, means = fn(env, T, rng)
        est = np.where(pareto_front(means))[0]
        sel[label] = est.tolist() if len(est) else [0]
    return sel


def run_experiment(env, T, rng, repeats, fw="ucb"):
    """Train all methods `repeats` times. The selection table and the
    sensitivity statistics come from the same training runs.

    Returns a dict with
      report  : per-method mean / worst / sensitivity over the full grid
      curves  : per-method regret at every grid point (averaged over runs)
      matrix  : mean regret, method (row) x named preset scenario (column)
      picks   : what each method selected in every run
    """
    scenarios = C.WEIGHTS
    scen_names = list(scenarios)
    names = method_names()
    grid = eval_weight_grid()

    curves = {nm: np.zeros(len(grid)) for nm in names}
    matrix = np.zeros((len(names), len(scen_names)))
    picks = {nm: [] for nm in names}

    for _ in range(repeats):
        sel = train_once(env, T, rng, fw=fw)
        for i, nm in enumerate(names):
            picks[nm].append(sel[nm])
            for g, w in enumerate(grid):
                curves[nm][g] += selection_regret(env, sel[nm], w)
            for j, sc in enumerate(scen_names):
                matrix[i, j] += selection_regret(env, sel[nm], scenarios[sc])

    for nm in curves:
        curves[nm] /= repeats
    matrix /= repeats

    report = {nm: dict(mean=float(c.mean()), worst=float(c.max()),
                       sensitivity=float(c.std()))
              for nm, c in curves.items()}
    return dict(report=report, curves=curves, matrix=matrix,
                scen_names=scen_names, method_names=names, picks=picks)


def describe_picks(env, picks):
    """Readable summary of each method's selection (most frequent player for
    Fixed-Weight methods, average front size for Pareto methods)."""
    rows = []
    for nm, runs in picks.items():
        if all(len(r) == 1 for r in runs):
            arm, freq = Counter(r[0] for r in runs).most_common(1)[0]
            who = f"{env.names[arm]}  ({freq}/{len(runs)} runs)"
        else:
            last = runs[-1]
            who = (f"{np.mean([len(r) for r in runs]):.0f} players on average; e.g. "
                   + ", ".join(env.names[a] for a in last[:3]) + "...")
        rows.append((nm, who))
    return rows
