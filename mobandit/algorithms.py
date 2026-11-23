"""Bandit algorithms.

Fixed-Weight methods collapse the 3-D reward into one number with a preset
weight vector w, then run a standard single-objective bandit.
Pareto methods use no weight: each round they pull a random arm from the
non-dominated set of their (optimistic or sampled) estimates.

Every function returns (chosen_arms, pull_counts, estimated_means).
"""

import numpy as np

from . import config as C
from .pareto import pareto_front


# --- Fixed-Weight methods -----------------------------------------------------
def run_ucb_scalar(env, w, T, rng):
    """UCB1 on the scalarized reward w . r (used for the paper's result tables)."""
    n = env.n_arms
    counts = np.zeros(n)
    means = np.zeros(n)
    chosen = np.zeros(T, dtype=np.int64)
    for t in range(T):
        if t < n:
            a = t                                    # pull each arm once
        else:
            bonus = np.sqrt(2 * np.log(t) / counts)
            a = int(np.argmax(means + bonus))
        r = env.pull(a, rng) @ w
        counts[a] += 1
        means[a] += (r - means[a]) / counts[a]
        chosen[t] = a
    return chosen, counts, means


def run_ts_scalar(env, w, T, rng):
    """Gaussian Thompson Sampling on the scalarized reward."""
    n = env.n_arms
    counts = np.zeros(n)
    means = np.zeros(n)
    chosen = np.zeros(T, dtype=np.int64)
    for t in range(T):
        var = 1.0 / (counts + 1.0)
        a = int(np.argmax(means + rng.standard_normal(n) * np.sqrt(var)))
        r = env.pull(a, rng) @ w
        counts[a] += 1
        means[a] += (r - means[a]) / counts[a]
        chosen[t] = a
    return chosen, counts, means


def run_etc_scalar(env, w, T, rng, m=C.ETC_EXPLORE_PER_ARM):
    """Explore-Then-Commit on the scalarized reward."""
    n = env.n_arms
    counts = np.zeros(n)
    means = np.zeros(n)
    chosen = np.zeros(T, dtype=np.int64)
    for t in range(T):
        a = t % n if t < n * m else int(np.argmax(means))
        r = env.pull(a, rng) @ w
        counts[a] += 1
        means[a] += (r - means[a]) / counts[a]
        chosen[t] = a
    return chosen, counts, means


# --- Pareto methods (weight-free) ---------------------------------------------
def run_pareto_ucb1(env, T, rng):
    """Pareto UCB1 (Drugan & Nowe, 2013): pull uniformly at random from the
    non-dominated set of mean + exploration-bonus vectors."""
    n = env.n_arms
    D = env.truth_means.shape[1]
    counts = np.zeros(n)
    means = np.zeros((n, D))
    chosen = np.zeros(T, dtype=np.int64)
    for t in range(T):
        if t < n:
            a = t
        else:
            bonus = np.sqrt(2 * np.log(t * (D ** 0.25)) / counts)[:, None]
            cand = np.where(pareto_front(means + bonus))[0]
            a = int(cand[rng.integers(len(cand))])
        r = env.pull(a, rng)
        counts[a] += 1
        means[a] += (r - means[a]) / counts[a]
        chosen[t] = a
    return chosen, counts, means


def run_pareto_ts(env, T, rng):
    """Pareto Thompson Sampling: sample a mean vector per arm (Gaussian per
    objective), then pull uniformly at random from the non-dominated samples."""
    n = env.n_arms
    D = env.truth_means.shape[1]
    counts = np.zeros(n)
    means = np.zeros((n, D))
    chosen = np.zeros(T, dtype=np.int64)
    for t in range(T):
        var = (1.0 / (counts + 1.0))[:, None]
        sample = means + rng.standard_normal((n, D)) * np.sqrt(var)
        cand = np.where(pareto_front(sample))[0]
        a = int(cand[rng.integers(len(cand))])
        r = env.pull(a, rng)
        counts[a] += 1
        means[a] += (r - means[a]) / counts[a]
        chosen[t] = a
    return chosen, counts, means


FIXED_WEIGHT_METHODS = {"ucb": run_ucb_scalar, "ts": run_ts_scalar, "etc": run_etc_scalar}
PARETO_METHODS = {"Pareto UCB1": run_pareto_ucb1, "Pareto TS": run_pareto_ts}
