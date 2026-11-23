"""Bandit environment: one arm per player."""

import numpy as np

from . import config as C
from .data import add_per_game_offense, filter_players, load_games, split_truth_pool
from .pareto import pareto_front


class NBAEnv:
    """Each arm holds a pool of per-game reward vectors (normalized to [0, 1]).
    Pulling an arm returns one randomly drawn game from its pool. The ground
    truth (means and Pareto front) comes from the held-out truth half."""

    def __init__(self, pool_rewards, truth_means, names=None):
        self.pool = pool_rewards
        self.truth_means = truth_means
        self.n_arms = len(pool_rewards)
        self.names = names if names is not None else [f"arm{i}" for i in range(self.n_arms)]
        self.pareto_mask = pareto_front(truth_means)
        self.pareto_set = set(np.where(self.pareto_mask)[0])

    def pull(self, arm, rng):
        games = self.pool[arm]
        return games[rng.integers(len(games))]


def build_env_from_csv(csv_path, seed=C.EXPERIMENT_SEED, exclude_players=None):
    """Build the environment from PlayerStatistics.csv.

    All three objectives are min-max normalized to [0, 1] (Pareto UCB1 assumes
    rewards in [0, 1]^D, and it makes the objectives comparable). Min-max is
    affine, so the normalized ground-truth mean equals the mean of the
    normalized per-game rewards, keeping the environment consistent with the
    reference front.
    """
    games = filter_players(load_games(csv_path))

    # optional robustness check: drop specific players and re-run everything
    if exclude_players:
        before = games["player"].nunique()
        games = games[~games["player"].isin(exclude_players)]
        print(f"  [robustness] excluded {before - games['player'].nunique()} "
              f"player(s): {', '.join(exclude_players)}")

    games = add_per_game_offense(games)
    truth_games, pool_games = split_truth_pool(games, seed=seed)

    objs = C.OBJECTIVES
    allv = games[objs].to_numpy()
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    norm = lambda M: (M - lo) / span

    players = sorted(games["player"].unique())
    pid = {p: i for i, p in enumerate(players)}
    pool_rewards = [None] * len(players)
    truth_means = np.zeros((len(players), len(objs)))
    for p, grp in pool_games.groupby("player"):
        pool_rewards[pid[p]] = norm(grp[objs].to_numpy())
    for p, grp in truth_games.groupby("player"):
        truth_means[pid[p]] = norm(grp[objs].to_numpy()).mean(axis=0)

    # guard: every arm needs a non-empty pool
    for i, pr in enumerate(pool_rewards):
        if pr is None or len(pr) == 0:
            pool_rewards[i] = truth_means[i][None, :]

    env = NBAEnv(pool_rewards, truth_means, names=players)
    print(f"Env built: {env.n_arms} arms, {len(env.pareto_set)} on the true Pareto front.")
    return env


def make_synthetic_env(n_arms=40, seed=0):
    """Small synthetic environment for tests (no data download needed)."""
    rng = np.random.default_rng(seed)
    pool, truth = [], []
    for _ in range(n_arms):
        base = rng.uniform(0.2, 0.8, size=3)
        n_games = rng.integers(20, 40)
        pool.append(np.clip(base + 0.15 * rng.standard_normal((n_games, 3)), 0, 1))
        truth.append(base)
    return NBAEnv(pool, np.array(truth))
