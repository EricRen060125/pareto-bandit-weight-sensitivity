"""Loading, filtering and splitting the NBA box-score data."""

import sys

import numpy as np
import pandas as pd

from . import config as C


def _derive_season(dt):
    """NBA season label = starting year (Oct-Dec -> that year, Jan-Sep -> year - 1).
    A game on 2025-06-13 therefore belongs to season 2024 (the 2024-25 season)."""
    return np.where(dt.dt.month >= 10, dt.dt.year, dt.dt.year - 1)


def load_games(csv_path):
    """Read PlayerStatistics.csv and return one row per player-game for the
    target season, with defense and playmaking as per-36-minute rates.

    The `offense` column here is a per-36 points placeholder; the two ways
    offense is actually defined are applied later by `add_per_game_offense`
    (bandit environment) and `aggregate_ts_means` (feasibility check).
    """
    usecols = list(C.COLS.values())
    try:
        df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    except ValueError as e:
        head = pd.read_csv(csv_path, nrows=0)
        sys.exit(f"[ERROR] {e}\nColumns in file: {list(head.columns)}\n"
                 f"Edit COLS in mobandit/config.py to match.")
    print(f"Loaded {len(df):,} player-game rows (all seasons).")

    dt = pd.to_datetime(df[C.COLS["gamedate"]], errors="coerce")
    df = df.assign(_season=_derive_season(dt))
    seen_types = df[C.COLS["gametype"]].dropna().unique()
    df = df[(df["_season"] == C.SEASON) & (df[C.COLS["gametype"]] == C.GAME_TYPE)]
    print(f"Scope: season {C.SEASON}-{C.SEASON + 1}, gameType='{C.GAME_TYPE}' "
          f"-> {len(df):,} rows.")
    if len(df) == 0:
        sys.exit(f"[ERROR] No rows after filtering. gameType values seen: "
                 f"{list(seen_types)}. Check SEASON and GAME_TYPE in config.py.")

    num = lambda col: pd.to_numeric(df[C.COLS[col]], errors="coerce")
    out = pd.DataFrame({
        "player": (df[C.COLS["first"]].astype(str).str.strip() + " "
                   + df[C.COLS["last"]].astype(str).str.strip()),
        "minutes": num("minutes"),
        "offense": num("pts"),
        "defense": num("stl") + num("blk"),
        "playmaking": num("ast"),
        "_pts": num("pts"),
        "_fga": num("fga"),
        "_fta": num("fta"),
    })

    # drop DNP / injury rows (no minutes recorded)
    before = len(out)
    out = out[out["minutes"] > 0].dropna()
    print(f"Removed {before - len(out):,} DNP/zero-minute rows -> {len(out):,} games kept.")

    scale = 36.0 / out["minutes"]
    out["offense"] = out["offense"] * scale
    out["defense"] = out["defense"] * scale
    out["playmaking"] = out["playmaking"] * scale
    return out.reset_index(drop=True)


def filter_players(games):
    """Keep players with at least MIN_GAMES games and MIN_MPG minutes per game."""
    per = games.groupby("player").agg(n_games=("offense", "size"),
                                      mpg=("minutes", "mean"))
    keep = per[(per["n_games"] >= C.MIN_GAMES) & (per["mpg"] >= C.MIN_MPG)].index
    kept = games[games["player"].isin(keep)].copy()
    print(f"Player filter (>= {C.MIN_GAMES} games AND >= {C.MIN_MPG} MPG): "
          f"{len(keep)} of {games['player'].nunique()} players kept.")
    return kept


def split_truth_pool(games, seed):
    """Split each player's games: a 'truth' half for the ground truth and a
    'pool' half for the bandit to sample from. No game appears in both."""
    rng = np.random.default_rng(seed)
    truth_idx, pool_idx = [], []
    for _, grp in games.groupby("player"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        cut = int(round(len(idx) * C.GT_FRACTION))
        truth_idx.extend(idx[:cut])
        pool_idx.extend(idx[cut:])
    return games.loc[truth_idx].copy(), games.loc[pool_idx].copy()


def add_per_game_offense(games):
    """Per-game True Shooting-style efficiency, so every objective has a clean
    per-game value (used by the bandit environment)."""
    denom = 2.0 * (games["_fga"] + 0.44 * games["_fta"])
    return games.assign(offense=np.where(denom > 0, games["_pts"] / denom, 0.0))


def aggregate_ts_means(truth_games):
    """Per-player means where offense is True Shooting% computed as a ratio of
    sums over the truth half (used by the feasibility check)."""
    means = truth_games.groupby("player")[C.OBJECTIVES].mean()
    agg = truth_games.groupby("player").agg(
        _pts=("_pts", "sum"), _fga=("_fga", "sum"), _fta=("_fta", "sum"))
    denom = 2.0 * (agg["_fga"] + 0.44 * agg["_fta"])
    means["offense"] = agg["_pts"] / denom.replace(0, np.nan)
    return means.dropna(subset=["offense"])


def per_game_table(csv_path, seed=C.EXPERIMENT_SEED, partition=None):
    """Filtered per-game rows with per-game offense, matching the bandit
    environment. With partition='truth', only the ground-truth half is kept."""
    games = add_per_game_offense(filter_players(load_games(csv_path)))
    if partition == "truth":
        truth, _ = split_truth_pool(games, seed=seed)
        print(f"  [partition] using the ground-truth half: "
              f"{len(truth):,} of {len(games):,} game rows")
        return truth
    return games


def zscore(df):
    return (df - df.mean()) / df.std(ddof=0)
