"""Experiment configuration.

Every constant that affects the results lives here, so the setup reported in
the paper can be read in one place.
"""

import numpy as np

# --- Data scope ---------------------------------------------------------------
SEASON = 2024                   # 2024 = the 2024-25 NBA season (Oct 2024 - Jun 2025)
GAME_TYPE = "Regular Season"    # playoffs excluded: fewer games, different context

# --- Player selection ---------------------------------------------------------
# Players are kept by a stability rule rather than a fixed count:
# enough games to split in half and still estimate a mean, and enough minutes
# that box-score noise does not dominate. With 2024-25 data this keeps 286 players.
MIN_GAMES = 40
MIN_MPG = 15.0

# --- Ground truth vs. bandit pool --------------------------------------------
# Each player's games are split in half. The ground-truth Pareto front is
# computed only from the "truth" half; the bandit samples only from the "pool"
# half. Same distribution, no overlap, no leakage.
GT_FRACTION = 0.5

# --- Objectives ---------------------------------------------------------------
# Fixed before looking at the data (not chosen because they conflict most):
#   offense    = True Shooting-style efficiency  PTS / (2 * (FGA + 0.44 * FTA))
#   defense    = (steals + blocks) per 36 minutes
#   playmaking = assists per 36 minutes
OBJECTIVES = ["offense", "defense", "playmaking"]

# --- Bandit experiment --------------------------------------------------------
HORIZON = 100_000               # rounds per run
N_REPEATS = 30                  # runs reported in the paper
EXPERIMENT_SEED = 0             # seed for the truth/pool split and the bandit RNG
FEASIBILITY_SEED = 42           # seed for the truth/pool split in the feasibility check

# Preset weight vectors (offense, defense, playmaking) for the Fixed-Weight methods.
WEIGHTS = {
    "balanced":   np.array([1 / 3, 1 / 3, 1 / 3]),
    "offense":    np.array([0.6, 0.2, 0.2]),
    "defense":    np.array([0.2, 0.6, 0.2]),
    "playmaking": np.array([0.2, 0.2, 0.6]),
}

ETC_EXPLORE_PER_ARM = 10        # pulls per arm in Explore-Then-Commit's exploration phase

# Column names in eoinamoore's PlayerStatistics.csv (Kaggle).
COLS = {
    "first": "firstName",
    "last": "lastName",
    "minutes": "numMinutes",
    "pts": "points",
    "stl": "steals",
    "blk": "blocks",
    "ast": "assists",
    "fga": "fieldGoalsAttempted",
    "fta": "freeThrowsAttempted",
    "gametype": "gameType",
    "gamedate": "gameDate",
}
