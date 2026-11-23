# Data

The experiments use `PlayerStatistics.csv` from the Kaggle dataset
**"NBA Dataset: Box Scores and Stats (1947 – Today)"** by Eoin A. Moore (CC0 licence):

https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores

1. Download the dataset from Kaggle (a free account is required).
2. Put `PlayerStatistics.csv` in this folder: `data/PlayerStatistics.csv`.

The file has one row per player per game (1,669,922 rows in the version used).
The code reads only the columns it needs and keeps the 2024-25 regular season,
which leaves 26,183 games from the 286 qualifying players.
The data is not included in this repository because of its size.

**Version.** The dataset is updated daily. The paper uses the copy downloaded on
22 August 2026. A newer download may include stat corrections for 2024-25 and
shift the results slightly.

Columns used: `firstName`, `lastName`, `numMinutes`, `points`, `steals`, `blocks`,
`assists`, `fieldGoalsAttempted`, `freeThrowsAttempted`, `gameType`, `gameDate`.
If your copy uses different column names, edit `COLS` in `mobandit/config.py`.
