# Georgia Tech win probability (CFBD)

This builds your own Georgia Tech win-probability model from CollegeFootballData (CFBD) API data, **without** using CFBD’s win expectancy.

## Setup

1) Set your CFBD API key as an environment variable (don’t commit it):

PowerShell:

`$env:CFBD_API_KEY="YOUR_KEY_HERE"`

2) Build the dataset (2014–2025) and train:

`python .\winprob.py build-dataset --from-year 2014 --to-year 2025`

`python .\winprob.py train`

Training uses **recency weighting** (newer seasons count more) and then **calibrates** probabilities on a validation set (isotonic regression). Model metrics are written to `models/metrics.json`.

3) Predict a matchup (example):

`python .\winprob.py predict --year 2025 --week 1 --opponent "Florida State" --home away`

`predict` prints both `p_win_raw` and the final `p_win` (calibrated if `models/gt_winprob_calibrator_isotonic.joblib` exists).
Calibration may use either isotonic or a sigmoid calibrator (chosen automatically); `predict` prints `calibrator`.

## Predict an entire season schedule

This pulls the schedule from CFBD (`/games`) and writes a CSV of predictions:

`python .\winprob.py predict-season --year 2026 --team "Georgia Tech"`
