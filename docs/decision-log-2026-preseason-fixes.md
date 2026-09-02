# Decision Log: 2026 Preseason Prediction Pipeline Fixes

Working log of the debugging and design decisions made while getting a real Week 1 2026
prediction (Georgia Tech vs. Colorado) out of this project. Kept because the *path* to the
final numbers involved more real engineering judgment than the numbers themselves — several
of the decisions below only happened because an assumption got challenged and turned out to
be wrong.

## Starting point: a plausible-looking number that wasn't trustworthy

The repo already had a full `predictions_2026.csv` from a prior run. Before treating it as an
answer, it got checked against the model/pipeline's actual last-modified dates:

- `predictions_2026.csv`: generated **April 4**
- `models/gt_winprob_logreg.joblib`: retrained **July 13**, after two documented bug fixes
  (a `/talent` and `/player/returning` field-name mismatch, and a year-scoped team
  classification fix)

**Decision: don't trust a prediction file older than the model/pipeline that produced it.**
Confirmed the file was actually stale by inspection, not just by date: the Week 1 Colorado
number was numerically identical to five unrelated opponents (Tennessee, Duke, Boston College,
Louisville, Wake Forest), which only makes sense if per-opponent features were silently
collapsing to the same missing/imputed values rather than reflecting real per-team differences.

## Problem: retraining broke on a `scikit-learn` version mismatch

Attempting to regenerate predictions hit:

```
AttributeError: module 'sklearn.compose._column_transformer' has no attribute '_RemainderColsList'
```

The committed model was pickled with `scikit-learn 1.6.1`; the environment had `1.9.0`.
**Decision: retrain on the currently installed version rather than force-install the old one.**
Tried pinning `scikit-learn==1.6.1` first — it has no prebuilt wheel for Python 3.14 and failed
compiling from source (`ninja: build stopped`). Rather than fighting the toolchain, retrained
from the dataset already on disk (`model_dataset_all_fbs.csv`, verified to already reflect the
July bug fixes by cross-checking its row/null counts against numbers documented in the README),
producing fresh, version-matched model artifacts.

## The real bug: a cache with no expiration

The regenerated Week 1 Colorado prediction came back with `talent_diff`, `returning_diff`,
`recruit_points_diff`, and `recruit_rank_diff` all `None` — silently median-imputed by the
model. The first explanation offered was "CFBD hasn't published 2026 talent/recruiting/returning
data yet," backed by a direct (but cached) API check showing 0 results for all four endpoints.

That explanation got pushed back on: recruiting-class rankings are known well before the season,
so "not published yet" didn't add up. **That pushback is what surfaced the actual bug.**
Re-querying CFBD with the cache bypassed:

| Endpoint | Cached result | Live result |
|---|---|---|
| `/talent` | 0 | 138 teams |
| `/recruiting/teams` | 0 | 221 teams |
| `/player/returning` | 0 | 136 teams |
| `/ratings/elo` (wk 1) | 0 | 16 teams |

The cache files themselves were dated **March 23, 2026** — correct at the time (that data
genuinely didn't exist yet), but `CFBDClient.get()` had no cache expiration, so an empty
response from March was being served as truth in September, silently and permanently.

**Decision: fix the cache before building anything on top of it.** Building carryover
estimators on top of a cache that never refreshes would have just meant those estimators never
actually fired — the stale empty cache would keep winning. Fixed `cfbd_client.py` to not
persist empty-list responses at all (an empty result usually means "not published yet," not
"never will be," and re-fetching it costs one extra call, not correctness). Then purged the 16
specific stale empty cache files for the four affected endpoints — deliberately *not* the ~300
empty `games` cache entries, since those are plausibly legitimate (e.g., a team with no
postseason game) and there was no evidence they were wrong.

## Decision: which carryover estimators were actually worth building

The original ask was "build carryover estimators for talent/recruiting/returning, the same way
Elo already has one." Rather than build all four on assumption, each one got a same-methodology
regression fit first (prior-year value → this-year value, across historical team-season pairs)
to check whether the idea actually held:

| Estimator | n (team-seasons) | R² | Decision |
|---|---|---|---|
| Talent composite | 1,902 | **0.974** | Built |
| Recruiting points | 2,140 | **0.891** | Built |
| Recruiting rank | 2,140 | **0.841** | Built |
| Returning production | 1,417 | **0.026** | **Not built** |

Returning production's R² of 0.026 means prior-year returning production tells you almost
nothing about this year's — which makes sense in hindsight: it's a function of who left the
roster *after* last season, which has no reason to correlate with who left the season before
that. Building a carryover estimator here would have produced a confident-looking number with
no real information behind it — worse than just leaving it missing and letting the model's
existing median-imputation handle it honestly. **The decision to skip it was the evidence-driven
call, not the "harder to build" one.**

## Decision: surface data completeness instead of hiding it

A Week 1 prediction and a Week 10 prediction were producing output in an identical shape, but
running on very different amounts of real information. Added `feature_completeness`,
`carryover_features`, and `missing_features` columns to the prediction output so the difference
is visible rather than silently blended into one number.

## A bug caught while building the above

While testing the new completeness columns against Georgia Tech vs. Mercer (Mercer is FCS),
`talent_diff` was showing up under `carryover_features` — implying an uncertain regression
estimate had been used. In reality, an FCS opponent's missing talent value was correctly falling
back to a defined floor constant (a deliberate convention for known-weak opponents), not an
estimate. Caught this by testing a game specifically chosen to exercise the non-FBS-opponent
code path, not by re-reading the code — and fixed the source-classification logic before
finalizing.

## Measuring whether the fixes actually help, not assuming they do

Whether these fixes make the model *more accurate* — as opposed to just *different* — isn't
answerable from the Week 1 2026 predictions themselves, since none of those games have been
played yet. A probability moving after a data fix is not evidence of improvement by itself.

The actual test: rebuilt the 12-year historical training dataset with the new carryover code
(filling previously-missing `talent_diff`/`recruit_points_diff`/`recruit_rank_diff` cells where
possible — went from 1,674/23/23 missing to 1,628/0/0), then re-ran the walk-forward backtest
against real, known historical outcomes on Georgia Tech's 85 held-out games (2019-2025):

| Metric | Before | After | Change |
|---|---|---|---|
| Raw accuracy | 70.59% | 71.76% | +1.18 pts (1 game) |
| Raw Brier | 0.2175 | 0.2175 | ~0 |
| Raw log loss | 0.6361 | 0.6361 | ~0 |
| Calibrated accuracy | 65.88% | 67.06% | +1.18 pts (1 game) |
| Calibrated Brier | 0.2163 | 0.2163 | ~0 |
| Calibrated ECE | 0.1304 | 0.1190 | slightly better |
| Raw ECE | 0.1570 | 0.1688 | slightly worse |

**Result: real, but small — not a headline improvement.** One game out of 85 flipped from
wrong to right; Brier score and log loss barely moved. Most of the 1,674 missing `talent_diff`
rows in the full 19,456-row dataset are for teams other than Georgia Tech's actual 2019-2025
opponents, and many were missing *multiple consecutive years*, which carryover can't fix (it
needs a prior-year value to carry over from). The honest conclusion isn't "this fix made the
model much better" — it's "this fix is correct and closes real gaps, its measured effect on
this specific team's held-out games is small, and the more meaningful value of the session was
catching and fixing a cache bug that was silently feeding stale/missing data into predictions at
all, which the accuracy number alone wouldn't have revealed."
