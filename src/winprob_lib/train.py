from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

WIN_MODEL_FEATURE_COLS = [
    "is_home",
    "neutral_site",
    "opp_is_fbs",
    "win_pct_diff",
    "point_diff_pg_diff",
    "points_for_pg_diff",
    "points_against_pg_diff",
    "w_win_pct_diff",
    "w_point_diff_pg_diff",
    "w_points_for_pg_diff",
    "w_points_against_pg_diff",
    "opp_elo_avg_diff",
    "w_opp_elo_avg_diff",
    "sos_point_diff_pg_diff",
    "w_sos_point_diff_pg_diff",
    "elo_diff",
    "talent_diff",
    "returning_diff",
    "recruit_points_diff",
    "recruit_rank_diff",
]


def _calibration_bins(p: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        out.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "n": float(n),
                "mean_pred": float(np.mean(p[mask])),
                "empirical_winrate": float(np.mean(y[mask])),
            }
        )
    return out


def _ece_from_bins(bins: list[dict[str, float]], *, total: int) -> float:
    if total <= 0:
        return float("nan")
    ece = 0.0
    for b in bins:
        frac = float(b["n"]) / float(total)
        ece += frac * abs(float(b["empirical_winrate"]) - float(b["mean_pred"]))
    return float(ece)


def _eval_probs(p: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    pred = (p >= 0.5).astype(int)
    bins = _calibration_bins(p, y, n_bins=10)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10bin": _ece_from_bins(bins, total=int(len(y))),
        "bins_10": bins,
    }


def _apply_calibrator(p_raw: float, calibrator: Any | None) -> tuple[float, str]:
    """
    Returns (p_calibrated, calibrator_type).
    - calibrator can be:
        - None
        - legacy isotonic object with .transform
        - dict {"type": "sigmoid_logit"|"isotonic", "model": <sklearn model>}
    """
    if calibrator is None:
        return p_raw, "none"

    if isinstance(calibrator, dict):
        cal_type = calibrator.get("type")
        mdl = calibrator.get("model")
        if cal_type == "sigmoid_logit" and mdl is not None:
            p = float(np.clip(p_raw, 1e-6, 1 - 1e-6))
            logit_raw = float(np.log(p / (1 - p)))
            return float(mdl.predict_proba(np.array([[logit_raw]]))[:, 1][0]), "sigmoid_logit"
        if cal_type == "isotonic" and mdl is not None:
            return float(mdl.transform([p_raw])[0]), "isotonic"
        return p_raw, str(cal_type or "unknown")

    # Legacy isotonic
    if hasattr(calibrator, "transform"):
        return float(calibrator.transform([p_raw])[0]), "isotonic"

    return p_raw, "unknown"


def _fit_win_model(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    half_life_years: float,
    calibration: str,
) -> dict[str, Any]:
    """
    Fits a logistic-regression win-probability pipeline on `train_df` (with recency-weighted
    samples), then fits both an isotonic and a sigmoid calibrator on `val_df` and selects
    between them by validation Brier score. Shared by `train_model()` (single fixed split,
    produces the deployed model artifacts) and `walk_forward_eval()` (many rolling splits,
    produces a pooled evaluation) so both use identical methodology.
    """
    if half_life_years <= 0:
        raise RuntimeError("half_life_years must be > 0")

    X_train = train_df[feature_cols]
    y_train = train_df["gt_win"].astype(int)
    X_val = val_df[feature_cols]
    y_val = val_df["gt_win"].astype(int)
    y_val_np = y_val.to_numpy()

    max_train_year = int(train_df["year"].max())
    year_delta = (max_train_year - train_df["year"]).astype(float)
    sample_weight = (0.5 ** (year_delta / half_life_years)).to_numpy()

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
    )
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train, model__sample_weight=sample_weight)

    p_val_raw = pipe.predict_proba(X_val)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_raw, y_val_np)
    p_val_iso = iso.transform(p_val_raw)

    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    sig = LogisticRegression(max_iter=2000, solver="lbfgs")
    sig.fit(_logit(p_val_raw).reshape(-1, 1), y_val_np)
    p_val_sig = sig.predict_proba(_logit(p_val_raw).reshape(-1, 1))[:, 1]

    brier_raw = float(brier_score_loss(y_val_np, p_val_raw))
    brier_iso = float(brier_score_loss(y_val_np, p_val_iso))
    brier_sig = float(brier_score_loss(y_val_np, p_val_sig))

    calibration = calibration.strip().lower()
    if calibration not in ("auto", "isotonic", "sigmoid", "none"):
        raise RuntimeError("calibration must be one of: auto, isotonic, sigmoid, none")

    if calibration == "none":
        calibrator = None
    elif calibration == "isotonic":
        calibrator = {"type": "isotonic", "model": iso}
    elif calibration == "sigmoid":
        calibrator = {"type": "sigmoid_logit", "model": sig}
    else:
        # auto: choose best Brier; tie-break toward sigmoid if isotonic is too "steppy".
        val_unique_iso = int(np.unique(np.round(p_val_iso, 6)).size)
        if brier_sig <= brier_iso and brier_sig <= brier_raw:
            calibrator = {"type": "sigmoid_logit", "model": sig}
        elif brier_iso <= brier_raw:
            # If isotonic collapses to a small number of outputs, prefer sigmoid when close.
            if val_unique_iso <= 6 and (brier_sig - brier_iso) <= 0.01:
                calibrator = {"type": "sigmoid_logit", "model": sig}
            else:
                calibrator = {"type": "isotonic", "model": iso}
        else:
            calibrator = None

    return {
        "pipe": pipe,
        "iso": iso,
        "sig": sig,
        "calibrator": calibrator,
        "max_train_year": max_train_year,
        "val_brier": {"raw": brier_raw, "isotonic": brier_iso, "sigmoid_logit": brier_sig},
    }


def train_model(
    dataset_csv: Path,
    *,
    model_dir: Path = Path("models"),
    half_life_years: float = 3.0,
    calibration: str = "auto",  # auto|isotonic|sigmoid|none
    target: str = "win",  # win|margin
) -> dict[str, Any]:
    df = pd.read_csv(dataset_csv)
    if df.empty:
        raise RuntimeError(f"No rows in dataset: {dataset_csv}")

    requested_feature_cols = WIN_MODEL_FEATURE_COLS
    target = target.strip().lower()
    if target not in ("win", "margin"):
        raise RuntimeError("--target must be win or margin")
    target_col = "gt_win" if target == "win" else "margin"

    df = df.dropna(subset=[target_col])
    df["year"] = df["year"].astype(int)

    # Drop columns that are entirely missing (common when an endpoint is unavailable on a user's plan).
    feature_cols = [c for c in requested_feature_cols if c in df.columns and df[c].notna().any()]
    if not feature_cols:
        raise RuntimeError(
            "No usable feature columns (all missing). "
            "Check that `elo_diff` and/or other features are populated in the dataset."
        )

    train_df = df[(df["year"] >= 2014) & (df["year"] <= 2022)]
    val_df = df[(df["year"] >= 2023) & (df["year"] <= 2024)]
    test_df = df[df["year"] == 2025]

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(
            f"Need non-empty splits. Got train={len(train_df)}, val={len(val_df)}, test={len(test_df)} rows."
        )

    if half_life_years <= 0:
        raise RuntimeError("--half-life-years must be > 0")

    if target == "margin":
        X_train = train_df[feature_cols]
        y_train = train_df[target_col].astype(float)
        X_val = val_df[feature_cols]
        y_val = val_df[target_col].astype(float)
        X_test = test_df[feature_cols]
        y_test = test_df[target_col].astype(float)

        max_train_year = int(train_df["year"].max())
        year_delta = (max_train_year - train_df["year"]).astype(float)
        sample_weight = (0.5 ** (year_delta / half_life_years)).to_numpy()

        pre = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    feature_cols,
                )
            ],
            remainder="drop",
        )
        pipe = Pipeline(steps=[("pre", pre), ("model", Ridge(alpha=1.0))])
        pipe.fit(X_train, y_train, model__sample_weight=sample_weight)

        p_val_raw = pipe.predict(X_val)
        p_test_raw = pipe.predict(X_test)
        y_val_np = y_val.to_numpy()
        y_test_np = y_test.to_numpy()

        resid_val = (y_val_np - p_val_raw).astype(float)
        sigma = float(np.std(resid_val)) or 1.0

        def _norm_cdf(z: np.ndarray) -> np.ndarray:
            return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

        p_val_prob = _norm_cdf(p_val_raw / sigma)
        p_test_prob = _norm_cdf(p_test_raw / sigma)

        metrics = {
            "train_config": {"half_life_years": half_life_years, "max_train_year": max_train_year, "target": target},
            "margin_sigma": sigma,
            "val": {"prob": _eval_probs(p_val_prob, (y_val_np > 0).astype(int))},
            "test": {"prob": _eval_probs(p_test_prob, (y_test_np > 0).astype(int))},
        }

        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_dir / "gt_margin_ridge.joblib")
        (model_dir / "margin_sigma.json").write_text(json.dumps({"sigma": sigma}, indent=2), encoding="utf-8")
        (model_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
        (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (model_dir / "model_meta.json").write_text(
            json.dumps({"target": "margin", "model_file": "gt_margin_ridge.joblib"}, indent=2),
            encoding="utf-8",
        )
        return {"feature_cols": feature_cols, "metrics": metrics}

    fit = _fit_win_model(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        half_life_years=half_life_years,
        calibration=calibration,
    )
    pipe = fit["pipe"]
    iso = fit["iso"]
    calibrator = fit["calibrator"]
    max_train_year = fit["max_train_year"]
    brier_raw = fit["val_brier"]["raw"]
    brier_iso = fit["val_brier"]["isotonic"]
    brier_sig = fit["val_brier"]["sigmoid_logit"]

    y_val_np = val_df["gt_win"].astype(int).to_numpy()
    y_test_np = test_df["gt_win"].astype(int).to_numpy()
    p_val_raw = pipe.predict_proba(val_df[feature_cols])[:, 1]
    p_test_raw = pipe.predict_proba(test_df[feature_cols])[:, 1]
    p_val_cal = np.array([_apply_calibrator(p, calibrator)[0] for p in p_val_raw])
    p_test_cal = np.array([_apply_calibrator(p, calibrator)[0] for p in p_test_raw])

    metrics = {
        "train_config": {
            "half_life_years": half_life_years,
            "max_train_year": max_train_year,
            "calibration": calibration,
            "target": target,
        },
        "calibration_choice": {
            "selected": (calibrator or {}).get("type") if isinstance(calibrator, dict) else "none",
            "val_brier": {"raw": brier_raw, "isotonic": brier_iso, "sigmoid_logit": brier_sig},
        },
        "val": {"raw": _eval_probs(p_val_raw, y_val_np), "calibrated": _eval_probs(p_val_cal, y_val_np)},
        "test": {"raw": _eval_probs(p_test_raw, y_test_np), "calibrated": _eval_probs(p_test_cal, y_test_np)},
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_dir / "gt_winprob_logreg.joblib")
    # Optional backwards-compatible isotonic file.
    # Note: it may not be the chosen calibrator.
    try:
        joblib.dump(iso, model_dir / "gt_winprob_calibrator_isotonic.joblib")
    except Exception:
        pass
    if calibrator is not None:
        joblib.dump(calibrator, model_dir / "gt_winprob_calibrator.joblib")
    (model_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (model_dir / "model_meta.json").write_text(
        json.dumps({"target": "win", "model_file": "gt_winprob_logreg.joblib"}, indent=2),
        encoding="utf-8",
    )

    return {"feature_cols": feature_cols, "metrics": metrics}


def walk_forward_eval(
    dataset_csv: Path,
    *,
    start_test_year: int,
    end_test_year: int | None = None,
    half_life_years: float = 3.0,
    calibration: str = "auto",
    eval_team: str | None = None,
    out_dir: Path = Path("reports") / "walkforward",
) -> dict[str, Any]:
    """
    Rolling walk-forward backtest: for each test year Y, trains on years <= Y-2, calibrates
    on year Y-1, and predicts year Y - then pools every year's held-out predictions into one
    set of metrics. A single fixed test year (e.g. 2025 alone) gives a noisy handful of
    games; this aggregates many years of held-out games into one trustworthy estimate,
    using the exact same fit/calibration methodology as `train_model()` (`_fit_win_model`)
    for every fold. Win-probability target only.

    `eval_team`, if given, restricts only the held-out test rows to that team's games (train
    and validation still use every row) - the metric for "does a model trained on pooled
    data actually predict this one team better", as opposed to overall FBS-wide accuracy.
    """
    df = pd.read_csv(dataset_csv)
    if df.empty:
        raise RuntimeError(f"No rows in dataset: {dataset_csv}")

    df = df.dropna(subset=["gt_win"])
    df["year"] = df["year"].astype(int)

    feature_cols = [c for c in WIN_MODEL_FEATURE_COLS if c in df.columns and df[c].notna().any()]
    if not feature_cols:
        raise RuntimeError("No usable feature columns (all missing).")

    if eval_team is not None and "team" not in df.columns:
        raise RuntimeError("--eval-team requires a 'team' column (only present in all-fbs pooled datasets).")

    max_year = int(df["year"].max())
    end_test_year = end_test_year if end_test_year is not None else max_year

    fold_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []

    for test_year in range(start_test_year, end_test_year + 1):
        train_df = df[df["year"] <= test_year - 2]
        val_df = df[df["year"] == test_year - 1]
        test_df = df[df["year"] == test_year]
        if eval_team is not None:
            test_df = test_df[test_df["team"] == eval_team]
        if train_df.empty or val_df.empty or test_df.empty:
            continue

        fit = _fit_win_model(
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            half_life_years=half_life_years,
            calibration=calibration,
        )
        pipe = fit["pipe"]
        calibrator = fit["calibrator"]

        y_test = test_df["gt_win"].astype(int).to_numpy()
        p_test_raw = pipe.predict_proba(test_df[feature_cols])[:, 1]
        p_test_cal = np.array([_apply_calibrator(p, calibrator)[0] for p in p_test_raw])

        for i, (_, row) in enumerate(test_df.iterrows()):
            fold_rows.append(
                {
                    "test_year": test_year,
                    "game_id": row.get("game_id"),
                    "week": row.get("week"),
                    "team": row.get("team"),
                    "opponent": row.get("opponent"),
                    "y_true": int(y_test[i]),
                    "p_raw": float(p_test_raw[i]),
                    "p_calibrated": float(p_test_cal[i]),
                }
            )

        fold_summaries.append(
            {
                "test_year": test_year,
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
                "calibrator": (calibrator or {}).get("type") if isinstance(calibrator, dict) else "none",
                "val_brier": fit["val_brier"],
            }
        )

    if not fold_rows:
        raise RuntimeError(
            "No walk-forward folds produced. Each test year needs at least one earlier "
            "train year and a non-empty val year (test_year - 1) with data - check "
            "--start-test-year against the dataset's year range."
        )

    pred_df = pd.DataFrame(fold_rows)
    y_all = pred_df["y_true"].to_numpy()
    p_raw_all = pred_df["p_raw"].to_numpy()
    p_cal_all = pred_df["p_calibrated"].to_numpy()

    overall = {
        "dataset": str(dataset_csv),
        "n_games": int(len(pred_df)),
        "n_folds": int(len(fold_summaries)),
        "start_test_year": start_test_year,
        "end_test_year": end_test_year,
        "raw": _eval_probs(p_raw_all, y_all),
        "calibrated": _eval_probs(p_cal_all, y_all),
        "folds": fold_summaries,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "walkforward_predictions.csv", index=False)
    (out_dir / "walkforward_metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    return overall
