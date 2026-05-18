from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.special import logit
from scipy.stats import bootstrap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix


def _safe_log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(log_loss(y_true, p, labels=[0, 1]))


def _ece_10bin(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        frac = float(np.mean(mask))
        ece += frac * abs(float(np.mean(y_true[mask])) - float(np.mean(p[mask])))
    return float(ece)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1 - phat) / n) + ((z**2) / (4 * n**2)))
    return float(center - half), float(center + half)


@dataclass(frozen=True)
class BacktestInputs:
    year: int
    predictions_csv: Path
    dataset_csv: Path
    out_dir: Path


def load_backtest(inputs: BacktestInputs) -> pd.DataFrame:
    pred = pd.read_csv(inputs.predictions_csv)
    ds = pd.read_csv(inputs.dataset_csv)

    ds_y = ds[ds["year"] == inputs.year][["week", "opponent", "gt_win"]].copy()
    merged = pred.merge(ds_y, on=["week", "opponent"], how="left")
    merged["gt_win"] = merged["gt_win"].astype("float")
    if "p_win" not in merged.columns:
        raise RuntimeError(f"Missing p_win column in {inputs.predictions_csv}")
    merged["p_win"] = merged["p_win"].astype("float")
    merged["p_win_raw"] = merged["p_win_raw"].astype("float") if "p_win_raw" in merged.columns else np.nan
    return merged.sort_values(["week", "start_date", "opponent"], kind="stable")


def _bootstrap_ci(data: np.ndarray, fn, confidence_level: float = 0.95, n_resamples: int = 10_000) -> tuple[float, float]:
    # SciPy bootstrap wants a tuple of arrays.
    res = bootstrap(
        (data,),
        statistic=lambda x: fn(x),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method="percentile",
        random_state=0,
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    df = df.dropna(subset=["gt_win", "p_win"]).copy()
    y = df["gt_win"].astype(int).to_numpy()
    p = np.clip(df["p_win"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)

    n = int(len(df))
    actual_wins = int(y.sum())
    expected_wins = float(p.sum())

    # Win rate CI
    win_ci = _wilson_ci(actual_wins, n)

    # Per-game losses for hypothesis tests / CIs
    per_game_brier = (p - y) ** 2
    per_game_logloss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    baseline_p = np.full_like(p, 0.5, dtype=float)
    baseline_logloss = -(y * np.log(baseline_p) + (1 - y) * np.log(1 - baseline_p))
    per_game_logloss_delta = per_game_logloss - baseline_logloss

    brier_ci = _bootstrap_ci(per_game_brier, np.mean)
    logloss_ci = _bootstrap_ci(per_game_logloss, np.mean)
    logloss_delta_ci = _bootstrap_ci(per_game_logloss_delta, np.mean)
    acc_ci = _bootstrap_ci((pred == y).astype(float), np.mean)

    # Calibration slope/intercept: fit y ~ logit(p)
    x = logit(p).reshape(-1, 1)
    cal = LogisticRegression(solver="lbfgs")
    cal.fit(x, y)
    slope = float(cal.coef_[0, 0])
    intercept = float(cal.intercept_[0])

    ece = _ece_10bin(y, p, n_bins=10)

    return {
        "n_games": n,
        "actual_wins": actual_wins,
        "expected_wins": expected_wins,
        "actual_win_rate": float(actual_wins / n) if n else float("nan"),
        "actual_win_rate_wilson_95": {"low": win_ci[0], "high": win_ci[1]},
        "accuracy": float(accuracy_score(y, pred)),
        "accuracy_bootstrap_95": {"low": acc_ci[0], "high": acc_ci[1]},
        "brier": float(brier_score_loss(y, p)),
        "brier_bootstrap_95": {"low": brier_ci[0], "high": brier_ci[1]},
        "log_loss": _safe_log_loss(y, p),
        "log_loss_bootstrap_95": {"low": logloss_ci[0], "high": logloss_ci[1]},
        "log_loss_vs_0p5_delta": float(np.mean(per_game_logloss_delta)),
        "log_loss_vs_0p5_delta_bootstrap_95": {"low": logloss_delta_ci[0], "high": logloss_delta_ci[1]},
        "ece_10bin": ece,
        "calibration_logit_fit": {"intercept": intercept, "slope": slope},
    }


def plot_reliability(df: pd.DataFrame, out_path: Path) -> None:
    df = df.dropna(subset=["gt_win", "p_win"]).copy()
    df["bin"] = pd.cut(df["p_win"], bins=np.linspace(0, 1, 11), include_lowest=True)
    grp = df.groupby("bin", observed=True).agg(n=("gt_win", "size"), mean_p=("p_win", "mean"), win_rate=("gt_win", "mean")).reset_index()

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.scatter(grp["mean_p"], grp["win_rate"], s=np.clip(grp["n"] * 40, 40, 300))
    for _, r in grp.iterrows():
        plt.annotate(str(int(r["n"])), (r["mean_p"], r["win_rate"]), textcoords="offset points", xytext=(6, 6), fontsize=8)
    plt.xlabel("Mean predicted win probability")
    plt.ylabel("Empirical win rate")
    plt.title("Reliability (10 bins; label = n)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_histogram(df: pd.DataFrame, out_path: Path) -> None:
    df = df.dropna(subset=["p_win"]).copy()
    plt.figure(figsize=(7, 4))
    sns.histplot(df["p_win"], bins=12, kde=True)
    plt.xlabel("Predicted win probability")
    plt.title("Distribution of predicted probabilities")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_cumulative_wins(df: pd.DataFrame, out_path: Path) -> None:
    df = df.sort_values(["week", "start_date", "opponent"], kind="stable").copy()
    df["p_win"] = df["p_win"].astype(float)
    df["gt_win"] = df["gt_win"].astype(float)
    df["exp_wins_cum"] = df["p_win"].cumsum()
    df["actual_wins_cum"] = df["gt_win"].cumsum()

    plt.figure(figsize=(7, 4))
    plt.plot(df["week"], df["exp_wins_cum"], marker="o", label="Expected wins (cum)")
    plt.plot(df["week"], df["actual_wins_cum"], marker="o", label="Actual wins (cum)")
    plt.xlabel("Week")
    plt.ylabel("Wins")
    plt.title("Cumulative expected vs actual wins")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_confusion_matrix(df: pd.DataFrame, out_path: Path, *, threshold: float = 0.5) -> None:
    """
    Confusion matrix for the winner-pick decision rule:
      predict win if p_win >= threshold else loss.

    Rows = actual, cols = predicted, labels = [Loss(0), Win(1)].
    """
    df = df.dropna(subset=["gt_win", "p_win"]).copy()
    y = df["gt_win"].astype(int).to_numpy()
    p = df["p_win"].astype(float).to_numpy()
    pred = (p >= float(threshold)).astype(int)

    cm = confusion_matrix(y, pred, labels=[0, 1])
    # cm = [[TN, FP],
    #       [FN, TP]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    plt.figure(figsize=(5.2, 4.4))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=1,
        linecolor="white",
        square=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Loss", "Win"])
    ax.set_yticklabels(["Loss", "Win"], rotation=0)
    plt.title(f"Confusion Matrix (threshold={threshold:.2f})\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def plot_model_vs_market(csv_path: Path, out_dir: Path, *, year: int | None = None) -> None:
    """
    Build simple visuals comparing your model probabilities to market-implied probabilities.

    Expects columns like:
      - week
      - opponent
      - your_p_win_pct
      - actual_pregame_p_win_pct
    (Matches `game_by_game_comparison_table.csv` and the embedded dashboard table.)
    """
    df = pd.read_csv(csv_path)
    need = {"week", "opponent", "your_p_win_pct", "actual_pregame_p_win_pct"}
    missing = sorted(need - set(df.columns))
    if missing:
        raise RuntimeError(f"{csv_path} missing columns: {missing}")

    df = df.copy()
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["your_p_win_pct"] = pd.to_numeric(df["your_p_win_pct"], errors="coerce")
    df["actual_pregame_p_win_pct"] = pd.to_numeric(df["actual_pregame_p_win_pct"], errors="coerce")
    df = df.dropna(subset=["week", "your_p_win_pct", "actual_pregame_p_win_pct"])
    if df.empty:
        raise RuntimeError("No usable rows after parsing probabilities.")

    df["delta_pp"] = df["your_p_win_pct"] - df["actual_pregame_p_win_pct"]
    # Category label for color: optimistic/pessimistic/close to market.
    df["stance"] = np.where(df["delta_pp"] >= 3.0, "More optimistic than market", np.where(df["delta_pp"] <= -3.0, "More pessimistic than market", "Close to market"))

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{year}" if year is not None else ""

    # Scatter: market vs model (perfect agreement = diagonal)
    plt.figure(figsize=(7.2, 6.0))
    palette = {
        "More optimistic than market": "#38bdf8",
        "More pessimistic than market": "#fb923c",
        "Close to market": "#e5e7eb",
    }
    sns.scatterplot(
        data=df,
        x="actual_pregame_p_win_pct",
        y="your_p_win_pct",
        hue="stance",
        palette=palette,
        s=90,
        edgecolor="black",
        linewidth=0.6,
    )
    plt.plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1)
    for _, r in df.iterrows():
        label = f"W{int(r['week'])}"
        opp = str(r.get("opponent") or "").strip()
        if opp:
            label = f"{label} {opp}"
        plt.annotate(label, (r["actual_pregame_p_win_pct"], r["your_p_win_pct"]), textcoords="offset points", xytext=(6, 6), fontsize=8)
    plt.xlabel("Market implied p-win (%)")
    plt.ylabel("Your model p-win (%)")
    title_year = f" ({year})" if year is not None else ""
    plt.title(f"Model vs market implied win probability{title_year}")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend(title="", loc="lower right", framealpha=0.15)
    plt.tight_layout()
    plt.savefig(out_dir / f"model_vs_market_scatter{suffix}.png", dpi=170)
    plt.close()

    # Bar: delta (model - market) by week
    df2 = df.sort_values(["week", "opponent"], kind="stable").copy()
    df2["xlab"] = df2.apply(lambda r: f"W{int(r['week'])}", axis=1)
    plt.figure(figsize=(8.6, 4.2))
    colors = df2["delta_pp"].apply(lambda v: "#38bdf8" if v >= 3.0 else ("#fb923c" if v <= -3.0 else "#e5e7eb")).tolist()
    plt.bar(np.arange(len(df2)), df2["delta_pp"], color=colors, edgecolor=(0, 0, 0, 0.35), linewidth=0.8)
    plt.axhline(0, color="gray", linewidth=1)
    plt.xticks(np.arange(len(df2)), df2["xlab"], rotation=0)
    plt.ylabel("Δ (model − market) in percentage points")
    plt.title(f"Where the model disagrees with the market{title_year}")
    plt.tight_layout()
    plt.savefig(out_dir / f"model_minus_market_delta{suffix}.png", dpi=170)
    plt.close()

    # Dumbbell: per-game market vs model (easy to read)
    df3 = df.sort_values(["week", "opponent"], kind="stable").copy()
    df3["label"] = df3.apply(lambda r: f"W{int(r['week'])} {str(r.get('opponent') or '').strip()}", axis=1)
    plt.figure(figsize=(9.2, max(4.0, 0.45 * len(df3))))
    y = np.arange(len(df3))
    x_mkt = df3["actual_pregame_p_win_pct"].to_numpy(dtype=float)
    x_mod = df3["your_p_win_pct"].to_numpy(dtype=float)
    for i in range(len(df3)):
        plt.plot([x_mkt[i], x_mod[i]], [y[i], y[i]], color=(1, 1, 1, 0.35), linewidth=2)
    plt.scatter(x_mkt, y, label="Market", color="#fbbf24", s=70, edgecolor="black", linewidth=0.5, zorder=3)
    plt.scatter(x_mod, y, label="Model", color="#38bdf8", s=70, edgecolor="black", linewidth=0.5, zorder=3)
    plt.yticks(y, df3["label"])
    plt.xlabel("Win probability (%)")
    plt.title(f"Per-game win probability: model vs market{title_year}")
    plt.xlim(0, 100)
    plt.grid(axis="x", color=(1, 1, 1, 0.06))
    plt.legend(loc="lower right", framealpha=0.15)
    plt.tight_layout()
    plt.savefig(out_dir / f"model_vs_market_dumbbell{suffix}.png", dpi=170)
    plt.close()


def plot_market_vs_fpi(
    *,
    market_csv: Path,
    fpi_csv: Path,
    out_dir: Path,
    year: int | None = None,
) -> None:
    """
    Compare two external pregame baselines: betting market vs ESPN FPI (GT win%).

    market_csv columns:
      - week
      - opponent
      - market_p_win_pct

    fpi_csv columns:
      - week
      - opponent
      - fpi_p_win_pct
    """
    m = pd.read_csv(market_csv)
    f = pd.read_csv(fpi_csv)
    for df, name, cols in [
        (m, "market_csv", {"week", "opponent", "market_p_win_pct"}),
        (f, "fpi_csv", {"week", "opponent", "fpi_p_win_pct"}),
    ]:
        missing = sorted(cols - set(df.columns))
        if missing:
            raise RuntimeError(f"{name} missing columns: {missing}")

    m = m.copy()
    f = f.copy()
    m["week"] = pd.to_numeric(m["week"], errors="coerce")
    f["week"] = pd.to_numeric(f["week"], errors="coerce")
    m["market_p_win_pct"] = pd.to_numeric(m["market_p_win_pct"], errors="coerce")
    f["fpi_p_win_pct"] = pd.to_numeric(f["fpi_p_win_pct"], errors="coerce")

    merged = m.merge(f[["week", "fpi_p_win_pct"]], on="week", how="inner")
    merged = merged.dropna(subset=["week", "market_p_win_pct", "fpi_p_win_pct"]).sort_values("week", kind="stable")
    if merged.empty:
        raise RuntimeError("No rows after merging market and FPI by week.")

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{year}" if year is not None else ""
    title_year = f" ({year})" if year is not None else ""

    # Grouped bars (Market vs FPI)
    plot_df = pd.DataFrame(
        {
            "week": merged["week"].astype(int),
            "Market": merged["market_p_win_pct"].astype(float),
            "ESPN FPI": merged["fpi_p_win_pct"].astype(float),
        }
    )
    long = plot_df.melt(id_vars=["week"], var_name="source", value_name="p_win_pct")
    long["week_label"] = long["week"].apply(lambda w: f"W{int(w)}")

    plt.figure(figsize=(10.5, 4.6))
    sns.barplot(
        data=long,
        x="week_label",
        y="p_win_pct",
        hue="source",
        palette={"Market": "#fbbf24", "ESPN FPI": "#a78bfa"},
        edgecolor="black",
        linewidth=0.4,
    )
    plt.axhline(50, color="gray", linestyle="--", linewidth=1)
    plt.ylim(0, 100)
    plt.ylabel("GT win probability (%)")
    plt.xlabel("Week")
    plt.title(f"Market vs ESPN FPI win probabilities{title_year}")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(out_dir / f"market_vs_fpi_bar{suffix}.png", dpi=170)
    plt.close()


def plot_model_vs_fpi(
    *,
    backtest_csv: Path,
    fpi_csv: Path,
    out_dir: Path,
    year: int | None = None,
) -> None:
    """
    Compare your model win probabilities vs ESPN FPI probabilities for each game.

    backtest_csv should be like reports/<year>/backtest_<year>.csv, with columns:
      - week, opponent, p_win
    fpi_csv columns:
      - week, opponent, fpi_p_win_pct
    """
    bt = pd.read_csv(backtest_csv)
    fpi = pd.read_csv(fpi_csv)
    need_bt = {"week", "opponent", "p_win"}
    need_fpi = {"week", "opponent", "fpi_p_win_pct"}
    miss_bt = sorted(need_bt - set(bt.columns))
    miss_fpi = sorted(need_fpi - set(fpi.columns))
    if miss_bt:
        raise RuntimeError(f"{backtest_csv} missing columns: {miss_bt}")
    if miss_fpi:
        raise RuntimeError(f"{fpi_csv} missing columns: {miss_fpi}")

    bt = bt.copy()
    fpi = fpi.copy()
    bt["week"] = pd.to_numeric(bt["week"], errors="coerce")
    fpi["week"] = pd.to_numeric(fpi["week"], errors="coerce")
    bt["p_win"] = pd.to_numeric(bt["p_win"], errors="coerce")
    fpi["fpi_p_win_pct"] = pd.to_numeric(fpi["fpi_p_win_pct"], errors="coerce")

    merged = bt.merge(fpi[["week", "fpi_p_win_pct"]], on="week", how="left")
    merged = merged.dropna(subset=["week", "p_win", "fpi_p_win_pct"]).sort_values("week", kind="stable")
    if merged.empty:
        raise RuntimeError("No rows after merging model backtest with FPI.")

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{year}" if year is not None else ""
    title_year = f" ({year})" if year is not None else ""

    plot_df = pd.DataFrame(
        {
            "week": merged["week"].astype(int),
            "Model": (merged["p_win"].astype(float) * 100.0),
            "ESPN FPI": merged["fpi_p_win_pct"].astype(float),
        }
    )
    long = plot_df.melt(id_vars=["week"], var_name="source", value_name="p_win_pct")
    long["week_label"] = long["week"].apply(lambda w: f"W{int(w)}")

    plt.figure(figsize=(10.5, 4.6))
    sns.barplot(
        data=long,
        x="week_label",
        y="p_win_pct",
        hue="source",
        palette={"Model": "#38bdf8", "ESPN FPI": "#a78bfa"},
        edgecolor="black",
        linewidth=0.4,
    )
    plt.axhline(50, color="gray", linestyle="--", linewidth=1)
    plt.ylim(0, 100)
    plt.ylabel("GT win probability (%)")
    plt.xlabel("Week")
    plt.title(f"Model vs ESPN FPI win probabilities{title_year}")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(out_dir / f"model_vs_fpi_bar{suffix}.png", dpi=170)
    plt.close()


def write_report(inputs: BacktestInputs) -> None:
    df = load_backtest(inputs)
    inputs.out_dir.mkdir(parents=True, exist_ok=True)

    merged_path = inputs.out_dir / f"backtest_{inputs.year}.csv"
    df.to_csv(merged_path, index=False)

    metrics = compute_metrics(df)
    (inputs.out_dir / f"metrics_{inputs.year}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_reliability(df, inputs.out_dir / f"reliability_{inputs.year}.png")
    plot_histogram(df, inputs.out_dir / f"hist_{inputs.year}.png")
    plot_cumulative_wins(df, inputs.out_dir / f"cumulative_wins_{inputs.year}.png")
    plot_confusion_matrix(df, inputs.out_dir / f"confusion_{inputs.year}.png")

    print(f"Wrote: {merged_path}")
    print(f"Wrote: {inputs.out_dir / f'metrics_{inputs.year}.json'}")
    print(f"Wrote: {inputs.out_dir / f'reliability_{inputs.year}.png'}")
    print(f"Wrote: {inputs.out_dir / f'hist_{inputs.year}.png'}")
    print(f"Wrote: {inputs.out_dir / f'cumulative_wins_{inputs.year}.png'}")
    print(f"Wrote: {inputs.out_dir / f'confusion_{inputs.year}.png'}")


def _bootstrap_mean_ci(x: np.ndarray) -> tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    return _bootstrap_ci(x.astype(float), np.mean)


def _bootstrap_diff_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    data = np.concatenate([a, b]).astype(float)
    n_a = len(a)

    def stat(x: np.ndarray) -> float:
        return float(np.mean(x[:n_a]) - np.mean(x[n_a:]))

    res = bootstrap(
        (data,),
        statistic=lambda x: stat(x),
        confidence_level=0.95,
        n_resamples=10_000,
        method="percentile",
        random_state=0,
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def gt_summary(
    *,
    dataset_csv: Path,
    year_from: int,
    year_to: int,
    out_dir: Path,
    include_postseason: bool = False,
) -> None:
    df = pd.read_csv(dataset_csv)
    df["year"] = df["year"].astype(int)

    if not include_postseason and "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.lower() == "regular"]

    df = df[(df["year"] >= year_from) & (df["year"] <= year_to)].copy()
    if df.empty:
        raise RuntimeError("No rows after filtering year range.")

    df["gt_win"] = df["gt_win"].astype(int)
    n = int(len(df))
    wins = int(df["gt_win"].sum())
    losses = int(n - wins)
    winrate = float(wins / n)
    win_ci = _wilson_ci(wins, n)

    if "margin" in df.columns:
        margin = df["margin"].astype(float).to_numpy()
    else:
        margin = np.full(n, np.nan)
    margin_mean = float(np.nanmean(margin))
    margin_ci = _bootstrap_mean_ci(margin[~np.isnan(margin)])

    # ===== Home-field effect (regression) =====
    # Estimate home advantage controlling for strength (elo_diff) using:
    # - linear regression on margin
    # - logistic regression on win probability
    # Use only non-neutral games (neutral-site games mix "home team" labeling).
    home_effect: dict[str, Any] = {"note": "Uses non-neutral games only; controls for elo_diff."}
    if {"is_home", "neutral_site", "elo_diff"}.issubset(set(df.columns)) and df["elo_diff"].notna().any():
        hw = df[(df["neutral_site"].astype(int) == 0) & (df["is_home"].isin([0, 1]))].copy()
        hw = hw.dropna(subset=["margin", "gt_win", "elo_diff"])
        n_hw = int(len(hw))
        home_effect["n_games_non_neutral"] = n_hw
        if n_hw >= 20:
            X = hw[["is_home", "elo_diff"]].astype(float).to_numpy()
            y_margin = hw["margin"].astype(float).to_numpy()
            y_win = hw["gt_win"].astype(int).to_numpy()

            # Fit point-estimate models
            lm = LinearRegression()
            lm.fit(X, y_margin)
            home_pts = float(lm.coef_[0])  # points added when is_home=1 vs 0

            logm = LogisticRegression(solver="lbfgs", max_iter=2000)
            logm.fit(X, y_win)
            home_logodds = float(logm.coef_[0, 0])
            home_odds_ratio = float(np.exp(home_logodds))

            # Win-prob shift at elo_diff=0 (a "even teams" reference point)
            def _p(is_home: int) -> float:
                return float(logm.predict_proba(np.array([[float(is_home), 0.0]]))[:, 1][0])

            p_away_even = _p(0)
            p_home_even = _p(1)
            p_shift_even = p_home_even - p_away_even

            # Bootstrap CIs by resampling games
            rng = np.random.default_rng(0)
            B = 3000
            boot_home_pts: list[float] = []
            boot_home_logodds: list[float] = []
            for _ in range(B):
                idx = rng.integers(0, n_hw, size=n_hw)
                Xb = X[idx]
                ymb = y_margin[idx]
                ywb = y_win[idx]
                try:
                    lmb = LinearRegression().fit(Xb, ymb)
                    boot_home_pts.append(float(lmb.coef_[0]))
                except Exception:
                    continue
                try:
                    # Guard against degenerate resamples (all wins/losses)
                    if len(np.unique(ywb)) < 2:
                        continue
                    logmb = LogisticRegression(solver="lbfgs", max_iter=2000).fit(Xb, ywb)
                    boot_home_logodds.append(float(logmb.coef_[0, 0]))
                except Exception:
                    continue

            def _pct_ci(vals: list[float]) -> dict[str, float]:
                if not vals:
                    return {"low": float("nan"), "high": float("nan")}
                a = np.percentile(vals, [2.5, 97.5])
                return {"low": float(a[0]), "high": float(a[1])}

            home_effect.update(
                {
                    "margin_model": {
                        "home_adv_points": home_pts,
                        "home_adv_points_bootstrap_95": _pct_ci(boot_home_pts),
                        "features": ["is_home", "elo_diff"],
                    },
                    "win_model": {
                        "home_log_odds": home_logodds,
                        "home_log_odds_bootstrap_95": _pct_ci(boot_home_logodds),
                        "home_odds_ratio": home_odds_ratio,
                        "home_odds_ratio_bootstrap_95": _pct_ci([float(np.exp(v)) for v in boot_home_logodds]),
                        "p_win_even_away": p_away_even,
                        "p_win_even_home": p_home_even,
                        "p_win_even_shift": p_shift_even,
                        "features": ["is_home", "elo_diff"],
                    },
                }
            )
        else:
            home_effect["warning"] = "Too few non-neutral games with elo_diff to estimate."
    else:
        home_effect["warning"] = "Missing columns for home-field regression (need is_home, neutral_site, elo_diff)."

    # ===== Derived stats for dashboard enhancements =====
    # 1) Close games (1-score): |margin| <= 8
    close_mask = np.isfinite(margin) & (np.abs(margin) <= 8)
    close_y = df.loc[close_mask, "gt_win"].astype(int).to_numpy()
    close_n = int(len(close_y))
    close_wins = int(close_y.sum()) if close_n else 0
    close_rate = float(close_wins / close_n) if close_n else float("nan")
    close_ci = _wilson_ci(close_wins, close_n) if close_n else (float("nan"), float("nan"))

    # 2) Blowouts / comfortable outcomes: margin >= 14 (win) and margin <= -14 (loss)
    blowout_win_mask = np.isfinite(margin) & (margin >= 14)
    blowout_loss_mask = np.isfinite(margin) & (margin <= -14)
    blowout_win_rate = float(np.mean(blowout_win_mask)) if n else float("nan")
    blowout_loss_rate = float(np.mean(blowout_loss_mask)) if n else float("nan")

    # 3) Pythagorean expectation (by year)
    # Use common football exponent ~2.37.
    pyth_exp = 2.37

    # 4) Opponent strength proxy (by year)
    # Prefer opponent pregame elo if present; otherwise fall back to "avg opponent elo faced so far" (gt_opp_elo_avg).
    opp_elo_col = None
    if "opp_pregame_elo" in df.columns:
        opp_elo_col = "opp_pregame_elo"
    elif "gt_opp_elo_avg" in df.columns:
        opp_elo_col = "gt_opp_elo_avg"

    # 5) Elo gap vs outcome (binning)
    elo_diff = df["elo_diff"].astype(float).to_numpy() if "elo_diff" in df.columns else np.full(n, np.nan)

    # Home/away splits (neutral counted separately)
    is_home = df.get("is_home", pd.Series([np.nan] * n)).astype(float).to_numpy()
    neutral = df.get("neutral_site", pd.Series([0] * n)).astype(int).to_numpy()

    mask_home = (is_home == 1) & (neutral == 0)
    mask_away = (is_home == 0) & (neutral == 0)
    mask_neutral = neutral == 1

    def _rate_ci(mask: np.ndarray) -> dict[str, float]:
        y = df.loc[mask, "gt_win"].astype(int).to_numpy()
        if len(y) == 0:
            return {"n": 0.0, "rate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
        lo, hi = _wilson_ci(int(y.sum()), int(len(y)))
        return {"n": float(len(y)), "rate": float(np.mean(y)), "ci_low": lo, "ci_high": hi}

    home_stats = _rate_ci(mask_home.to_numpy() if hasattr(mask_home, "to_numpy") else mask_home)
    away_stats = _rate_ci(mask_away.to_numpy() if hasattr(mask_away, "to_numpy") else mask_away)
    neutral_stats = _rate_ci(mask_neutral.to_numpy() if hasattr(mask_neutral, "to_numpy") else mask_neutral)

    # Hypothesis-style: home vs away difference in win rate (bootstrap CI)
    home_y = df.loc[mask_home, "gt_win"].astype(int).to_numpy()
    away_y = df.loc[mask_away, "gt_win"].astype(int).to_numpy()
    home_away_diff_ci = _bootstrap_diff_ci(home_y.astype(float), away_y.astype(float))

    # Yearly trend
    yearly = (
        df.groupby("year", observed=True)
        .agg(n_games=("gt_win", "size"), wins=("gt_win", "sum"), win_rate=("gt_win", "mean"), margin_mean=("margin", "mean"))
        .reset_index()
        .sort_values("year")
    )
    yearly["win_ci_low"] = np.nan
    yearly["win_ci_high"] = np.nan
    for i, r in yearly.iterrows():
        lo, hi = _wilson_ci(int(r["wins"]), int(r["n_games"]))
        yearly.loc[i, "win_ci_low"] = lo
        yearly.loc[i, "win_ci_high"] = hi

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "yearly.csv").write_text(yearly.to_csv(index=False), encoding="utf-8")

    # Extra yearly stats for dashboard
    extra_rows: list[dict[str, Any]] = []
    for y, sub in df.groupby("year", observed=True):
        sub = sub.copy()
        sub_margin = sub["margin"].astype(float).to_numpy() if "margin" in sub.columns else np.full(len(sub), np.nan)
        sub_n = int(len(sub))
        sub_wins = int(sub["gt_win"].sum())

        # close games
        sub_close_mask = np.isfinite(sub_margin) & (np.abs(sub_margin) <= 8)
        sub_close_y = sub.loc[sub_close_mask, "gt_win"].astype(int).to_numpy()
        sub_close_n = int(len(sub_close_y))
        sub_close_wins = int(sub_close_y.sum()) if sub_close_n else 0
        sub_close_rate = float(sub_close_wins / sub_close_n) if sub_close_n else float("nan")

        # blowouts
        sub_blowout_win = int(np.sum(np.isfinite(sub_margin) & (sub_margin >= 14)))
        sub_blowout_loss = int(np.sum(np.isfinite(sub_margin) & (sub_margin <= -14)))

        # pythag expectation
        if "gt_points" in sub.columns and "opp_points" in sub.columns:
            pf = float(sub["gt_points"].astype(float).sum())
            pa = float(sub["opp_points"].astype(float).sum())
            if pf + pa > 0:
                pyth = (pf**pyth_exp) / ((pf**pyth_exp) + (pa**pyth_exp))
            else:
                pyth = float("nan")
        else:
            pyth = float("nan")

        # opponent Elo proxy
        opp_elo = float("nan")
        if opp_elo_col is not None:
            s = sub[opp_elo_col].astype(float)
            # For gt_opp_elo_avg, week-1 and early weeks can be 0; ignore zeros.
            s = s.replace([0.0], np.nan)
            if s.notna().any():
                # Use mean of the season values (reasonable for opp_pregame_elo), otherwise mean of nonzero gt_opp_elo_avg.
                opp_elo = float(s.mean())

        extra_rows.append(
            {
                "year": int(y),
                "n_games": sub_n,
                "wins": sub_wins,
                "win_rate": float(sub_wins / sub_n) if sub_n else float("nan"),
                "close_n": sub_close_n,
                "close_win_rate": sub_close_rate,
                "blowout_wins_n": sub_blowout_win,
                "blowout_losses_n": sub_blowout_loss,
                "pyth_win_pct": float(pyth),
                "avg_opp_elo_proxy": opp_elo,
            }
        )

    yearly_extras = pd.DataFrame(extra_rows).sort_values("year")
    (out_dir / "yearly_extras.csv").write_text(yearly_extras.to_csv(index=False), encoding="utf-8")

    # Elo diff bin chart data (overall, not by year)
    elo_bins: list[dict[str, Any]] = []
    if np.isfinite(elo_diff).any():
        tmp = df.copy()
        tmp["elo_diff"] = tmp["elo_diff"].astype(float)
        tmp = tmp[np.isfinite(tmp["elo_diff"])]
        # 100-point bins
        lo = math.floor(tmp["elo_diff"].min() / 100.0) * 100
        hi = math.ceil(tmp["elo_diff"].max() / 100.0) * 100
        edges = np.arange(lo, hi + 100, 100)
        tmp["bin"] = pd.cut(tmp["elo_diff"], bins=edges, include_lowest=True)
        g = tmp.groupby("bin", observed=True).agg(n=("gt_win", "size"), win_rate=("gt_win", "mean"), mean_elo_diff=("elo_diff", "mean")).reset_index()
        for _, r in g.iterrows():
            elo_bins.append({"n": int(r["n"]), "win_rate": float(r["win_rate"]), "mean_elo_diff": float(r["mean_elo_diff"])})

    extras = {
        "close_games": {"n": close_n, "win_rate": close_rate, "win_rate_wilson_95": {"low": close_ci[0], "high": close_ci[1]}},
        "blowouts": {"win_rate_margin_ge_14": blowout_win_rate, "loss_rate_margin_le_neg14": blowout_loss_rate},
        "pythagorean": {"exponent": pyth_exp},
        "opp_strength_proxy": {"column_used": opp_elo_col or "none"},
        "elo_gap_bins": elo_bins,
    }
    (out_dir / "extras.json").write_text(json.dumps(extras, indent=2), encoding="utf-8")

    metrics = {
        "years": {"from": year_from, "to": year_to, "include_postseason": include_postseason},
        "record": {"wins": wins, "losses": losses, "n_games": n, "win_rate": winrate, "win_rate_wilson_95": {"low": win_ci[0], "high": win_ci[1]}},
        "margin": {"mean": margin_mean, "mean_bootstrap_95": {"low": margin_ci[0], "high": margin_ci[1]}},
        "splits": {"home": home_stats, "away": away_stats, "neutral": neutral_stats, "home_minus_away_bootstrap_95": {"low": home_away_diff_ci[0], "high": home_away_diff_ci[1]}},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "home_field_effect.json").write_text(json.dumps(home_effect, indent=2), encoding="utf-8")

    # Plots
    # Win rate by year with CI
    plt.figure(figsize=(8, 4))
    plt.bar(yearly["year"].astype(str), yearly["win_rate"], color="#3b82f6")
    plt.errorbar(
        x=np.arange(len(yearly)),
        y=yearly["win_rate"],
        yerr=[yearly["win_rate"] - yearly["win_ci_low"], yearly["win_ci_high"] - yearly["win_rate"]],
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1,
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Win rate")
    plt.title(f"GT win rate by season ({year_from}-{year_to})")
    plt.tight_layout()
    plt.savefig(out_dir / "winrate_by_year.png", dpi=160)
    plt.close()

    # Margin distribution
    if "margin" in df.columns and df["margin"].notna().any():
        plt.figure(figsize=(7, 4))
        sns.histplot(df["margin"].astype(float), bins=20, kde=True)
        plt.axvline(0, color="gray", linestyle="--", linewidth=1)
        plt.title("Margin distribution (GT points - opponent points)")
        plt.xlabel("Margin")
        plt.tight_layout()
        plt.savefig(out_dir / "margin_hist.png", dpi=160)
        plt.close()

        plt.figure(figsize=(9, 4))
        sns.boxplot(data=df, x="year", y="margin")
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Margin by season ({year_from}-{year_to})")
        plt.tight_layout()
        plt.savefig(out_dir / "margin_by_year.png", dpi=160)
        plt.close()

    # Close game win rate by year
    if yearly_extras["close_n"].sum() > 0:
        plt.figure(figsize=(9, 4))
        plt.bar(yearly_extras["year"].astype(str), yearly_extras["close_win_rate"], color="#22c55e")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("Win rate in 1-score games")
        plt.title("Close games (|margin|≤8): win rate by season")
        for i, r in yearly_extras.iterrows():
            plt.text(i, float(r["close_win_rate"]) + 0.02, str(int(r["close_n"])), ha="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "close_games_by_year.png", dpi=160)
        plt.close()

    # Blowout counts by year
    plt.figure(figsize=(9, 4))
    x = np.arange(len(yearly_extras))
    plt.bar(x, yearly_extras["blowout_wins_n"], label="Blowout wins (≥14)", color="#60a5fa")
    plt.bar(x, yearly_extras["blowout_losses_n"], bottom=yearly_extras["blowout_wins_n"], label="Blowout losses (≤-14)", color="#fb7185")
    plt.xticks(x, yearly_extras["year"].astype(str), rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Blowouts by season (counts)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "blowouts_by_year.png", dpi=160)
    plt.close()

    # Pythagorean expected vs actual win%
    if yearly_extras["pyth_win_pct"].notna().any():
        plt.figure(figsize=(9, 4))
        plt.plot(yearly_extras["year"], yearly_extras["win_rate"], marker="o", label="Actual win%")
        plt.plot(yearly_extras["year"], yearly_extras["pyth_win_pct"], marker="o", label="Pythag expected win%")
        plt.ylim(0, 1)
        plt.ylabel("Win rate")
        plt.title(f"Pythagorean expectation vs actual (exponent {pyth_exp})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "pythag_vs_actual.png", dpi=160)
        plt.close()

    # Opponent strength proxy by year
    if yearly_extras["avg_opp_elo_proxy"].notna().any():
        plt.figure(figsize=(9, 4))
        plt.plot(yearly_extras["year"], yearly_extras["avg_opp_elo_proxy"], marker="o", color="#f59e0b")
        plt.ylabel("Avg opponent Elo (proxy)")
        plt.title(f"Opponent strength by season (proxy: {opp_elo_col})")
        plt.tight_layout()
        plt.savefig(out_dir / "opp_strength_by_year.png", dpi=160)
        plt.close()

    # Elo gap vs win rate (binned)
    if len(elo_bins) > 1:
        eb = pd.DataFrame(elo_bins).sort_values("mean_elo_diff")
        plt.figure(figsize=(8, 4))
        plt.plot(eb["mean_elo_diff"], eb["win_rate"], marker="o", color="#a78bfa")
        plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Mean Elo diff in bin (GT − opp)")
        plt.ylabel("Empirical win rate")
        plt.title("Elo gap vs win rate (100-point bins)")
        plt.tight_layout()
        plt.savefig(out_dir / "elo_gap_vs_win.png", dpi=160)
        plt.close()

    print(f"Wrote: {out_dir / 'metrics.json'}")
    print(f"Wrote: {out_dir / 'yearly.csv'}")
    print(f"Wrote: {out_dir / 'winrate_by_year.png'}")
    print(f"Wrote: {out_dir / 'home_field_effect.json'}")
    if (out_dir / "margin_hist.png").exists():
        print(f"Wrote: {out_dir / 'margin_hist.png'}")
    if (out_dir / "margin_by_year.png").exists():
        print(f"Wrote: {out_dir / 'margin_by_year.png'}")
    if (out_dir / "yearly_extras.csv").exists():
        print(f"Wrote: {out_dir / 'yearly_extras.csv'}")
    if (out_dir / "extras.json").exists():
        print(f"Wrote: {out_dir / 'extras.json'}")
    for p in (
        "close_games_by_year.png",
        "blowouts_by_year.png",
        "pythag_vs_actual.png",
        "opp_strength_by_year.png",
        "elo_gap_vs_win.png",
    ):
        if (out_dir / p).exists():
            print(f"Wrote: {out_dir / p}")


def main() -> int:
    p = argparse.ArgumentParser(description="Statistical analysis + plots for GT win-prob backtests.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("backtest", help="Merge predictions with actual results and output stats/plots")
    r.add_argument("--year", type=int, required=True)
    r.add_argument("--pred", type=Path, default=None, help="Predictions CSV (default: data_processed/predictions_<year>.csv)")
    r.add_argument("--dataset", type=Path, default=Path("data_processed") / "model_dataset.csv")
    r.add_argument("--out-dir", type=Path, default=Path("reports"))

    s = sub.add_parser("gt-summary", help="Descriptive stats + CIs for GT games in a year range")
    s.add_argument("--from-year", type=int, default=2014)
    s.add_argument("--to-year", type=int, default=2025)
    s.add_argument("--dataset", type=Path, default=Path("data_processed") / "model_dataset.csv")
    s.add_argument("--out-dir", type=Path, default=Path("reports"))
    s.add_argument("--include-postseason", action="store_true")

    m = sub.add_parser("market-plot", help="Create diagrams comparing model probabilities vs market implied probabilities")
    m.add_argument("--year", type=int, default=None, help="Year label for output filenames/titles (optional)")
    m.add_argument("--csv", type=Path, default=Path("game_by_game_comparison_table.csv"), help="Input CSV (default: game_by_game_comparison_table.csv)")
    m.add_argument("--out-dir", type=Path, default=Path("reports") / "market", help="Output directory for images")

    mf = sub.add_parser("market-vs-fpi", help="Bar chart comparing market win% vs ESPN FPI win% (GT)")
    mf.add_argument("--year", type=int, default=None)
    mf.add_argument("--market", type=Path, default=Path("data_processed") / "market_probs_2025_manual.csv")
    mf.add_argument("--fpi", type=Path, default=Path("data_processed") / "fpi_probs_2025_espn.csv")
    mf.add_argument("--out-dir", type=Path, default=Path("reports") / "market_manual")

    mvf = sub.add_parser("model-vs-fpi", help="Bar chart comparing model win% vs ESPN FPI win% (GT)")
    mvf.add_argument("--year", type=int, required=True)
    mvf.add_argument("--backtest", type=Path, default=None, help="Backtest CSV (default: reports/<year>/backtest_<year>.csv)")
    mvf.add_argument("--fpi", type=Path, default=Path("data_processed") / "fpi_probs_2025_espn.csv")
    mvf.add_argument("--out-dir", type=Path, default=Path("reports") / "model_fpi")

    args = p.parse_args()

    if args.cmd == "backtest":
        pred = args.pred or (Path("data_processed") / f"predictions_{args.year}.csv")
        inputs = BacktestInputs(
            year=args.year,
            predictions_csv=pred,
            dataset_csv=args.dataset,
            out_dir=args.out_dir / str(args.year),
        )
        write_report(inputs)
        return 0

    if args.cmd == "gt-summary":
        out_dir = args.out_dir / f"gt_summary_{args.from_year}_{args.to_year}"
        gt_summary(
            dataset_csv=args.dataset,
            year_from=args.from_year,
            year_to=args.to_year,
            out_dir=out_dir,
            include_postseason=bool(args.include_postseason),
        )
        return 0

    if args.cmd == "market-plot":
        plot_model_vs_market(args.csv, args.out_dir, year=args.year)
        print(f"Wrote: {args.out_dir / ('model_vs_market_scatter' + (f'_{args.year}' if args.year else '') + '.png')}")
        print(f"Wrote: {args.out_dir / ('model_minus_market_delta' + (f'_{args.year}' if args.year else '') + '.png')}")
        print(f"Wrote: {args.out_dir / ('model_vs_market_dumbbell' + (f'_{args.year}' if args.year else '') + '.png')}")
        return 0

    if args.cmd == "market-vs-fpi":
        plot_market_vs_fpi(market_csv=args.market, fpi_csv=args.fpi, out_dir=args.out_dir, year=args.year)
        suffix = f"_{args.year}" if args.year else ""
        print(f"Wrote: {args.out_dir / ('market_vs_fpi_bar' + suffix + '.png')}")
        return 0

    if args.cmd == "model-vs-fpi":
        backtest = args.backtest or (Path("reports") / str(args.year) / f"backtest_{args.year}.csv")
        plot_model_vs_fpi(backtest_csv=backtest, fpi_csv=args.fpi, out_dir=args.out_dir, year=args.year)
        print(f"Wrote: {args.out_dir / ('model_vs_fpi_bar_' + str(args.year) + '.png')}")
        return 0

    raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
