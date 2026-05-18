from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from winprob import CFBDClient, resolve_team_name


def _moneyline_to_implied_prob(ml: float) -> float | None:
    if not np.isfinite(ml):
        return None
    if ml == 0:
        return None
    if ml < 0:
        return float((-ml) / ((-ml) + 100.0))
    return float(100.0 / (ml + 100.0))


def _pick_line(lines: list[dict[str, Any]], provider_preference: str | None) -> dict[str, Any] | None:
    if not lines:
        return None

    def has_moneyline(x: dict[str, Any]) -> bool:
        return any(k in x and x.get(k) is not None for k in ("homeMoneyline", "awayMoneyline", "home_moneyline", "away_moneyline"))

    pref = (provider_preference or "").strip().lower()
    if pref:
        for l in lines:
            p = str(l.get("provider") or "").strip().lower()
            if p == pref:
                return l
    # Otherwise: prefer a line with moneylines
    for l in lines:
        if has_moneyline(l):
            return l
    return lines[0]


def _get_any(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def build_model_vs_market(
    *,
    year: int,
    team: str,
    predictions_csv: Path,
    dataset_csv: Path,
    out_csv: Path,
    provider: str | None = None,
) -> pd.DataFrame:
    pred = pd.read_csv(predictions_csv)
    ds = pd.read_csv(dataset_csv)

    client = CFBDClient()
    team_resolved, _ = resolve_team_name(client, team, use_cache=True)
    team = team_resolved

    # Pull betting lines for this team/year (regular season)
    lines_items = client.get("/lines", {"year": year, "seasonType": "regular", "team": team}, use_cache=True)
    lines_items = lines_items if isinstance(lines_items, list) else []

    # Index betting lines by game id
    by_game_id: dict[int, dict[str, Any]] = {}
    for item in lines_items:
        if not isinstance(item, dict):
            continue
        gid = _get_any(item, "id", "gameId", "game_id")
        try:
            gid_i = int(gid)
        except Exception:
            continue
        by_game_id[gid_i] = item

    # Actual results for year (if present)
    ds_y = ds[ds["year"].astype(int) == int(year)][["game_id", "week", "opponent", "gt_win"]].copy()
    if "game_id" in pred.columns and "game_id" in ds_y.columns:
        merged = pred.merge(ds_y[["game_id", "gt_win"]], on="game_id", how="left")
    else:
        merged = pred.merge(ds_y[["week", "opponent", "gt_win"]], on=["week", "opponent"], how="left")

    out_rows: list[dict[str, Any]] = []
    for _, r in merged.iterrows():
        gid = r.get("game_id")
        try:
            gid_i = int(gid)
        except Exception:
            gid_i = None
        li = by_game_id.get(gid_i) if gid_i is not None else None
        line_used = None
        mkt_p = None
        ml = None
        prov = None

        if isinstance(li, dict):
            lines = li.get("lines") or []
            lines = lines if isinstance(lines, list) else []
            picked = _pick_line([x for x in lines if isinstance(x, dict)], provider)
            line_used = picked
            if picked:
                prov = picked.get("provider")
                home_team = _get_any(li, "homeTeam", "home_team")
                away_team = _get_any(li, "awayTeam", "away_team")
                home_ml = _get_any(picked, "homeMoneyline", "home_moneyline")
                away_ml = _get_any(picked, "awayMoneyline", "away_moneyline")
                # Determine GT's moneyline
                if isinstance(home_team, str) and home_team == team:
                    ml = float(home_ml) if home_ml is not None and not (isinstance(home_ml, str) and home_ml.strip() == "") else None
                elif isinstance(away_team, str) and away_team == team:
                    ml = float(away_ml) if away_ml is not None and not (isinstance(away_ml, str) and away_ml.strip() == "") else None
                if ml is not None and np.isfinite(ml):
                    mkt_p = _moneyline_to_implied_prob(float(ml))

        p_win = float(r.get("p_win")) if not pd.isna(r.get("p_win")) else float("nan")
        delta = (p_win - mkt_p) if (mkt_p is not None and np.isfinite(p_win)) else float("nan")
        gt_win = r.get("gt_win")

        out_rows.append(
            {
                "year": int(year),
                "week": int(r.get("week")) if not pd.isna(r.get("week")) else None,
                "game_id": gid_i,
                "opponent": r.get("opponent"),
                "location": r.get("location"),
                "p_win_model": p_win,
                "p_win_market": mkt_p,
                "delta_model_minus_market": delta,
                "market_provider": prov,
                "gt_moneyline": ml,
                "gt_win": int(gt_win) if not pd.isna(gt_win) else None,
            }
        )

    out_df = pd.DataFrame(out_rows).sort_values(["week", "opponent"], kind="stable")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_df


def main() -> int:
    p = argparse.ArgumentParser(description="Compare model win probs vs market implied probs from CFBD betting lines.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build a model-vs-market CSV for a season")
    b.add_argument("--year", type=int, required=True)
    b.add_argument("--team", type=str, default="Georgia Tech")
    b.add_argument("--pred", type=Path, default=None, help="Predictions CSV (default: data_processed/predictions_<year>.csv)")
    b.add_argument("--dataset", type=Path, default=Path("data_processed") / "model_dataset.csv")
    b.add_argument("--out", type=Path, default=None, help="Output CSV path")
    b.add_argument("--provider", type=str, default=None, help="Preferred provider name (optional)")

    args = p.parse_args()
    if args.cmd == "build":
        pred = args.pred or (Path("data_processed") / f"predictions_{args.year}.csv")
        out = args.out or (Path("data_processed") / f"model_vs_market_{args.year}.csv")
        df = build_model_vs_market(
            year=args.year,
            team=args.team,
            predictions_csv=pred,
            dataset_csv=args.dataset,
            out_csv=out,
            provider=args.provider,
        )
        print(f"Wrote {len(df)} rows to {out}")
        missing = int(df["p_win_market"].isna().sum()) if "p_win_market" in df.columns else 0
        if missing:
            print(f"Note: {missing} rows missing market probability (moneyline not available).")
        return 0
    raise RuntimeError("unknown cmd")


if __name__ == "__main__":
    raise SystemExit(main())

