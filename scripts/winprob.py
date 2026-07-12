from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Re-exported so `from winprob import CFBDClient, resolve_team_name` (used by
# scripts/market_compare.py) keeps working now that the implementation lives in
# src/winprob_lib/.
from winprob_lib.cfbd_client import CFBDClient, resolve_team_name  # noqa: F401
from winprob_lib.features import build_dataset, build_dataset_all_fbs
from winprob_lib.predict import _fetch_matchup_features, _load_model_bundle, _predict_pwin, predict_schedule
from winprob_lib.train import train_model, walk_forward_eval


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="winprob.py", description="Train/predict Georgia Tech win probability from CFBD.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-dataset", help="Download CFBD data and build data/model_dataset.csv")
    b.add_argument("--from-year", type=int, default=2014)
    b.add_argument("--to-year", type=int, default=2025)
    b.add_argument("--team", type=str, default="Georgia Tech", help="Used only when --scope=team")
    b.add_argument(
        "--scope",
        choices=["team", "all-fbs"],
        default="team",
        help=(
            "team: one team's schedule only (--team). all-fbs: pool every FBS team's games "
            "(both home/away perspective) into one team-agnostic training set - far more "
            "rows, same relative-diff feature schema. (default: team)"
        ),
    )
    b.add_argument("--out", type=Path, default=Path("data_processed") / "model_dataset.csv")
    b.add_argument("--no-postseason", action="store_true")

    t = sub.add_parser("train", help="Train model and write models/ artifacts")
    t.add_argument("--dataset", type=Path, default=Path("data_processed") / "model_dataset.csv")
    t.add_argument("--half-life-years", type=float, default=3.0, help="Recency weighting half-life in years (default: 3)")
    t.add_argument(
        "--calibration",
        choices=["auto", "isotonic", "sigmoid", "none"],
        default="auto",
        help="Probability calibration method (default: auto)",
    )
    t.add_argument("--target", choices=["win", "margin"], default="win", help="Train target (default: win)")

    pr = sub.add_parser("predict", help="Predict win probability for a matchup")
    pr.add_argument("--year", type=int, required=True)
    pr.add_argument("--week", type=int, required=True)
    pr.add_argument("--opponent", type=str, required=True)
    pr.add_argument("--home", choices=["home", "away", "neutral"], required=True)
    pr.add_argument("--model", type=Path, default=Path("models") / "gt_winprob_logreg.joblib")

    ps = sub.add_parser("predict-season", help="Predict win probability for every regular-season game in CFBD schedule")
    ps.add_argument("--year", type=int, required=True)
    ps.add_argument("--team", type=str, default="Georgia Tech")
    ps.add_argument("--out", type=Path, default=None)
    ps.add_argument("--model", type=Path, default=Path("models") / "gt_winprob_logreg.joblib")

    wf = sub.add_parser(
        "walkforward",
        help="Rolling walk-forward backtest: retrain per season and pool held-out predictions across many years",
    )
    wf.add_argument("--dataset", type=Path, default=Path("data_processed") / "model_dataset.csv")
    wf.add_argument("--start-test-year", type=int, default=2019)
    wf.add_argument("--end-test-year", type=int, default=None)
    wf.add_argument("--half-life-years", type=float, default=3.0)
    wf.add_argument("--calibration", choices=["auto", "isotonic", "sigmoid", "none"], default="auto")
    wf.add_argument(
        "--eval-team",
        type=str,
        default=None,
        help="Restrict held-out test rows to this team's games only (requires an all-fbs pooled dataset)",
    )
    wf.add_argument("--out-dir", type=Path, default=Path("reports") / "walkforward")

    args = p.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "build-dataset":
        if args.scope == "all-fbs":
            df = build_dataset_all_fbs(
                year_from=args.from_year,
                year_to=args.to_year,
                include_postseason=not args.no_postseason,
            )
        else:
            df = build_dataset(
                year_from=args.from_year,
                year_to=args.to_year,
                team=args.team,
                include_postseason=not args.no_postseason,
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df)} rows to {args.out}")
        return 0

    if args.cmd == "train":
        result = train_model(
            args.dataset,
            half_life_years=args.half_life_years,
            calibration=args.calibration,
            target=args.target,
        )
        print(json.dumps(result["metrics"], indent=2))
        print("Wrote models/gt_winprob_logreg.joblib and models/metrics.json")
        return 0

    if args.cmd == "predict":
        pipe, calibrator, feature_cols = _load_model_bundle(args.model)
        X_full = _fetch_matchup_features(
            team="Georgia Tech",
            year=args.year,
            week=args.week,
            opponent=args.opponent,
            home=args.home,
        )
        X = X_full.reindex(columns=feature_cols) if feature_cols else X_full
        p_win, extra = _predict_pwin(model_path=args.model, pipe=pipe, calibrator=calibrator, X=X)
        out: dict = {
            "team": "Georgia Tech",
            "opponent": args.opponent,
            "year": args.year,
            "week": args.week,
            "p_win": p_win,
            **extra,
        }
        print(json.dumps(out, indent=2))
        return 0

    if args.cmd == "predict-season":
        df = predict_schedule(year=args.year, team=args.team, model_path=args.model)
        out_path = args.out
        if out_path is None:
            out_path = Path("data_processed") / f"predictions_{args.year}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} predictions to {out_path}")
        return 0

    if args.cmd == "walkforward":
        result = walk_forward_eval(
            args.dataset,
            start_test_year=args.start_test_year,
            end_test_year=args.end_test_year,
            half_life_years=args.half_life_years,
            calibration=args.calibration,
            eval_team=args.eval_team,
            out_dir=args.out_dir,
        )
        print(
            json.dumps(
                {
                    "n_games": result["n_games"],
                    "n_folds": result["n_folds"],
                    "years": f"{result['start_test_year']}-{result['end_test_year']}",
                    "raw": result["raw"],
                    "calibrated": result["calibrated"],
                },
                indent=2,
            )
        )
        print(f"Wrote {args.out_dir / 'walkforward_predictions.csv'} and {args.out_dir / 'walkforward_metrics.json'}")
        return 0

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
