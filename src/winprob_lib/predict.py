from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .cfbd_client import CFBDClient, _is_fbs_team, _team_classification_map, resolve_team_name
from .features import FCS_ELO_FLOOR, NONFBS_RECRUIT_POINTS_FLOOR, NONFBS_RECRUIT_RANK_FLOOR, NONFBS_TALENT_FLOOR
from .features import (
    _compute_prior_team_form,
    _elo_snapshot,
    _preseason_elo_estimate,
    _resolve_recruit_points_diff,
    _resolve_recruit_rank_diff,
    _resolve_talent_diff,
)
from .train import _apply_calibrator
from .util import _as_float, _get_any, _index_by_team

# The 5 features whose live-vs-fallback status meaningfully varies game to game (unlike
# location/context, which are always known, or in-season form, which is legitimately 0 before
# a team's first game rather than "missing"). Used to build the completeness/source columns
# in predict_schedule()'s output -- see _fetch_matchup_features()'s second return value.
ROSTER_STRENGTH_FEATURES = ["elo_diff", "talent_diff", "returning_diff", "recruit_points_diff", "recruit_rank_diff"]


def _fetch_matchup_features(
    *,
    team: str,
    year: int,
    week: int,
    opponent: str,
    home: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    client = CFBDClient()
    class_map = _team_classification_map(client, year=year, use_cache=True)

    is_neutral = 1 if home.lower() == "neutral" else 0
    is_home = 1 if home.lower() == "home" else 0
    opp_is_fbs = _is_fbs_team(class_map, opponent)

    # Season-to-date form (pregame) from games already played in this season.
    team_games = client.get("/games", {"year": year, "seasonType": "regular", "team": team}, use_cache=True)
    opp_games = client.get("/games", {"year": year, "seasonType": "regular", "team": opponent}, use_cache=True)
    team_games = team_games if isinstance(team_games, list) else []
    opp_games = opp_games if isinstance(opp_games, list) else []
    gt_form = _compute_prior_team_form(games_for_team=team_games, team=team, week=week, class_map=class_map)
    opp_form = _compute_prior_team_form(games_for_team=opp_games, team=opponent, week=week, class_map=class_map)

    # Elo as-of previous week (if present).
    elo_diff = None
    elo_source = "missing"
    try:
        snap = _elo_snapshot(client, year=year, week=max(1, week - 1))
        gt_elo = _as_float((snap.get(team) or {}).get("elo"))
        opp_elo = _as_float((snap.get(opponent) or {}).get("elo"))
        if gt_elo is not None and opp_elo is not None:
            elo_diff = gt_elo - opp_elo
            elo_source = "live"
        elif opp_is_fbs == 0 and gt_elo is not None:
            elo_diff = float(gt_elo - FCS_ELO_FLOOR)
            elo_source = "live"
    except Exception:
        elo_diff = None

    # Fallback: no current-season Elo published yet at all (e.g. predicting before/early in
    # a season CFBD hasn't rated yet) -- use each team's prior-season-final Elo, regressed
    # toward the mean via the empirically-fit carryover estimate.
    if elo_diff is None:
        gt_elo_carry = _preseason_elo_estimate(client, team=team, year=year, use_cache=True)
        opp_elo_carry = _preseason_elo_estimate(client, team=opponent, year=year, use_cache=True)
        if gt_elo_carry is not None and opp_elo_carry is not None:
            elo_diff = gt_elo_carry - opp_elo_carry
            elo_source = "carryover"
        elif opp_is_fbs == 0 and gt_elo_carry is not None:
            elo_diff = float(gt_elo_carry - FCS_ELO_FLOOR)
            elo_source = "carryover"

    # Season-level
    def _try_index(endpoint: str, params: dict[str, Any], team_field: str) -> dict[str, dict[str, Any]]:
        try:
            items = client.get(endpoint, params, use_cache=True)
            if isinstance(items, list):
                return _index_by_team(items, team_field=team_field)
        except Exception:
            return {}
        return {}

    talent = _try_index("/talent", {"year": year}, team_field="team")
    returning = _try_index("/player/returning", {"year": year}, team_field="team")
    recruiting = _try_index("/recruiting/teams", {"year": year}, team_field="team")

    def _source(live_team: Any, live_opp: Any, resolved: float | None) -> str:
        if resolved is None:
            return "missing"
        if live_team is not None and (live_opp is not None or opp_is_fbs == 0):
            # An opponent-side floor substitution (non-FBS opponent, no live value) is a
            # defined convention, not an uncertain estimate -- don't count it as "carryover".
            return "live"
        return "carryover"

    talent_diff = _resolve_talent_diff(
        client, talent, team=team, opponent=opponent, opp_is_fbs=opp_is_fbs, year=year, use_cache=True
    )
    talent_source = _source(
        (talent.get(team) or {}).get("talent"), (talent.get(opponent) or {}).get("talent"), talent_diff
    )

    def _returning_total(v: dict[str, Any]) -> float | None:
        for key in ("totalPPA", "total", "totalPpa", "total_ppa", "total_returning"):
            out = _as_float(v.get(key))
            if out is not None:
                return out
        out = _as_float(v.get("ppa"))
        return out

    gt_ret = _returning_total(returning.get(team) or {})
    opp_ret = _returning_total(returning.get(opponent) or {})
    # No carryover for returning production -- see the note above _preseason_talent_estimate()
    # in features.py for why (prior-year value doesn't predict this-year value, R^2=0.026).
    returning_diff = (gt_ret - opp_ret) if gt_ret is not None and opp_ret is not None else None
    returning_source = "live" if returning_diff is not None else "missing"

    recruit_points_diff = _resolve_recruit_points_diff(
        client, recruiting, team=team, opponent=opponent, opp_is_fbs=opp_is_fbs, year=year, use_cache=True
    )
    recruit_points_source = _source(
        (recruiting.get(team) or {}).get("points"), (recruiting.get(opponent) or {}).get("points"), recruit_points_diff
    )

    recruit_rank_diff = _resolve_recruit_rank_diff(
        client, recruiting, team=team, opponent=opponent, opp_is_fbs=opp_is_fbs, year=year, use_cache=True
    )
    recruit_rank_source = _source(
        (recruiting.get(team) or {}).get("rank"), (recruiting.get(opponent) or {}).get("rank"), recruit_rank_diff
    )

    sources = {
        "elo_diff": elo_source,
        "talent_diff": talent_source,
        "returning_diff": returning_source,
        "recruit_points_diff": recruit_points_source,
        "recruit_rank_diff": recruit_rank_source,
    }
    n_live = sum(1 for f in ROSTER_STRENGTH_FEATURES if sources[f] == "live")
    meta = {
        "feature_completeness": n_live / len(ROSTER_STRENGTH_FEATURES),
        "carryover_features": ",".join(f for f in ROSTER_STRENGTH_FEATURES if sources[f] == "carryover"),
        "missing_features": ",".join(f for f in ROSTER_STRENGTH_FEATURES if sources[f] == "missing"),
    }

    X = pd.DataFrame(
        [
            {
                "is_home": is_home,
                "neutral_site": is_neutral,
                "opp_is_fbs": float(opp_is_fbs),
                "win_pct_diff": gt_form["win_pct"] - opp_form["win_pct"],
                "point_diff_pg_diff": gt_form["point_diff_pg"] - opp_form["point_diff_pg"],
                "points_for_pg_diff": gt_form["points_for_pg"] - opp_form["points_for_pg"],
                "points_against_pg_diff": gt_form["points_against_pg"] - opp_form["points_against_pg"],
                "w_win_pct_diff": gt_form["w_win_pct"] - opp_form["w_win_pct"],
                "w_point_diff_pg_diff": gt_form["w_point_diff_pg"] - opp_form["w_point_diff_pg"],
                "w_points_for_pg_diff": gt_form["w_points_for_pg"] - opp_form["w_points_for_pg"],
                "w_points_against_pg_diff": gt_form["w_points_against_pg"] - opp_form["w_points_against_pg"],
                "opp_elo_avg_diff": gt_form["opp_elo_avg"] - opp_form["opp_elo_avg"],
                "w_opp_elo_avg_diff": gt_form["w_opp_elo_avg"] - opp_form["w_opp_elo_avg"],
                "sos_point_diff_pg_diff": gt_form["sos_point_diff_pg"] - opp_form["sos_point_diff_pg"],
                "w_sos_point_diff_pg_diff": gt_form["w_sos_point_diff_pg"] - opp_form["w_sos_point_diff_pg"],
                "elo_diff": elo_diff,
                "talent_diff": talent_diff,
                "returning_diff": returning_diff,
                "recruit_points_diff": recruit_points_diff,
                "recruit_rank_diff": recruit_rank_diff,
            }
        ]
    )
    return X, meta


def _load_model_bundle(model_path: Path) -> tuple[Any, Any | None, list[str] | None]:
    pipe = joblib.load(model_path)
    is_margin = "margin" in model_path.name.lower()

    calibrator = None
    calibrator_path = model_path.parent / "gt_winprob_calibrator.joblib"
    if calibrator_path.exists():
        calibrator = joblib.load(calibrator_path)
    else:
        # Legacy fallback.
        legacy_path = model_path.parent / "gt_winprob_calibrator_isotonic.joblib"
        calibrator = joblib.load(legacy_path) if legacy_path.exists() else None

    # Margin-target and win-target artifacts use separate sidecar filenames (see
    # train_model() in train.py) so training one target doesn't overwrite the other's.
    feature_path = model_path.parent / ("feature_columns_margin.json" if is_margin else "feature_columns.json")
    feature_cols: list[str] | None = None
    if feature_path.exists():
        try:
            feature_cols = json.loads(feature_path.read_text(encoding="utf-8"))
        except Exception:
            feature_cols = None
    return pipe, calibrator, feature_cols


def _load_model_meta(model_path: Path) -> dict[str, Any]:
    is_margin = "margin" in model_path.name.lower()
    meta_path = model_path.parent / ("model_meta_margin.json" if is_margin else "model_meta.json")
    if not meta_path.exists():
        # Best-effort guess for older artifacts.
        return {"target": "margin"} if is_margin else {"target": "win"}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {"target": "margin"} if is_margin else {"target": "win"}


def _norm_cdf_scalar(x: float) -> float:
    return float(0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))


def _predict_pwin(
    *,
    model_path: Path,
    pipe: Any,
    calibrator: Any | None,
    X: pd.DataFrame,
) -> tuple[float, dict[str, Any]]:
    meta = _load_model_meta(model_path)
    target = str(meta.get("target", "win")).lower()

    if target == "margin":
        pred_margin = float(pipe.predict(X)[0])
        sigma_path = model_path.parent / "margin_sigma.json"
        sigma = 14.0
        if sigma_path.exists():
            try:
                sigma = float(json.loads(sigma_path.read_text(encoding="utf-8")).get("sigma", sigma))
            except Exception:
                sigma = sigma
        sigma = sigma if sigma > 1e-6 else 14.0
        p_win = _norm_cdf_scalar(pred_margin / sigma)
        return p_win, {"model_target": "margin", "pred_margin": pred_margin, "sigma": sigma, "calibrator": "margin_normal"}

    # win target
    p_raw = float(pipe.predict_proba(X)[0, 1])
    p_win, cal_type = _apply_calibrator(p_raw, calibrator)
    return p_win, {"model_target": "win", "p_win_raw": p_raw, "calibrator": cal_type}


def predict_schedule(
    *,
    year: int,
    team: str = "Georgia Tech",
    season_type: str = "regular",
    model_path: Path = Path("models") / "gt_winprob_logreg.joblib",
) -> pd.DataFrame:
    pipe, calibrator, feature_cols = _load_model_bundle(model_path)
    client = CFBDClient()
    team_resolved, _cands = resolve_team_name(client, team, use_cache=True)
    team = team_resolved

    games = client.get("/games", {"year": year, "seasonType": season_type, "team": team}, use_cache=True)
    if not isinstance(games, list):
        raise RuntimeError("CFBD /games did not return a list; cannot predict schedule.")

    out_rows: list[dict[str, Any]] = []
    for g in games:
        if not isinstance(g, dict):
            continue
        week = _get_any(g, "week")
        if not isinstance(week, int):
            continue

        home_team = _get_any(g, "homeTeam", "home_team")
        away_team = _get_any(g, "awayTeam", "away_team")
        neutral = bool(_get_any(g, "neutralSite", "neutral_site"))
        if not isinstance(home_team, str) or not isinstance(away_team, str):
            continue
        if team != home_team and team != away_team:
            continue

        if team == home_team:
            opponent = away_team
            home_flag = "neutral" if neutral else "home"
        else:
            opponent = home_team
            home_flag = "neutral" if neutral else "away"

        X_full, feature_meta = _fetch_matchup_features(team=team, year=year, week=week, opponent=opponent, home=home_flag)
        X = X_full.reindex(columns=feature_cols) if feature_cols else X_full
        p_win, extra = _predict_pwin(model_path=model_path, pipe=pipe, calibrator=calibrator, X=X)

        out_rows.append(
            {
                "game_id": _get_any(g, "id"),
                "season": _get_any(g, "season"),
                "week": week,
                "start_date": _get_any(g, "startDate"),
                "opponent": opponent,
                "location": home_flag,
                "neutral_site": int(neutral),
                "p_win": p_win,
                **extra,
                **feature_meta,
            }
        )

    df = pd.DataFrame(out_rows)
    if not df.empty:
        df = df.sort_values(["week", "start_date", "opponent"], kind="stable")
    return df
