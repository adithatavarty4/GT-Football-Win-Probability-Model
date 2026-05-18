from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CFBD_BASE_URL = "https://api.collegefootballdata.com"

FCS_ELO_FLOOR = 800.0
NONFBS_RECRUIT_POINTS_FLOOR = 0.0
NONFBS_RECRUIT_RANK_FLOOR = 999.0
NONFBS_TALENT_FLOOR = 0.0


def _stable_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: params[k] for k in sorted(params.keys()) if params[k] is not None}


def _cache_key(endpoint: str, params: dict[str, Any]) -> str:
    stable = json.dumps({"endpoint": endpoint, "params": _stable_params(params)}, sort_keys=True, separators=(",", ":"))
    # Stable across runs (unlike Python's built-in hash()).
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]


def _get_any(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def _norm_team_name(name: str) -> str:
    return " ".join(name.strip().casefold().split())


def resolve_team_name(client: CFBDClient, team_input: str, *, use_cache: bool = True) -> tuple[str, list[str]]:
    """
    CFBD endpoints aren't always consistent about which strings are accepted for the `team` filter.
    Resolve a user-provided team string to the canonical `school` value from `/teams`.
    Returns (resolved_school_name, candidate_school_names).
    """
    try:
        items = client.get("/teams", {}, use_cache=use_cache)
    except Exception:
        return team_input, []
    if not isinstance(items, list):
        return team_input, []

    want = _norm_team_name(team_input)
    exact: list[dict[str, Any]] = []
    fuzzy: list[dict[str, Any]] = []

    for t in items:
        if not isinstance(t, dict):
            continue
        school = t.get("school")
        abbr = t.get("abbreviation")
        alts = t.get("alternateNames") or []
        if not isinstance(school, str):
            continue

        school_n = _norm_team_name(school)
        abbr_n = _norm_team_name(abbr) if isinstance(abbr, str) else ""
        alt_ns = [_norm_team_name(a) for a in alts if isinstance(a, str)]

        if want in (school_n, abbr_n) or want in alt_ns:
            exact.append(t)
            continue
        if want and (want in school_n or school_n in want):
            fuzzy.append(t)

    candidates = exact or fuzzy
    candidate_schools = [c.get("school") for c in candidates if isinstance(c.get("school"), str)]
    resolved = candidate_schools[0] if candidate_schools else team_input
    return resolved, candidate_schools


@dataclass(frozen=True)
class CFBDClient:
    api_key_env: str = "CFBD_API_KEY"
    base_url: str = CFBD_BASE_URL
    cache_dir: Path = Path("data_raw") / "cfbd_cache"
    timeout_s: int = 45

    def _headers(self) -> dict[str, str]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key env var {self.api_key_env}. "
                f"Set it in PowerShell: $env:{self.api_key_env}='YOUR_KEY'"
            )
        return {"Authorization": f"Bearer {api_key}"}

    def get(self, endpoint: str, params: dict[str, Any], *, use_cache: bool = True) -> Any:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(endpoint, params)
        cache_path = self.cache_dir / f"{endpoint.strip('/').replace('/', '__')}-{key}.json"
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        resp = requests.get(url, headers=self._headers(), params=_stable_params(params), timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        if use_cache:
            cache_path.write_text(json.dumps(data), encoding="utf-8")
        return data


def _team_classification_map(client: CFBDClient, *, use_cache: bool = True) -> dict[str, str]:
    """
    Map CFBD `school` -> `classification` (fbs/fcs/...)
    """
    items = client.get("/teams", {}, use_cache=use_cache)
    if not isinstance(items, list):
        return {}
    out: dict[str, str] = {}
    for t in items:
        if not isinstance(t, dict):
            continue
        school = t.get("school")
        classification = t.get("classification")
        if isinstance(school, str) and isinstance(classification, str):
            out[school] = classification.lower().strip()
    return out


def _is_fbs_team(class_map: dict[str, str], team: str) -> int:
    return 1 if class_map.get(team, "").lower() == "fbs" else 0


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    if isinstance(value, str):
        v = value.strip()
        if v == "":
            return None
        try:
            out = float(v)
            if np.isfinite(out):
                return out
        except ValueError:
            return None
    return None


def _index_by_team(items: list[dict[str, Any]], team_field: str = "team") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        team = item.get(team_field)
        if isinstance(team, str) and team.strip():
            out[team.strip()] = item
    return out


def _winner_label(team: str, game: dict[str, Any]) -> int | None:
    home = _get_any(game, "home_team", "homeTeam")
    away = _get_any(game, "away_team", "awayTeam")
    hp = _get_any(game, "home_points", "homePoints")
    ap = _get_any(game, "away_points", "awayPoints")
    if not isinstance(home, str) or not isinstance(away, str):
        return None
    if hp is None or ap is None:
        return None
    try:
        hp_i = int(hp)
        ap_i = int(ap)
    except Exception:
        return None

    if team == home:
        return 1 if hp_i > ap_i else 0
    if team == away:
        return 1 if ap_i > hp_i else 0
    return None


def _opponent_and_flags(team: str, game: dict[str, Any]) -> tuple[str | None, int | None, int | None]:
    home = _get_any(game, "home_team", "homeTeam")
    away = _get_any(game, "away_team", "awayTeam")
    neutral = _get_any(game, "neutral_site", "neutralSite")
    if not isinstance(home, str) or not isinstance(away, str):
        return None, None, None
    if team == home:
        return away, 1, 1 if bool(neutral) else 0
    if team == away:
        return home, 0, 1 if bool(neutral) else 0
    return None, None, None


def _team_game_points(team: str, game: dict[str, Any]) -> tuple[int | None, int | None]:
    """
    Returns (team_points, opp_points) for a completed game dict, or (None, None) if unavailable.
    """
    home = _get_any(game, "home_team", "homeTeam")
    away = _get_any(game, "away_team", "awayTeam")
    hp = _get_any(game, "home_points", "homePoints")
    ap = _get_any(game, "away_points", "awayPoints")
    if not isinstance(home, str) or not isinstance(away, str):
        return None, None
    if hp is None or ap is None:
        return None, None
    try:
        hp_i = int(hp)
        ap_i = int(ap)
    except Exception:
        return None, None

    if team == home:
        return hp_i, ap_i
    if team == away:
        return ap_i, hp_i
    return None, None


def _game_opponent_and_elo(
    *,
    team: str,
    game: dict[str, Any],
) -> tuple[str | None, float | None]:
    """
    Returns (opponent_name, opponent_pregame_elo) for this game, or (None, None).
    """
    home = _get_any(game, "home_team", "homeTeam")
    away = _get_any(game, "away_team", "awayTeam")
    if not isinstance(home, str) or not isinstance(away, str):
        return None, None
    home_elo = _as_float(_get_any(game, "homePregameElo", "home_pregame_elo"))
    away_elo = _as_float(_get_any(game, "awayPregameElo", "away_pregame_elo"))

    if team == home:
        return away, away_elo
    if team == away:
        return home, home_elo
    return None, None


def _compute_prior_team_form(
    *,
    games_for_team: list[dict[str, Any]],
    team: str,
    week: int,
    half_life_weeks: float = 4.0,
    class_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """
    Compute season-to-date features using only games with week < `week` and completed==true.
    """
    played = 0
    wins = 0
    points_for = 0
    points_against = 0
    weight_sum = 0.0
    weight_wins = 0.0
    weight_pf = 0.0
    weight_pa = 0.0
    opp_elo_sum = 0.0
    opp_elo_w_sum = 0.0
    sos_pd_sum = 0.0
    sos_pd_w_sum = 0.0

    for g in games_for_team:
        if not isinstance(g, dict):
            continue
        gw = _get_any(g, "week")
        if not isinstance(gw, int) or gw >= week:
            continue
        if not bool(_get_any(g, "completed")):
            continue
        tp, op = _team_game_points(team, g)
        if tp is None or op is None:
            continue

        opp, opp_elo = _game_opponent_and_elo(team=team, game=g)
        if opp is not None and opp_elo is None and class_map is not None:
            if _is_fbs_team(class_map, opp) == 0:
                opp_elo = FCS_ELO_FLOOR

        # Exponential decay by recency (more recent weeks matter more).
        # delta=1 means last week; delta grows into the past.
        delta_weeks = max(0, (week - 1) - gw)
        if half_life_weeks <= 0:
            w = 1.0
        else:
            w = float(0.5 ** (delta_weeks / half_life_weeks))

        played += 1
        wins += 1 if tp > op else 0
        points_for += tp
        points_against += op
        weight_sum += w
        weight_wins += w * (1.0 if tp > op else 0.0)
        weight_pf += w * float(tp)
        weight_pa += w * float(op)

        if opp_elo is not None and np.isfinite(opp_elo):
            opp_elo_sum += float(opp_elo)
            opp_elo_w_sum += w * float(opp_elo)
            # Strength-of-schedule adjusted point diff: harder opponents boost this.
            # Scale opponent elo so units stay comparable.
            scale = float(opp_elo) / 1500.0
            pd = float(tp - op)
            sos_pd_sum += pd * scale
            sos_pd_w_sum += w * pd * scale

    if played == 0:
        return {
            "games_played": 0.0,
            "win_pct": 0.0,
            "points_for_pg": 0.0,
            "points_against_pg": 0.0,
            "point_diff_pg": 0.0,
            "w_win_pct": 0.0,
            "w_points_for_pg": 0.0,
            "w_points_against_pg": 0.0,
            "w_point_diff_pg": 0.0,
            "opp_elo_avg": 0.0,
            "w_opp_elo_avg": 0.0,
            "sos_point_diff_pg": 0.0,
            "w_sos_point_diff_pg": 0.0,
        }

    win_pct = wins / played
    pf_pg = points_for / played
    pa_pg = points_against / played
    pd_pg = (points_for - points_against) / played

    if weight_sum <= 0:
        w_win_pct = float(win_pct)
        w_pf_pg = float(pf_pg)
        w_pa_pg = float(pa_pg)
    else:
        w_win_pct = float(weight_wins / weight_sum)
        w_pf_pg = float(weight_pf / weight_sum)
        w_pa_pg = float(weight_pa / weight_sum)
    w_pd_pg = float(w_pf_pg - w_pa_pg)
    opp_elo_avg = float(opp_elo_sum / played) if played > 0 else 0.0
    w_opp_elo_avg = float(opp_elo_w_sum / weight_sum) if weight_sum > 0 else opp_elo_avg
    sos_pd_pg = float(sos_pd_sum / played) if played > 0 else 0.0
    w_sos_pd_pg = float(sos_pd_w_sum / weight_sum) if weight_sum > 0 else sos_pd_pg

    return {
        "games_played": float(played),
        "win_pct": float(win_pct),
        "points_for_pg": float(pf_pg),
        "points_against_pg": float(pa_pg),
        "point_diff_pg": float(pd_pg),
        "w_win_pct": float(w_win_pct),
        "w_points_for_pg": float(w_pf_pg),
        "w_points_against_pg": float(w_pa_pg),
        "w_point_diff_pg": float(w_pd_pg),
        "opp_elo_avg": float(opp_elo_avg),
        "w_opp_elo_avg": float(w_opp_elo_avg),
        "sos_point_diff_pg": float(sos_pd_pg),
        "w_sos_point_diff_pg": float(w_sos_pd_pg),
    }


def _fetch_team_games(
    client: CFBDClient,
    *,
    year: int,
    season_type: str,
    team: str,
    use_cache: bool,
    cache: dict[tuple[int, str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    key = (year, season_type, team)
    if key in cache:
        return cache[key]
    items = client.get("/games", {"year": year, "seasonType": season_type, "team": team}, use_cache=use_cache)
    games = items if isinstance(items, list) else []
    cache[key] = games
    return games


def _elo_snapshot(client: CFBDClient, *, year: int, week: int) -> dict[str, dict[str, Any]]:
    # Endpoint is documented in the CFBD ecosystem but not always in the Postman browse page.
    # We keep it optional: if it fails, caller can proceed without Elo.
    items = client.get("/ratings/elo", {"year": year, "week": week}, use_cache=True)
    if not isinstance(items, list):
        return {}
    return _index_by_team(items, team_field="team")


def build_dataset(
    *,
    year_from: int,
    year_to: int,
    team: str = "Georgia Tech",
    include_postseason: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    client = CFBDClient()
    class_map = _team_classification_map(client, use_cache=use_cache)

    original_team_input = team
    resolved_team, team_candidates = resolve_team_name(client, team, use_cache=use_cache)
    team = resolved_team

    rows: list[dict[str, Any]] = []
    debug_counts: dict[str, int] = {
        "games_fetched": 0,
        "games_with_score": 0,
        "skipped_team_mismatch": 0,
        "skipped_missing_opponent": 0,
        "skipped_missing_score": 0,
        "rows_added": 0,
    }

    elo_cache: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    team_games_cache: dict[tuple[int, str, str], list[dict[str, Any]]] = {}

    for year in range(year_from, year_to + 1):
        # Season-level features (constant across all games in a season).
        talent_by_team: dict[str, dict[str, Any]] = {}
        returning_by_team: dict[str, dict[str, Any]] = {}
        recruiting_by_team: dict[str, dict[str, Any]] = {}

        try:
            talent_by_team = _index_by_team(
                client.get("/talent", {"year": year}, use_cache=use_cache),
                team_field="school",
            )
        except Exception:
            talent_by_team = {}

        try:
            returning_by_team = _index_by_team(
                client.get("/player/returning", {"year": year}, use_cache=use_cache),
                team_field="team",
            )
        except Exception:
            returning_by_team = {}

        try:
            recruiting_by_team = _index_by_team(
                client.get("/recruiting/teams", {"year": year}, use_cache=use_cache),
                team_field="team",
            )
        except Exception:
            recruiting_by_team = {}

        season_types = ["regular"] + (["postseason"] if include_postseason else [])

        games: list[dict[str, Any]] = []
        for season_type in season_types:
            items = client.get(
                "/games",
                {"year": year, "seasonType": season_type, "team": team},
                use_cache=use_cache,
            )
            if isinstance(items, list):
                games.extend(items)
                debug_counts["games_fetched"] += len(items)

        for game in games:
            game_id = _get_any(game, "id")
            week = _get_any(game, "week")
            season_type = _get_any(game, "season_type", "seasonType")

            opponent, is_home, is_neutral = _opponent_and_flags(team, game)
            if opponent is None or is_home is None or is_neutral is None:
                debug_counts["skipped_missing_opponent"] += 1
                continue

            gt_win = _winner_label(team, game)
            if gt_win is None:
                # Could be missing scores or a weird team naming mismatch.
                home = _get_any(game, "home_team", "homeTeam")
                away = _get_any(game, "away_team", "awayTeam")
                if team != home and team != away:
                    debug_counts["skipped_team_mismatch"] += 1
                else:
                    debug_counts["skipped_missing_score"] += 1
                continue
            debug_counts["games_with_score"] += 1

            # Season-to-date form features (pregame): computed from games before this week.
            form_season_type = "regular" if str(season_type).lower() == "regular" else "postseason"
            gt_games = _fetch_team_games(
                client,
                year=year,
                season_type=form_season_type,
                team=team,
                use_cache=use_cache,
                cache=team_games_cache,
            )
            opp_games = _fetch_team_games(
                client,
                year=year,
                season_type=form_season_type,
                team=opponent,
                use_cache=use_cache,
                cache=team_games_cache,
            )
            current_week = week if isinstance(week, int) else 99
            gt_form = _compute_prior_team_form(games_for_team=gt_games, team=team, week=current_week, class_map=class_map)
            opp_form = _compute_prior_team_form(
                games_for_team=opp_games, team=opponent, week=current_week, class_map=class_map
            )
            opp_is_fbs = _is_fbs_team(class_map, opponent)
            opp_classification = class_map.get(opponent)

            # Prefer pregame Elo embedded directly in the games response (pregame, per-game).
            elo_diff = None
            home_pregame_elo = _as_float(_get_any(game, "homePregameElo", "home_pregame_elo"))
            away_pregame_elo = _as_float(_get_any(game, "awayPregameElo", "away_pregame_elo"))
            home_name = _get_any(game, "home_team", "homeTeam")
            away_name = _get_any(game, "away_team", "awayTeam")
            if (
                isinstance(home_name, str)
                and isinstance(away_name, str)
                and home_pregame_elo is not None
                and away_pregame_elo is not None
            ):
                if team == home_name and opponent == away_name:
                    elo_diff = home_pregame_elo - away_pregame_elo
                elif team == away_name and opponent == home_name:
                    elo_diff = away_pregame_elo - home_pregame_elo

            # FCS/Non-FBS floor: if opponent isn't FBS and Elo is missing, assume a low floor
            # rather than letting the imputer treat it like a median FBS team.
            if elo_diff is None and opp_is_fbs == 0:
                gt_elo = None
                if isinstance(home_name, str) and team == home_name:
                    gt_elo = home_pregame_elo
                if isinstance(away_name, str) and team == away_name:
                    gt_elo = away_pregame_elo
                if gt_elo is not None:
                    elo_diff = float(gt_elo - FCS_ELO_FLOOR)

            # Fallback: weekly Elo snapshots (optional; may be missing depending on plan/access).
            if elo_diff is None and isinstance(week, int) and week >= 1:
                asof_week = max(1, week - 1)
                cache_key = (year, asof_week)
                if cache_key not in elo_cache:
                    try:
                        elo_cache[cache_key] = _elo_snapshot(client, year=year, week=asof_week)
                    except Exception:
                        elo_cache[cache_key] = {}
                snap = elo_cache[cache_key]
                gt_elo = _as_float((snap.get(team) or {}).get("elo"))
                opp_elo = _as_float((snap.get(opponent) or {}).get("elo"))
                if gt_elo is not None and opp_elo is not None:
                    elo_diff = gt_elo - opp_elo
                elif opp_is_fbs == 0 and gt_elo is not None:
                    elo_diff = float(gt_elo - FCS_ELO_FLOOR)

            gt_talent = _as_float((talent_by_team.get(team) or {}).get("talent"))
            opp_talent = _as_float((talent_by_team.get(opponent) or {}).get("talent"))
            if opp_talent is None and opp_is_fbs == 0 and gt_talent is not None:
                opp_talent = NONFBS_TALENT_FLOOR
            talent_diff = (gt_talent - opp_talent) if gt_talent is not None and opp_talent is not None else None

            # Returning production fields vary by era/plan; prefer a robust "total" if present.
            def _returning_total(v: dict[str, Any]) -> float | None:
                for key in ("total", "totalPpa", "total_ppa", "total_returning"):
                    out = _as_float(v.get(key))
                    if out is not None:
                        return out
                out = _as_float(v.get("ppa"))
                return out

            gt_ret = _returning_total(returning_by_team.get(team) or {})
            opp_ret = _returning_total(returning_by_team.get(opponent) or {})
            returning_diff = (gt_ret - opp_ret) if gt_ret is not None and opp_ret is not None else None

            gt_rec_points = _as_float((recruiting_by_team.get(team) or {}).get("points"))
            opp_rec_points = _as_float((recruiting_by_team.get(opponent) or {}).get("points"))
            if opp_rec_points is None and opp_is_fbs == 0 and gt_rec_points is not None:
                opp_rec_points = NONFBS_RECRUIT_POINTS_FLOOR
            recruit_points_diff = (
                gt_rec_points - opp_rec_points
                if gt_rec_points is not None and opp_rec_points is not None
                else None
            )

            gt_rec_rank = _as_float((recruiting_by_team.get(team) or {}).get("rank"))
            opp_rec_rank = _as_float((recruiting_by_team.get(opponent) or {}).get("rank"))
            if opp_rec_rank is None and opp_is_fbs == 0 and gt_rec_rank is not None:
                opp_rec_rank = NONFBS_RECRUIT_RANK_FLOOR
            # Lower rank is better, so (opp - gt) makes positive => GT better.
            recruit_rank_diff = (
                opp_rec_rank - gt_rec_rank if gt_rec_rank is not None and opp_rec_rank is not None else None
            )

            rows.append(
                {
                    "game_id": game_id,
                    "year": year,
                    "week": week,
                    "season_type": season_type,
                    "opponent": opponent,
                    "opp_classification": opp_classification,
                    "opp_is_fbs": float(opp_is_fbs),
                    "is_home": int(is_home),
                    "neutral_site": int(is_neutral),
                    "gt_win": int(gt_win),
                    "gt_points": float(_team_game_points(team, game)[0] or 0),
                    "opp_points": float(_team_game_points(team, game)[1] or 0),
                    "margin": float((_team_game_points(team, game)[0] or 0) - (_team_game_points(team, game)[1] or 0)),
                    "gt_games_played": gt_form["games_played"],
                    "opp_games_played": opp_form["games_played"],
                    "games_played_diff": gt_form["games_played"] - opp_form["games_played"],
                    "gt_win_pct": gt_form["win_pct"],
                    "opp_win_pct": opp_form["win_pct"],
                    "win_pct_diff": gt_form["win_pct"] - opp_form["win_pct"],
                    "gt_point_diff_pg": gt_form["point_diff_pg"],
                    "opp_point_diff_pg": opp_form["point_diff_pg"],
                    "point_diff_pg_diff": gt_form["point_diff_pg"] - opp_form["point_diff_pg"],
                    "gt_points_for_pg": gt_form["points_for_pg"],
                    "opp_points_for_pg": opp_form["points_for_pg"],
                    "points_for_pg_diff": gt_form["points_for_pg"] - opp_form["points_for_pg"],
                    "gt_points_against_pg": gt_form["points_against_pg"],
                    "opp_points_against_pg": opp_form["points_against_pg"],
                    "points_against_pg_diff": gt_form["points_against_pg"] - opp_form["points_against_pg"],
                    "gt_w_win_pct": gt_form["w_win_pct"],
                    "opp_w_win_pct": opp_form["w_win_pct"],
                    "w_win_pct_diff": gt_form["w_win_pct"] - opp_form["w_win_pct"],
                    "gt_w_point_diff_pg": gt_form["w_point_diff_pg"],
                    "opp_w_point_diff_pg": opp_form["w_point_diff_pg"],
                    "w_point_diff_pg_diff": gt_form["w_point_diff_pg"] - opp_form["w_point_diff_pg"],
                    "gt_w_points_for_pg": gt_form["w_points_for_pg"],
                    "opp_w_points_for_pg": opp_form["w_points_for_pg"],
                    "w_points_for_pg_diff": gt_form["w_points_for_pg"] - opp_form["w_points_for_pg"],
                    "gt_w_points_against_pg": gt_form["w_points_against_pg"],
                    "opp_w_points_against_pg": opp_form["w_points_against_pg"],
                    "w_points_against_pg_diff": gt_form["w_points_against_pg"] - opp_form["w_points_against_pg"],
                    "gt_opp_elo_avg": gt_form["opp_elo_avg"],
                    "opp_opp_elo_avg": opp_form["opp_elo_avg"],
                    "opp_elo_avg_diff": gt_form["opp_elo_avg"] - opp_form["opp_elo_avg"],
                    "gt_w_opp_elo_avg": gt_form["w_opp_elo_avg"],
                    "opp_w_opp_elo_avg": opp_form["w_opp_elo_avg"],
                    "w_opp_elo_avg_diff": gt_form["w_opp_elo_avg"] - opp_form["w_opp_elo_avg"],
                    "gt_sos_point_diff_pg": gt_form["sos_point_diff_pg"],
                    "opp_sos_point_diff_pg": opp_form["sos_point_diff_pg"],
                    "sos_point_diff_pg_diff": gt_form["sos_point_diff_pg"] - opp_form["sos_point_diff_pg"],
                    "gt_w_sos_point_diff_pg": gt_form["w_sos_point_diff_pg"],
                    "opp_w_sos_point_diff_pg": opp_form["w_sos_point_diff_pg"],
                    "w_sos_point_diff_pg_diff": gt_form["w_sos_point_diff_pg"] - opp_form["w_sos_point_diff_pg"],
                    "elo_diff": elo_diff,
                    "talent_diff": talent_diff,
                    "returning_diff": returning_diff,
                    "recruit_points_diff": recruit_points_diff,
                    "recruit_rank_diff": recruit_rank_diff,
                }
            )
            debug_counts["rows_added"] += 1

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["year", "week", "game_id"], kind="stable")
    else:
        # Helpful diagnostics when users report "0 rows".
        diag_path = Path("data_processed") / "build_diagnostics.json"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag_path.write_text(
            json.dumps(
                {
                    "team_input": original_team_input,
                    "team_resolved": resolved_team,
                    "team_candidates": team_candidates,
                    "year_from": year_from,
                    "year_to": year_to,
                    **debug_counts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return df


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

    requested_feature_cols = [
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

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(float if target == "margin" else int)
    X_val = val_df[feature_cols]
    y_val = val_df[target_col].astype(float if target == "margin" else int)
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(float if target == "margin" else int)

    # Recency weighting: newer seasons matter more than older ones.
    # Half-life means weights halve every N years into the past.
    if half_life_years <= 0:
        raise RuntimeError("--half-life-years must be > 0")
    max_train_year = int(train_df["year"].max())
    year_delta = (max_train_year - train_df["year"]).astype(float)
    sample_weight = (0.5 ** (year_delta / half_life_years)).to_numpy()

    numeric_features = feature_cols
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
                numeric_features,
            )
        ],
        remainder="drop",
    )

    if target == "win":
        model: Any = LogisticRegression(max_iter=2000, solver="lbfgs")
    else:
        model = Ridge(alpha=1.0)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train, model__sample_weight=sample_weight)

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

    if target == "win":
        p_val_raw = pipe.predict_proba(X_val)[:, 1]
        p_test_raw = pipe.predict_proba(X_test)[:, 1]
    else:
        # For margin model, "raw" is predicted margin.
        p_val_raw = pipe.predict(X_val)
        p_test_raw = pipe.predict(X_test)

    # Calibration (win-prob model only). Margin model produces probability via a normal CDF.
    #
    # Isotonic is flexible but can collapse into a few flat steps on small datasets.
    # Fit both isotonic and a simple sigmoid (Platt-style) calibrator, then pick the
    # one that improves validation Brier score the most.
    y_val_np = y_val.to_numpy()
    y_test_np = y_test.to_numpy()

    if target == "margin":
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

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_raw, y_val_np)
    p_val_iso = iso.transform(p_val_raw)
    p_test_iso = iso.transform(p_test_raw)

    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    sig = LogisticRegression(max_iter=2000, solver="lbfgs")
    sig.fit(_logit(p_val_raw).reshape(-1, 1), y_val_np)
    p_val_sig = sig.predict_proba(_logit(p_val_raw).reshape(-1, 1))[:, 1]
    p_test_sig = sig.predict_proba(_logit(p_test_raw).reshape(-1, 1))[:, 1]

    brier_raw = float(brier_score_loss(y_val_np, p_val_raw))
    brier_iso = float(brier_score_loss(y_val_np, p_val_iso))
    brier_sig = float(brier_score_loss(y_val_np, p_val_sig))

    calibration = calibration.strip().lower()
    if calibration not in ("auto", "isotonic", "sigmoid", "none"):
        raise RuntimeError("calibration must be one of: auto, isotonic, sigmoid, none")

    if calibration == "none":
        calibrator = None
        p_val_cal = p_val_raw
        p_test_cal = p_test_raw
    elif calibration == "isotonic":
        calibrator = {"type": "isotonic", "model": iso}
        p_val_cal = p_val_iso
        p_test_cal = p_test_iso
    elif calibration == "sigmoid":
        calibrator = {"type": "sigmoid_logit", "model": sig}
        p_val_cal = p_val_sig
        p_test_cal = p_test_sig
    else:
        # auto: choose best Brier; tie-break toward sigmoid if isotonic is too "steppy".
        val_unique_iso = int(np.unique(np.round(p_val_iso, 6)).size)
        if brier_sig <= brier_iso and brier_sig <= brier_raw:
            calibrator = {"type": "sigmoid_logit", "model": sig}
            p_val_cal = p_val_sig
            p_test_cal = p_test_sig
        elif brier_iso <= brier_raw:
            # If isotonic collapses to a small number of outputs, prefer sigmoid when close.
            if val_unique_iso <= 6 and (brier_sig - brier_iso) <= 0.01:
                calibrator = {"type": "sigmoid_logit", "model": sig}
                p_val_cal = p_val_sig
                p_test_cal = p_test_sig
            else:
                calibrator = {"type": "isotonic", "model": iso}
                p_val_cal = p_val_iso
                p_test_cal = p_test_iso
        else:
            calibrator = None
            p_val_cal = p_val_raw
            p_test_cal = p_test_raw

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


def _fetch_matchup_features(
    *,
    team: str,
    year: int,
    week: int,
    opponent: str,
    home: str,
) -> pd.DataFrame:
    client = CFBDClient()
    class_map = _team_classification_map(client, use_cache=True)

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
    try:
        snap = _elo_snapshot(client, year=year, week=max(1, week - 1))
        gt_elo = _as_float((snap.get(team) or {}).get("elo"))
        opp_elo = _as_float((snap.get(opponent) or {}).get("elo"))
        if gt_elo is not None and opp_elo is not None:
            elo_diff = gt_elo - opp_elo
        elif opp_is_fbs == 0 and gt_elo is not None:
            elo_diff = float(gt_elo - FCS_ELO_FLOOR)
    except Exception:
        elo_diff = None

    # Season-level
    def _try_index(endpoint: str, params: dict[str, Any], team_field: str) -> dict[str, dict[str, Any]]:
        try:
            items = client.get(endpoint, params, use_cache=True)
            if isinstance(items, list):
                return _index_by_team(items, team_field=team_field)
        except Exception:
            return {}
        return {}

    talent = _try_index("/talent", {"year": year}, team_field="school")
    returning = _try_index("/player/returning", {"year": year}, team_field="team")
    recruiting = _try_index("/recruiting/teams", {"year": year}, team_field="team")

    gt_talent = _as_float((talent.get(team) or {}).get("talent"))
    opp_talent = _as_float((talent.get(opponent) or {}).get("talent"))
    if opp_talent is None and opp_is_fbs == 0 and gt_talent is not None:
        opp_talent = NONFBS_TALENT_FLOOR
    talent_diff = (gt_talent - opp_talent) if gt_talent is not None and opp_talent is not None else None

    def _returning_total(v: dict[str, Any]) -> float | None:
        for key in ("total", "totalPpa", "total_ppa", "total_returning"):
            out = _as_float(v.get(key))
            if out is not None:
                return out
        out = _as_float(v.get("ppa"))
        return out

    gt_ret = _returning_total(returning.get(team) or {})
    opp_ret = _returning_total(returning.get(opponent) or {})
    returning_diff = (gt_ret - opp_ret) if gt_ret is not None and opp_ret is not None else None

    gt_rec_points = _as_float((recruiting.get(team) or {}).get("points"))
    opp_rec_points = _as_float((recruiting.get(opponent) or {}).get("points"))
    if opp_rec_points is None and opp_is_fbs == 0 and gt_rec_points is not None:
        opp_rec_points = NONFBS_RECRUIT_POINTS_FLOOR
    recruit_points_diff = (
        gt_rec_points - opp_rec_points if gt_rec_points is not None and opp_rec_points is not None else None
    )

    gt_rec_rank = _as_float((recruiting.get(team) or {}).get("rank"))
    opp_rec_rank = _as_float((recruiting.get(opponent) or {}).get("rank"))
    if opp_rec_rank is None and opp_is_fbs == 0 and gt_rec_rank is not None:
        opp_rec_rank = NONFBS_RECRUIT_RANK_FLOOR
    recruit_rank_diff = opp_rec_rank - gt_rec_rank if gt_rec_rank is not None and opp_rec_rank is not None else None

    return pd.DataFrame(
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


def _load_model_bundle(model_path: Path) -> tuple[Any, Any | None, list[str] | None]:
    pipe = joblib.load(model_path)
    calibrator = None
    calibrator_path = model_path.parent / "gt_winprob_calibrator.joblib"
    if calibrator_path.exists():
        calibrator = joblib.load(calibrator_path)
    else:
        # Legacy fallback.
        legacy_path = model_path.parent / "gt_winprob_calibrator_isotonic.joblib"
        calibrator = joblib.load(legacy_path) if legacy_path.exists() else None

    feature_path = model_path.parent / "feature_columns.json"
    feature_cols: list[str] | None = None
    if feature_path.exists():
        try:
            feature_cols = json.loads(feature_path.read_text(encoding="utf-8"))
        except Exception:
            feature_cols = None
    return pipe, calibrator, feature_cols


def _load_model_meta(model_path: Path) -> dict[str, Any]:
    meta_path = model_path.parent / "model_meta.json"
    if not meta_path.exists():
        # Best-effort guess for older artifacts.
        name = model_path.name.lower()
        if "margin" in name:
            return {"target": "margin"}
        return {"target": "win"}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {"target": "win"}


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

        X_full = _fetch_matchup_features(team=team, year=year, week=week, opponent=opponent, home=home_flag)
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
            }
        )

    df = pd.DataFrame(out_rows)
    if not df.empty:
        df = df.sort_values(["week", "start_date", "opponent"], kind="stable")
    return df


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="winprob.py", description="Train/predict Georgia Tech win probability from CFBD.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-dataset", help="Download CFBD data and build data/model_dataset.csv")
    b.add_argument("--from-year", type=int, default=2014)
    b.add_argument("--to-year", type=int, default=2025)
    b.add_argument("--team", type=str, default="Georgia Tech")
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

    args = p.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "build-dataset":
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
        out: dict[str, Any] = {
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

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
