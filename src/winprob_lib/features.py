from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cfbd_client import CFBDClient, _is_fbs_team, _team_classification_map, resolve_team_name
from .util import _as_float, _get_any, _index_by_team

FCS_ELO_FLOOR = 800.0
NONFBS_RECRUIT_POINTS_FLOOR = 0.0
NONFBS_RECRUIT_RANK_FLOOR = 999.0
NONFBS_TALENT_FLOOR = 0.0


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


# Mean-reversion carryover for a team's Elo entering a season with no current-season Elo
# published yet (e.g. predicting a game before/early in a season). Fit by linear regression
# of "team's actual week-1 pregame Elo in year Y" on "team's final pregame Elo of year Y-1"
# across 1,433 team-seasons (2015-2025) from this project's own cached CFBD data:
# R^2 = 0.97, reverting toward a long-run mean of ~1490 (close to the conventional 1500
# Elo baseline). Last season's ending rating is a strong predictor of where a team actually
# starts the next one, but roster turnover pulls it partway back toward average.
ELO_CARRYOVER_SLOPE = 0.6712
ELO_CARRYOVER_INTERCEPT = 489.85


def _team_final_elo(client: CFBDClient, *, team: str, year: int, use_cache: bool = True) -> float | None:
    """
    Best-available proxy for a team's Elo at the end of `year`: the pregame Elo of their
    chronologically last game that season (their postseason game if they played one, else
    their last regular-season game).
    """
    try:
        reg_games = client.get("/games", {"year": year, "seasonType": "regular", "team": team}, use_cache=use_cache)
    except Exception:
        reg_games = []
    try:
        post_games = client.get("/games", {"year": year, "seasonType": "postseason", "team": team}, use_cache=use_cache)
    except Exception:
        post_games = []

    def _team_pregame_elo(game: dict[str, Any]) -> float | None:
        home = _get_any(game, "home_team", "homeTeam")
        away = _get_any(game, "away_team", "awayTeam")
        if team == home:
            return _as_float(_get_any(game, "homePregameElo", "home_pregame_elo"))
        if team == away:
            return _as_float(_get_any(game, "awayPregameElo", "away_pregame_elo"))
        return None

    pool = post_games if isinstance(post_games, list) and post_games else (reg_games if isinstance(reg_games, list) else [])
    entries: list[tuple[int, float]] = []
    for g in pool:
        if not isinstance(g, dict):
            continue
        wk = _get_any(g, "week")
        elo = _team_pregame_elo(g)
        if isinstance(wk, int) and elo is not None:
            entries.append((wk, elo))
    if not entries:
        return None
    entries.sort(key=lambda e: e[0])
    return float(entries[-1][1])


def _preseason_elo_estimate(client: CFBDClient, *, team: str, year: int, use_cache: bool = True) -> float | None:
    """
    Carryover Elo estimate for a team entering `year`, for use only when no current-season
    Elo data exists yet. Returns None if the prior season's final Elo can't be found either
    (e.g. the team wasn't FBS, or it's the first year in the data).
    """
    final_prior = _team_final_elo(client, team=team, year=year - 1, use_cache=use_cache)
    if final_prior is None:
        return None
    return ELO_CARRYOVER_INTERCEPT + ELO_CARRYOVER_SLOPE * final_prior


def build_dataset(
    *,
    year_from: int,
    year_to: int,
    team: str = "Georgia Tech",
    include_postseason: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    client = CFBDClient()

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
        class_map = _team_classification_map(client, year=year, use_cache=use_cache)

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

            # Season-to-date form features (pregame): computed from regular-season games
            # before this week. Postseason week numbers reset to 1, so postseason games
            # always use the (by then complete) regular season as their prior-form history
            # rather than the postseason games list, which would otherwise never satisfy
            # "week < current_week" and silently zero out the form features.
            is_regular = str(season_type).lower() == "regular"
            current_week = week if isinstance(week, int) else 99
            form_week = current_week if is_regular else 9999
            gt_games = _fetch_team_games(
                client,
                year=year,
                season_type="regular",
                team=team,
                use_cache=use_cache,
                cache=team_games_cache,
            )
            opp_games = _fetch_team_games(
                client,
                year=year,
                season_type="regular",
                team=opponent,
                use_cache=use_cache,
                cache=team_games_cache,
            )
            gt_form = _compute_prior_team_form(games_for_team=gt_games, team=team, week=form_week, class_map=class_map)
            opp_form = _compute_prior_team_form(
                games_for_team=opp_games, team=opponent, week=form_week, class_map=class_map
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

            # Fallback: no current-season Elo published yet at all (e.g. predicting a game
            # before/early in a season) -- use each team's prior-season-final Elo, regressed
            # toward the mean via the empirically-fit carryover estimate.
            if elo_diff is None:
                gt_elo_carry = _preseason_elo_estimate(client, team=team, year=year, use_cache=use_cache)
                opp_elo_carry = _preseason_elo_estimate(client, team=opponent, year=year, use_cache=use_cache)
                if gt_elo_carry is not None and opp_elo_carry is not None:
                    elo_diff = gt_elo_carry - opp_elo_carry
                elif opp_is_fbs == 0 and gt_elo_carry is not None:
                    elo_diff = float(gt_elo_carry - FCS_ELO_FLOOR)

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
                    "team": team,
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


def build_dataset_all_fbs(
    *,
    year_from: int,
    year_to: int,
    include_postseason: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Pools every FBS team's games (both home and away perspective) into one team-agnostic
    training set, using the same relative-diff feature schema as `build_dataset()`.

    Unlike `build_dataset()`, this fetches `/games` once per year league-wide instead of
    once per team, and season-level endpoints (`/talent`, `/player/returning`,
    `/recruiting/teams`) already cover every team in a single call - so this is *fewer*
    CFBD requests than looping `build_dataset()` per team, not more.

    A non-FBS team never becomes the focal `team` for a row (there's no reason to train
    the model to predict an FCS team's win probability), but non-FBS opponents are kept,
    using the same floor-fallback constants as `build_dataset()`.
    """
    client = CFBDClient()

    def _returning_total(v: dict[str, Any]) -> float | None:
        for key in ("total", "totalPpa", "total_ppa", "total_returning"):
            out = _as_float(v.get(key))
            if out is not None:
                return out
        return _as_float(v.get("ppa"))

    rows: list[dict[str, Any]] = []
    elo_cache: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}

    for year in range(year_from, year_to + 1):
        class_map = _team_classification_map(client, year=year, use_cache=use_cache)
        try:
            talent_by_team = _index_by_team(client.get("/talent", {"year": year}, use_cache=use_cache), team_field="school")
        except Exception:
            talent_by_team = {}
        try:
            returning_by_team = _index_by_team(
                client.get("/player/returning", {"year": year}, use_cache=use_cache), team_field="team"
            )
        except Exception:
            returning_by_team = {}
        try:
            recruiting_by_team = _index_by_team(
                client.get("/recruiting/teams", {"year": year}, use_cache=use_cache), team_field="team"
            )
        except Exception:
            recruiting_by_team = {}

        regular_games = client.get("/games", {"year": year, "seasonType": "regular"}, use_cache=use_cache)
        regular_games = regular_games if isinstance(regular_games, list) else []

        # Every team's own regular-season game list, built once from the league-wide fetch
        # above instead of one /games call per team.
        games_by_team: dict[str, list[dict[str, Any]]] = {}
        for g in regular_games:
            home = _get_any(g, "home_team", "homeTeam")
            away = _get_any(g, "away_team", "awayTeam")
            if isinstance(home, str):
                games_by_team.setdefault(home, []).append(g)
            if isinstance(away, str):
                games_by_team.setdefault(away, []).append(g)

        all_games = list(regular_games)
        if include_postseason:
            post_games = client.get("/games", {"year": year, "seasonType": "postseason"}, use_cache=use_cache)
            all_games.extend(post_games if isinstance(post_games, list) else [])

        for game in all_games:
            game_id = _get_any(game, "id")
            week = _get_any(game, "week")
            season_type = _get_any(game, "season_type", "seasonType")
            home = _get_any(game, "home_team", "homeTeam")
            away = _get_any(game, "away_team", "awayTeam")
            if not isinstance(home, str) or not isinstance(away, str):
                continue

            focal_sides: list[tuple[str, str]] = []
            if _is_fbs_team(class_map, home) == 1:
                focal_sides.append((home, away))
            if _is_fbs_team(class_map, away) == 1:
                focal_sides.append((away, home))

            for team, opponent in focal_sides:
                gt_win = _winner_label(team, game)
                if gt_win is None:
                    continue
                _opp_check, is_home, is_neutral = _opponent_and_flags(team, game)
                if is_home is None or is_neutral is None:
                    continue

                # Same postseason fix as build_dataset(): bowl weeks reset to 1, so prior
                # form always comes from the (by then complete) regular season, not the
                # postseason games list.
                is_regular = str(season_type).lower() == "regular"
                current_week = week if isinstance(week, int) else 99
                form_week = current_week if is_regular else 9999

                gt_form = _compute_prior_team_form(
                    games_for_team=games_by_team.get(team, []), team=team, week=form_week, class_map=class_map
                )
                opp_form = _compute_prior_team_form(
                    games_for_team=games_by_team.get(opponent, []), team=opponent, week=form_week, class_map=class_map
                )
                opp_is_fbs = _is_fbs_team(class_map, opponent)
                opp_classification = class_map.get(opponent)

                elo_diff = None
                home_pregame_elo = _as_float(_get_any(game, "homePregameElo", "home_pregame_elo"))
                away_pregame_elo = _as_float(_get_any(game, "awayPregameElo", "away_pregame_elo"))
                if home_pregame_elo is not None and away_pregame_elo is not None:
                    if team == home:
                        elo_diff = home_pregame_elo - away_pregame_elo
                    elif team == away:
                        elo_diff = away_pregame_elo - home_pregame_elo

                if elo_diff is None and opp_is_fbs == 0:
                    team_elo = home_pregame_elo if team == home else (away_pregame_elo if team == away else None)
                    if team_elo is not None:
                        elo_diff = float(team_elo - FCS_ELO_FLOOR)

                if elo_diff is None and isinstance(week, int) and week >= 1:
                    asof_week = max(1, week - 1)
                    cache_key = (year, asof_week)
                    if cache_key not in elo_cache:
                        try:
                            elo_cache[cache_key] = _elo_snapshot(client, year=year, week=asof_week)
                        except Exception:
                            elo_cache[cache_key] = {}
                    snap = elo_cache[cache_key]
                    team_elo = _as_float((snap.get(team) or {}).get("elo"))
                    opp_elo = _as_float((snap.get(opponent) or {}).get("elo"))
                    if team_elo is not None and opp_elo is not None:
                        elo_diff = team_elo - opp_elo
                    elif opp_is_fbs == 0 and team_elo is not None:
                        elo_diff = float(team_elo - FCS_ELO_FLOOR)

                # Fallback: no current-season Elo published yet at all -- use each team's
                # prior-season-final Elo, regressed toward the mean.
                if elo_diff is None:
                    team_elo_carry = _preseason_elo_estimate(client, team=team, year=year, use_cache=use_cache)
                    opp_elo_carry = _preseason_elo_estimate(client, team=opponent, year=year, use_cache=use_cache)
                    if team_elo_carry is not None and opp_elo_carry is not None:
                        elo_diff = team_elo_carry - opp_elo_carry
                    elif opp_is_fbs == 0 and team_elo_carry is not None:
                        elo_diff = float(team_elo_carry - FCS_ELO_FLOOR)

                gt_talent = _as_float((talent_by_team.get(team) or {}).get("talent"))
                opp_talent = _as_float((talent_by_team.get(opponent) or {}).get("talent"))
                if opp_talent is None and opp_is_fbs == 0 and gt_talent is not None:
                    opp_talent = NONFBS_TALENT_FLOOR
                talent_diff = (gt_talent - opp_talent) if gt_talent is not None and opp_talent is not None else None

                gt_ret = _returning_total(returning_by_team.get(team) or {})
                opp_ret = _returning_total(returning_by_team.get(opponent) or {})
                returning_diff = (gt_ret - opp_ret) if gt_ret is not None and opp_ret is not None else None

                gt_rec_points = _as_float((recruiting_by_team.get(team) or {}).get("points"))
                opp_rec_points = _as_float((recruiting_by_team.get(opponent) or {}).get("points"))
                if opp_rec_points is None and opp_is_fbs == 0 and gt_rec_points is not None:
                    opp_rec_points = NONFBS_RECRUIT_POINTS_FLOOR
                recruit_points_diff = (
                    gt_rec_points - opp_rec_points if gt_rec_points is not None and opp_rec_points is not None else None
                )

                gt_rec_rank = _as_float((recruiting_by_team.get(team) or {}).get("rank"))
                opp_rec_rank = _as_float((recruiting_by_team.get(opponent) or {}).get("rank"))
                if opp_rec_rank is None and opp_is_fbs == 0 and gt_rec_rank is not None:
                    opp_rec_rank = NONFBS_RECRUIT_RANK_FLOOR
                recruit_rank_diff = (
                    opp_rec_rank - gt_rec_rank if gt_rec_rank is not None and opp_rec_rank is not None else None
                )

                team_pts, opp_pts = _team_game_points(team, game)

                rows.append(
                    {
                        "game_id": game_id,
                        "team": team,
                        "year": year,
                        "week": week,
                        "season_type": season_type,
                        "opponent": opponent,
                        "opp_classification": opp_classification,
                        "opp_is_fbs": float(opp_is_fbs),
                        "is_home": int(is_home),
                        "neutral_site": int(is_neutral),
                        "gt_win": int(gt_win),
                        "gt_points": float(team_pts or 0),
                        "opp_points": float(opp_pts or 0),
                        "margin": float((team_pts or 0) - (opp_pts or 0)),
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

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["year", "week", "game_id", "team"], kind="stable")
    return df
