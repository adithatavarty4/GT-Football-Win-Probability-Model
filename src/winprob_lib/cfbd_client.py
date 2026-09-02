from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

CFBD_BASE_URL = "https://api.collegefootballdata.com"


def _stable_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: params[k] for k in sorted(params.keys()) if params[k] is not None}


def _cache_key(endpoint: str, params: dict[str, Any]) -> str:
    stable = json.dumps({"endpoint": endpoint, "params": _stable_params(params)}, sort_keys=True, separators=(",", ":"))
    # Stable across runs (unlike Python's built-in hash()).
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]


def _norm_team_name(name: str) -> str:
    return " ".join(name.strip().casefold().split())


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
        # Don't cache an empty response: for endpoints that fill in over time (talent,
        # recruiting, returning production, in-season Elo), an empty list usually means
        # "not published yet", not "will always be empty". Caching it here has no expiry,
        # so it would permanently and silently hide real data that shows up later.
        if use_cache and data:
            cache_path.write_text(json.dumps(data), encoding="utf-8")
        return data


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


def _team_classification_map(client: CFBDClient, *, year: int, use_cache: bool = True) -> dict[str, str]:
    """
    Map CFBD `school` -> `classification` (fbs/fcs/...) as of `year`.

    Classification is year-scoped deliberately: several teams moved FCS -> FBS during
    2014-2025 (e.g. Jacksonville State, FCS through 2022 and FBS from 2023). A single
    present-day snapshot applied to all years would mislabel their older games.
    """
    items = client.get("/teams", {"year": year}, use_cache=use_cache)
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
