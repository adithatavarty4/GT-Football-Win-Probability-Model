from __future__ import annotations

from typing import Any

import numpy as np


def _get_any(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


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
