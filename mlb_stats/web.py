"""FastAPI backend for the browser UI. Reuses the same data-prep functions
as the CLI (mlb_stats.api, mlb_stats.stats, mlb_stats.plots) — this module
only adds JSON serialization and HTTP routing on top of them.

Run with: uvicorn mlb_stats.web:app --reload
"""

import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from mlb_stats.api import find_player, get_game_log
from mlb_stats.plots import build_stat_dataframe, add_rolling_stat, compute_game_value
from mlb_stats.stats import STAT_CONFIGS, get_stat_config

app = FastAPI(title="MLB Stats")

STATIC_DIR = Path(__file__).parent / "static"


def _current_season() -> int:
    """Resolved per request rather than at import, since a long-running
    server process would otherwise keep serving last year's default after
    New Year (unlike the CLI, where a module-level constant is fine)."""
    return datetime.date.today().year


def _serialize(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a stat DataFrame into JSON-friendly records (ISO date
    strings, NaN -> null), keeping only the columns the frontend needs."""
    out = df[["date", "opponent", "game", "cumulative", "rolling"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.where(pd.notna(out), None).to_dict(orient="records")


def _load_stat_dataframe(player_name: str, season: int, stat_key: str, window: int) -> tuple[pd.DataFrame, str]:
    config = get_stat_config(stat_key)
    player_id, full_name = find_player(player_name)
    splits = get_game_log(player_id, season, config["group"])
    df = build_stat_dataframe(splits, stat_key)
    df = add_rolling_stat(df, stat_key, window)
    df["game"] = compute_game_value(df, stat_key)
    return df, full_name


@app.get("/api/stats")
def list_stats() -> dict[str, dict[str, str]]:
    """Available stats for populating the frontend's dropdown."""
    return {key: {"label": config["label"], "group": config["group"]} for key, config in STAT_CONFIGS.items()}


@app.get("/api/player")
def player_stat(name: str, stat: str = "era", season: int | None = None, window: int = 5) -> dict[str, Any]:
    try:
        df, full_name = _load_stat_dataframe(name, season or _current_season(), stat, window)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "name": full_name,
        "stat": stat,
        "label": get_stat_config(stat)["label"],
        "data": _serialize(df),
    }


@app.get("/api/compare")
def compare_stat(
    player1: str, player2: str, stat: str = "era", season: int | None = None, window: int = 5
) -> dict[str, Any]:
    try:
        season = season or _current_season()
        df1, name1 = _load_stat_dataframe(player1, season, stat, window)
        df2, name2 = _load_stat_dataframe(player2, season, stat, window)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "stat": stat,
        "label": get_stat_config(stat)["label"],
        "player1": {"name": name1, "data": _serialize(df1)},
        "player2": {"name": name2, "data": _serialize(df2)},
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
