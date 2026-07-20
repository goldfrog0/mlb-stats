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

from mlb_stats.api import (
    find_division,
    find_player,
    find_team,
    get_division_standings,
    get_game_log,
    get_game_pitches,
    get_league_team_stats,
    get_primary_position,
    get_team_schedule,
    search_players,
)
from mlb_stats.plots import (
    build_pitch_dataframe,
    build_standings_dataframe,
    build_stat_dataframe,
    build_team_win_dataframe,
    add_rolling_stat,
    compute_game_value,
    filter_splits_by_date,
)
from mlb_stats.stats import STAT_CONFIGS, get_stat_config
from mlb_stats.war import build_war_approx_dataframe, league_fip, league_woba, position_adjustment

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


def _load_stat_dataframe(name: str, season: int, stat_key: str, window: int) -> tuple[pd.DataFrame, str]:
    config = get_stat_config(stat_key)
    if config["group"] == "team":
        team_id, full_name = find_team(name)
        games = get_team_schedule(team_id, season)
        df = build_team_win_dataframe(games, team_id)
    elif config.get("computation") == "war_approx":
        player_id, full_name = find_player(name)
        splits = get_game_log(player_id, season, config["group"])
        team_totals = get_league_team_stats(season, config["group"])
        if config["group"] == "batting":
            baseline = league_woba(team_totals)
            pos_adj = position_adjustment(get_primary_position(player_id))
        else:
            baseline = league_fip(team_totals)
            pos_adj = 0.0
        df = build_war_approx_dataframe(splits, config["group"], baseline, pos_adj)
    else:
        player_id, full_name = find_player(name)
        splits = get_game_log(player_id, season, config["group"])
        df = build_stat_dataframe(splits, stat_key)
    df = add_rolling_stat(df, stat_key, window)
    df["game"] = compute_game_value(df, stat_key)
    return df, full_name


@app.get("/api/search-players")
def search_players_endpoint(q: str) -> list[dict[str, Any]]:
    """Player-name suggestions for autocomplete. Short queries are
    rejected server-side too (not just by the frontend's debounce/min-
    length guard) since they'd otherwise return dozens of barely-relevant
    matches for no benefit."""
    if len(q.strip()) < 2:
        return []
    return search_players(q.strip())


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


@app.get("/api/pitch-velocities")
def pitch_velocities(
    name: str, season: int | None = None, start: str | None = None, end: str | None = None,
) -> dict[str, Any]:
    """Every pitch a pitcher threw in the date range, one record per
    pitch (date, opponent, pitch_type, velo). The frontend does its own
    per-game grouping, so no rolling/serialization machinery applies --
    dates are already plain ISO strings in the pitch DataFrame."""
    try:
        player_id, full_name = find_player(name)
        splits = get_game_log(player_id, season or _current_season(), "pitching")
        splits = filter_splits_by_date(splits, start, end)
        games = [(s, get_game_pitches(s["game"]["gamePk"])) for s in splits]
        df = build_pitch_dataframe(games, player_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {"name": full_name, "pitches": df.to_dict(orient="records")}


@app.get("/api/standings")
def standings(division: str, season: int | None = None) -> dict[str, Any]:
    try:
        division_id, division_name = find_division(division)
        team_records = get_division_standings(division_id, season or _current_season())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    df = build_standings_dataframe(team_records)
    return {"division": division_name, "teams": df.to_dict(orient="records")}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
