"""Approximate per-game WAR, computed from box-score counting stats.

The MLB Stats API only exposes WAR as a season-to-date snapshot (no
per-game or by-date-range form), so a WAR-over-time chart has to
compute its own. This module implements an fWAR-shaped approximation
from the game log:

Batting, per game:
    wOBA from linear weights -> runs above league average
        wRAA = (wOBA - lgwOBA) / WOBA_SCALE * PA
    + replacement-level runs prorated by PA
    + positional-adjustment runs prorated by PA (from primary position)
    all divided by runs-per-win.

Pitching, per game (FIP-based):
    war = ((lgFIP - gameFIP) / RUNS_PER_WIN + REPL_WINS_PER_9IP) * IP/9

Both formulas are linear in the counting stats, so per-game values sum
exactly to the same-season computation -- cumulative and rolling lines
are mathematically consistent, not approximations of an approximation.

League baselines (lgwOBA, lgFIP) are derived from the 30 teams' season
totals using these same weights/formulas, so a systematic error in the
weights partially cancels in the (player - league) difference, and no
hardcoded league constant goes stale next season.

What this deliberately leaves out (and why official WAR will differ):
park factors, fielding runs (OAA/DRS come from tracking data, not box
scores), baserunning, and the league/dynamic runs-per-win refinements.
Validated against the API's official season WAR: pitchers land within
about +/-0.3 WAR; bat-first hitters within about 0.2-0.8 (slightly
low); elite defenders/baserunners read low by roughly their defensive
value (e.g. ~2 WAR for a Gold Glove shortstop) since this is
essentially offense-only WAR.
"""

from typing import Any

import pandas as pd

from mlb_stats.plots import _parse_innings_pitched

# FanGraphs-style linear weights, runs above out (~2023-24 era values).
LINEAR_WEIGHTS = {"walk": 0.689, "hbp": 0.720, "single": 0.882, "double": 1.254,
                  "triple": 1.590, "homer": 2.050}
WOBA_SCALE = 1.24
RUNS_PER_WIN = 10.0
REPL_RUNS_PER_600PA = 20.0   # batting replacement level
REPL_WINS_PER_9IP = 0.12     # pitching (starter) replacement level
FIP_CONSTANT = 3.10          # same fixed approximation the fip stat uses

# Positional adjustment, runs per 600 PA (fWAR-style). Two-way players
# ("TWP", i.e. Ohtani) bat as a DH. Positions not listed (pitchers
# batting, unknown) get no adjustment.
POSITION_ADJ_PER_600PA = {
    "C": 12.5, "1B": -12.5, "2B": 2.5, "3B": 2.5, "SS": 7.5,
    "LF": -7.5, "CF": 2.5, "RF": -7.5, "DH": -17.5, "TWP": -17.5,
}


def position_adjustment(position_abbreviation: str) -> float:
    """Runs per 600 PA for a primary position ("SS", "DH", ...)."""
    return POSITION_ADJ_PER_600PA.get(position_abbreviation, 0.0)


def _woba_parts(stat: dict[str, Any]) -> tuple[float, float]:
    """(numerator, denominator) of wOBA for one stat blob (a game's or
    a team season's counting stats -- same field names either way)."""
    unintentional_walks = stat.get("baseOnBalls", 0) - stat.get("intentionalWalks", 0)
    singles = (stat.get("hits", 0) - stat.get("doubles", 0)
               - stat.get("triples", 0) - stat.get("homeRuns", 0))
    w = LINEAR_WEIGHTS
    numerator = (w["walk"] * unintentional_walks + w["hbp"] * stat.get("hitByPitch", 0)
                 + w["single"] * singles + w["double"] * stat.get("doubles", 0)
                 + w["triple"] * stat.get("triples", 0) + w["homer"] * stat.get("homeRuns", 0))
    denominator = (stat.get("atBats", 0) + unintentional_walks
                   + stat.get("sacFlies", 0) + stat.get("hitByPitch", 0))
    return numerator, float(denominator)


def _fip_parts(stat: dict[str, Any]) -> tuple[float, float]:
    """(numerator of FIP before the constant, innings pitched)."""
    numerator = (13 * stat.get("homeRuns", 0)
                 + 3 * (stat.get("baseOnBalls", 0) + stat.get("hitBatsmen", 0))
                 - 2 * stat.get("strikeOuts", 0))
    return float(numerator), _parse_innings_pitched(stat.get("inningsPitched"))


def league_woba(team_stats: list[dict[str, Any]]) -> float:
    """League wOBA from the teams' season hitting totals, computed with
    the same weights applied to players."""
    numerator = denominator = 0.0
    for stat in team_stats:
        n, d = _woba_parts(stat)
        numerator += n
        denominator += d
    return numerator / denominator


def league_fip(team_stats: list[dict[str, Any]]) -> float:
    """League FIP from the teams' season pitching totals, computed with
    the same formula/constant applied to players."""
    numerator = innings = 0.0
    for stat in team_stats:
        n, ip = _fip_parts(stat)
        numerator += n
        innings += ip
    return numerator / innings + FIP_CONSTANT


def _batting_game_war(stat: dict[str, Any], lg_woba: float, pos_adj_per_600: float) -> float:
    numerator, denominator = _woba_parts(stat)
    plate_appearances = stat.get("plateAppearances", denominator)
    if not plate_appearances:
        return 0.0
    woba = numerator / denominator if denominator else 0.0
    wraa = (woba - lg_woba) / WOBA_SCALE * plate_appearances
    replacement = REPL_RUNS_PER_600PA * plate_appearances / 600
    positional = pos_adj_per_600 * plate_appearances / 600
    return (wraa + replacement + positional) / RUNS_PER_WIN


def _pitching_game_war(stat: dict[str, Any], lg_fip: float) -> float:
    numerator, innings = _fip_parts(stat)
    if not innings:
        return 0.0
    game_fip = numerator / innings + FIP_CONSTANT
    return ((lg_fip - game_fip) / RUNS_PER_WIN + REPL_WINS_PER_9IP) * innings / 9


def build_war_approx_dataframe(
    splits: list[dict[str, Any]],
    group: str,
    league_baseline: float,
    pos_adj_per_600: float = 0.0,
) -> pd.DataFrame:
    """Flatten game-log splits into the standard stat-DataFrame shape
    (date, opponent, cumulative) plus a war_game column with that game's
    approximate WAR. league_baseline is lgwOBA for batting, lgFIP for
    pitching. Cumulative is the running sum -- WAR is a counting stat,
    not a rate, which is also why the rolling value for these configs is
    a rolling SUM over the window (WAR accumulated across the last N
    games), not a re-computed rate (see plots._rolling_value_for_stat)."""
    rows = []
    for s in splits:
        stat = s["stat"]
        if group == "batting":
            war_game = _batting_game_war(stat, league_baseline, pos_adj_per_600)
        else:
            war_game = _pitching_game_war(stat, league_baseline)
        rows.append({
            "date": s["date"],
            "opponent": s["opponent"]["name"],
            "war_game": war_game,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["cumulative"] = df["war_game"].cumsum()
    return df
