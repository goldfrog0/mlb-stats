"""Shared test fixtures.

The whole suite runs offline: these fixtures produce synthetic game-log
"splits" in exactly the shape the MLB Stats API returns them (see
mlb_stats/api.py), with small hand-checkable numbers so tests can assert
against values computed by hand rather than re-derived by the same code
under test.
"""

# Force a non-interactive matplotlib backend before mlb_stats.plots ever
# imports pyplot, so CLI tests can render without a display.
import matplotlib

matplotlib.use("Agg")

from typing import Any  # noqa: E402

import pytest  # noqa: E402


def _split(date: str, opponent: str, stat: dict[str, Any]) -> dict[str, Any]:
    return {"date": date, "opponent": {"name": opponent}, "stat": stat}


@pytest.fixture
def pitching_splits() -> list[dict[str, Any]]:
    """Six starts. Innings are all 6.0 except the last, which uses the
    box-score notation "6.2" = 6 and 2/3 innings (the parsing bug this
    project once had). Cumulative "era" strings are what the API would
    report for these totals.
    """
    earned_runs = [0, 1, 2, 3, 4, 5]
    innings = ["6.0", "6.0", "6.0", "6.0", "6.0", "6.2"]
    walks = [1, 2, 1, 2, 1, 2]
    hits = [4, 5, 6, 4, 5, 6]
    strikeouts = [7, 6, 5, 8, 7, 6]
    home_runs = [0, 1, 0, 1, 0, 1]
    hit_batsmen = [0, 0, 1, 0, 0, 1]
    cumulative_era = ["0.00", "0.75", "1.50", "2.25", "3.00", "3.67"]

    return [
        _split(
            f"2026-04-0{i + 1}",
            f"Opponent {i + 1}",
            {
                "era": cumulative_era[i],
                "earnedRuns": earned_runs[i],
                "inningsPitched": innings[i],
                "baseOnBalls": walks[i],
                "hits": hits[i],
                "strikeOuts": strikeouts[i],
                "homeRuns": home_runs[i],
                "hitBatsmen": hit_batsmen[i],
            },
        )
        for i in range(6)
    ]


TEAM_ID, TEAM_NAME = 119, "Los Angeles Dodgers"
OPPONENT_ID, OPPONENT_NAME = 137, "San Francisco Giants"


def _team_game(date: str, home_win: bool, team_is_home: bool, final: bool = True) -> dict[str, Any]:
    home_id, home_name = (TEAM_ID, TEAM_NAME) if team_is_home else (OPPONENT_ID, OPPONENT_NAME)
    away_id, away_name = (OPPONENT_ID, OPPONENT_NAME) if team_is_home else (TEAM_ID, TEAM_NAME)

    teams: dict[str, Any] = {
        "home": {"team": {"id": home_id, "name": home_name}},
        "away": {"team": {"id": away_id, "name": away_name}},
    }
    if final:
        teams["home"]["isWinner"] = home_win
        teams["away"]["isWinner"] = not home_win

    return {
        "officialDate": date,
        "status": {"abstractGameState": "Final" if final else "Preview"},
        "teams": teams,
    }


@pytest.fixture
def team_schedule_games() -> list[dict[str, Any]]:
    """Six completed games for TEAM_ID (mixed home/away), producing the
    sequence W L W W L L (3-3), plus one future/scheduled game that
    build_team_win_dataframe should filter out."""
    return [
        _team_game("2026-04-01", home_win=True, team_is_home=True),    # W (home team won, we're home)
        _team_game("2026-04-02", home_win=True, team_is_home=False),   # L (home team won, we're away)
        _team_game("2026-04-03", home_win=False, team_is_home=False),  # W (home team lost, we're away)
        _team_game("2026-04-04", home_win=True, team_is_home=True),    # W (home team won, we're home)
        _team_game("2026-04-05", home_win=False, team_is_home=True),   # L (home team lost, we're home)
        _team_game("2026-04-06", home_win=False, team_is_home=True),   # L (home team lost, we're home)
        _team_game("2026-04-07", home_win=True, team_is_home=True, final=False),  # future, excluded
    ]


@pytest.fixture
def batting_splits() -> list[dict[str, Any]]:
    """Six games with per-game counting stats plus the cumulative rate
    strings the API would report for these totals."""
    at_bats = [4, 4, 3, 5, 4, 4]
    hits = [1, 2, 0, 3, 1, 2]
    walks = [1, 0, 1, 0, 1, 0]
    hit_by_pitch = [0, 0, 0, 1, 0, 0]
    sac_flies = [0, 1, 0, 0, 0, 0]
    total_bases = [1, 3, 0, 7, 2, 3]
    cumulative_avg = [".250", ".375", ".273", ".375", ".350", ".375"]
    cumulative_obp = [".400", ".400", ".357", ".450", ".440", ".448"]
    cumulative_slg = [".250", ".500", ".364", ".688", ".650", ".667"]
    cumulative_ops = [".650", ".900", ".721", "1.138", "1.090", "1.115"]

    return [
        _split(
            f"2026-04-0{i + 1}",
            f"Opponent {i + 1}",
            {
                "avg": cumulative_avg[i],
                "obp": cumulative_obp[i],
                "slg": cumulative_slg[i],
                "ops": cumulative_ops[i],
                "atBats": at_bats[i],
                "hits": hits[i],
                "baseOnBalls": walks[i],
                "hitByPitch": hit_by_pitch[i],
                "sacFlies": sac_flies[i],
                "totalBases": total_bases[i],
            },
        )
        for i in range(6)
    ]


@pytest.fixture
def division_team_records() -> list[dict[str, Any]]:
    """Four teams' standings records, in the shape /standings returns
    them (already ranked best-to-worst, matching real AL East data used
    to verify this feature against the live API)."""
    teams = [
        ("Rays", 1, 56, 38, ".596", "-", "L1"),
        ("Yankees", 2, 54, 42, ".563", "3.0", "W4"),
        ("Red Sox", 3, 46, 48, ".489", "10.0", "W9"),
        ("Blue Jays", 4, 45, 51, ".469", "12.0", "L2"),
    ]
    return [
        {
            "team": {"id": 100 + rank, "name": name},
            "divisionRank": str(rank),
            "divisionGamesBack": gb,
            "gamesBack": gb,
            "leagueRecord": {"wins": wins, "losses": losses, "ties": 0, "pct": pct},
            "streak": {"streakCode": streak},
        }
        for name, rank, wins, losses, pct, gb, streak in teams
    ]
