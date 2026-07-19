"""Registry of supported stats.

Each entry describes how to compute a proper rolling (or season, or
per-game) value for that stat:

    value = multiplier * (sum of weighted numerator fields) / (sum of
            denominator fields) + constant

This mirrors how ERA has to be computed (summing earned runs and
innings separately, not averaging per-game ERA values), which is the
correct way to roll up any rate stat. numerator_fields maps each raw
API field to its weight (most stats just weight everything 1, but e.g.
FIP needs 13*HR + 3*(BB+HBP) - 2*K). constant is added after the
multiplier -- most stats don't need one, but FIP does (a fixed
league-average adjustment so it sits on the same scale as ERA).

cumulative_field is the season-to-date value the API already returns
per game (e.g. "era"). Not every stat has one -- FIP has no such field,
so cumulative_field is None for it, and build_stat_dataframe() computes
the season-cumulative line itself instead (cumulative sum of the
weighted numerator over cumulative sum of the denominator).

To add a new stat, add a config entry here — no changes needed elsewhere.
"""

from typing import NotRequired, TypedDict


class StatConfig(TypedDict):
    label: str
    group: str
    cumulative_field: str | None
    numerator_fields: NotRequired[dict[str, float]]
    denominator_fields: NotRequired[list[str]]
    multiplier: NotRequired[float]
    constant: NotRequired[float]
    composite_of: NotRequired[list[str]]
    # Escape hatch for stats that aren't a numerator/denominator rate at
    # all. "war_approx" entries are built by war.build_war_approx_dataframe
    # (which needs league baselines and a positional adjustment the
    # rate machinery has no concept of) and roll up as sums, not rates.
    computation: NotRequired[str]


STAT_CONFIGS: dict[str, StatConfig] = {
    "era": {
        "label": "ERA",
        "group": "pitching",
        "cumulative_field": "era",
        "numerator_fields": {"earnedRuns": 1},
        "denominator_fields": ["inningsPitched"],
        "multiplier": 9,
    },
    "whip": {
        "label": "WHIP",
        "group": "pitching",
        "cumulative_field": "whip",
        "numerator_fields": {"baseOnBalls": 1, "hits": 1},
        "denominator_fields": ["inningsPitched"],
        "multiplier": 1,
    },
    "k9": {
        "label": "K/9",
        "group": "pitching",
        "cumulative_field": "strikeoutsPer9Inn",
        "numerator_fields": {"strikeOuts": 1},
        "denominator_fields": ["inningsPitched"],
        "multiplier": 9,
    },
    "bb9": {
        "label": "BB/9",
        "group": "pitching",
        "cumulative_field": "walksPer9Inn",
        "numerator_fields": {"baseOnBalls": 1},
        "denominator_fields": ["inningsPitched"],
        "multiplier": 9,
    },
    "fip": {
        "label": "FIP",
        "group": "pitching",
        # The MLB Stats API doesn't return a season-cumulative FIP field,
        # so it's computed ourselves (see build_stat_dataframe).
        "cumulative_field": None,
        "numerator_fields": {"homeRuns": 13, "baseOnBalls": 3, "hitBatsmen": 3, "strikeOuts": -2},
        "denominator_fields": ["inningsPitched"],
        "multiplier": 1,
        # The real FIP constant is recalculated by MLB every season from
        # league-wide totals (usually ~3.05-3.20) so that league FIP lines
        # up with league ERA. That requires a separate league-totals query
        # this app doesn't make, so this is a commonly-cited fixed
        # approximation rather than the exact value for any given season.
        "constant": 3.10,
    },
    "avg": {
        "label": "AVG",
        "group": "batting",
        "cumulative_field": "avg",
        "numerator_fields": {"hits": 1},
        "denominator_fields": ["atBats"],
        "multiplier": 1,
    },
    "obp": {
        "label": "OBP",
        "group": "batting",
        "cumulative_field": "obp",
        "numerator_fields": {"hits": 1, "baseOnBalls": 1, "hitByPitch": 1},
        "denominator_fields": ["atBats", "baseOnBalls", "hitByPitch", "sacFlies"],
        "multiplier": 1,
    },
    "slg": {
        "label": "SLG",
        "group": "batting",
        "cumulative_field": "slg",
        "numerator_fields": {"totalBases": 1},
        "denominator_fields": ["atBats"],
        "multiplier": 1,
    },
    "ops": {
        "label": "OPS",
        "group": "batting",
        "cumulative_field": "ops",
        "composite_of": ["obp", "slg"],
    },
    "bwar": {
        "label": "Batting WAR (approx)",
        "group": "batting",
        "cumulative_field": None,
        "computation": "war_approx",
    },
    "pwar": {
        "label": "Pitching WAR (approx)",
        "group": "pitching",
        "cumulative_field": None,
        "computation": "war_approx",
    },
    "win_pct": {
        "label": "Win%",
        "group": "team",
        # Unlike every other entry here, the "team" group doesn't go
        # through build_stat_dataframe (a team schedule is shaped nothing
        # like a player's game log), so this field isn't actually
        # consulted -- build_team_win_dataframe() sets the cumulative
        # column itself, computed rather than trusting the API's
        # pre-rounded leagueRecord.pct string.
        "cumulative_field": None,
        "numerator_fields": {"win": 1},
        "denominator_fields": ["gamesPlayed"],
        "multiplier": 1,
    },
}


def get_stat_config(stat_key: str) -> StatConfig:
    try:
        return STAT_CONFIGS[stat_key]
    except KeyError:
        valid = ", ".join(sorted(STAT_CONFIGS))
        raise ValueError(f"Unknown stat '{stat_key}'. Choose from: {valid}")
