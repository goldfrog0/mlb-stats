"""Registry of supported stats.

Each entry describes how to compute a proper rolling value for that stat:
multiplier * sum(numerator fields over the window) / sum(denominator fields
over the window). This mirrors how ERA has to be computed (summing earned
runs and innings separately, not averaging per-game ERA values), which is
the correct way to roll up any rate stat.

To add a new stat, add a config entry here — no changes needed elsewhere.
"""

from typing import TypedDict


class StatConfig(TypedDict):
    label: str
    group: str
    cumulative_field: str
    numerator_fields: list[str]
    denominator_fields: list[str]
    multiplier: int


STAT_CONFIGS: dict[str, StatConfig] = {
    "era": {
        "label": "ERA",
        "group": "pitching",
        "cumulative_field": "era",
        "numerator_fields": ["earnedRuns"],
        "denominator_fields": ["inningsPitched"],
        "multiplier": 9,
    },
    "whip": {
        "label": "WHIP",
        "group": "pitching",
        "cumulative_field": "whip",
        "numerator_fields": ["baseOnBalls", "hits"],
        "denominator_fields": ["inningsPitched"],
        "multiplier": 1,
    },
    "k9": {
        "label": "K/9",
        "group": "pitching",
        "cumulative_field": "strikeoutsPer9Inn",
        "numerator_fields": ["strikeOuts"],
        "denominator_fields": ["inningsPitched"],
        "multiplier": 9,
    },
    "bb9": {
        "label": "BB/9",
        "group": "pitching",
        "cumulative_field": "walksPer9Inn",
        "numerator_fields": ["baseOnBalls"],
        "denominator_fields": ["inningsPitched"],
        "multiplier": 9,
    },
    "avg": {
        "label": "AVG",
        "group": "batting",
        "cumulative_field": "avg",
        "numerator_fields": ["hits"],
        "denominator_fields": ["atBats"],
        "multiplier": 1,
    },
    "obp": {
        "label": "OBP",
        "group": "batting",
        "cumulative_field": "obp",
        "numerator_fields": ["hits", "baseOnBalls", "hitByPitch"],
        "denominator_fields": ["atBats", "baseOnBalls", "hitByPitch", "sacFlies"],
        "multiplier": 1,
    },
    "slg": {
        "label": "SLG",
        "group": "batting",
        "cumulative_field": "slg",
        "numerator_fields": ["totalBases"],
        "denominator_fields": ["atBats"],
        "multiplier": 1,
    },
}


def get_stat_config(stat_key: str) -> StatConfig:
    try:
        return STAT_CONFIGS[stat_key]
    except KeyError:
        valid = ", ".join(sorted(STAT_CONFIGS))
        raise ValueError(f"Unknown stat '{stat_key}'. Choose from: {valid}")
