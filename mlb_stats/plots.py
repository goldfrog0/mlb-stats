from typing import Any

import pandas as pd
import matplotlib.pyplot as plt

from mlb_stats.stats import get_stat_config


def _parse_innings_pitched(value: Any) -> float:
    """Innings pitched is reported in box-score notation, e.g. "6.2" means
    6 and 2/3 innings, not 6.2 decimal — the fractional part counts outs
    (0, 1, or 2), not tenths."""
    if value is None:
        return 0.0
    whole_str, _, thirds_str = str(value).partition(".")
    whole = float(whole_str) if whole_str else 0.0
    thirds = {"0": 0.0, "1": 1 / 3, "2": 2 / 3}.get(thirds_str, 0.0)
    return whole + thirds


def _to_float(value: Any) -> float:
    """Parse API stat values, which can be strings like '.253' or
    placeholders like '.---' for undefined rates."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_field(value: Any, field_name: str) -> float:
    if field_name == "inningsPitched":
        return _parse_innings_pitched(value)
    return _to_float(value)


def build_stat_dataframe(splits: list[dict[str, Any]], stat_key: str) -> pd.DataFrame:
    """Flatten raw API splits into a DataFrame with the fields needed to
    compute and roll up stat_key."""
    config = get_stat_config(stat_key)
    fields = set(config["numerator_fields"]) | set(config["denominator_fields"])

    rows = []
    for s in splits:
        stat = s["stat"]
        row = {
            "date": s["date"],
            "opponent": s["opponent"]["name"],
            "cumulative": _to_float(stat.get(config["cumulative_field"])),
        }
        for field in fields:
            row[field] = _parse_field(stat.get(field), field)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_rolling_stat(df: pd.DataFrame, stat_key: str, window: int) -> pd.DataFrame:
    """Add a rolling stat column computed from summed numerator/denominator
    fields over the window (not by averaging per-game rates, which is
    mathematically invalid for a rate stat)."""
    config = get_stat_config(stat_key)

    numerator = sum(df[f] for f in config["numerator_fields"])
    denominator = sum(df[f] for f in config["denominator_fields"])

    rolling_numerator = numerator.rolling(window=window).sum()
    rolling_denominator = denominator.rolling(window=window).sum()

    df["rolling"] = (rolling_numerator / rolling_denominator) * config["multiplier"]
    return df


def format_stat_table(df: pd.DataFrame, stat_key: str) -> str:
    """Render the per-game data behind the plot (date, opponent, this
    game's own value, season-cumulative value, rolling value) as a plain
    aligned text table."""
    config = get_stat_config(stat_key)
    game_col = f"game_{stat_key}"
    season_col = f"season_{stat_key}"
    rolling_col = f"rolling_{stat_key}"

    game_numerator = sum(df[f] for f in config["numerator_fields"])
    game_denominator = sum(df[f] for f in config["denominator_fields"])
    game_value = (game_numerator / game_denominator) * config["multiplier"]

    table = df[["date", "opponent"]].copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    table[game_col] = game_value.round(3)
    table[season_col] = df["cumulative"].round(3)
    table[rolling_col] = df["rolling"].round(3)

    return table.to_string(index=False)


def plot_stat(df: pd.DataFrame, player_name: str, season: int, window: int, stat_key: str) -> None:
    """Render the stat plot."""
    config = get_stat_config(stat_key)
    label = config["label"]

    plt.figure(figsize=(11, 5))
    plt.plot(
        df["date"], df["cumulative"],
        alpha=0.3, label=f"Season cumulative {label}", color="gray"
    )
    plt.plot(
        df["date"], df["rolling"],
        label=f"Rolling {window}-game {label}", color="crimson", linewidth=2
    )
    plt.title(f"{player_name} — {label} Over Time ({season} Season)")
    plt.ylabel(label)
    plt.xlabel("Date")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_stat_comparison(
    df1: pd.DataFrame,
    name1: str,
    df2: pd.DataFrame,
    name2: str,
    season: int,
    window: int,
    stat_key: str,
) -> None:
    """Render a two-player comparison plot, overlaying each player's rolling
    stat (the cumulative lines are dropped here since four lines together
    gets cluttered)."""
    config = get_stat_config(stat_key)
    label = config["label"]

    plt.figure(figsize=(11, 5))
    plt.plot(df1["date"], df1["rolling"], label=name1, color="crimson", linewidth=2)
    plt.plot(df2["date"], df2["rolling"], label=name2, color="steelblue", linewidth=2)
    plt.title(f"{name1} vs {name2} — {label} Rolling {window}-Game Average ({season} Season)")
    plt.ylabel(label)
    plt.xlabel("Date")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
