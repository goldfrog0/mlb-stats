from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from mlb_stats.stats import get_stat_config

COMPARISON_LAYOUTS = ("overlay", "stacked", "side-by-side", "chefs-special")


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


def _weighted_numerator(df: pd.DataFrame, numerator_fields: dict[str, float]) -> pd.Series:
    return sum(df[field] * weight for field, weight in numerator_fields.items())


def _denominator(df: pd.DataFrame, denominator_fields: list[str]) -> pd.Series:
    return sum(df[field] for field in denominator_fields)


def _fields_for_stat(stat_key: str) -> set[str]:
    config = get_stat_config(stat_key)
    if "composite_of" in config:
        fields: set[str] = set()
        for component_key in config["composite_of"]:
            fields |= _fields_for_stat(component_key)
        return fields
    return set(config["numerator_fields"]) | set(config["denominator_fields"])


def build_stat_dataframe(splits: list[dict[str, Any]], stat_key: str) -> pd.DataFrame:
    """Flatten raw API splits into a DataFrame with the fields needed to
    compute and roll up stat_key."""
    config = get_stat_config(stat_key)
    fields = _fields_for_stat(stat_key)

    rows = []
    for s in splits:
        stat = s["stat"]
        row: dict[str, Any] = {
            "date": s["date"],
            "opponent": s["opponent"]["name"],
        }
        if config["cumulative_field"] is not None:
            row["cumulative"] = _to_float(stat.get(config["cumulative_field"]))
        for field in fields:
            row[field] = _parse_field(stat.get(field), field)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if config["cumulative_field"] is None:
        # No season-cumulative field for this stat in the API (e.g. FIP) --
        # compute it ourselves from the cumulative numerator/denominator.
        numerator = _weighted_numerator(df, config["numerator_fields"])
        denominator = _denominator(df, config["denominator_fields"])
        df["cumulative"] = (
            (numerator.cumsum() / denominator.cumsum()) * config["multiplier"] + config.get("constant", 0.0)
        )

    return df


def build_team_win_dataframe(games: list[dict[str, Any]], team_id: int) -> pd.DataFrame:
    """Flatten a team's schedule into the same shape build_stat_dataframe
    produces (date, opponent, cumulative, plus the raw numerator/
    denominator fields the "win_pct" config rolls up), so
    add_rolling_stat/compute_game_value/format_stat_table/the plotting
    functions all work unchanged for team win percentage as for any
    player stat -- a team schedule just needs its own flattening since
    it's shaped nothing like a player's game log.

    Only completed ("Final") games are included; future/scheduled games
    have no result yet. Ties (neither side isWinner) count as a
    non-win -- essentially moot in practice since modern MLB doesn't
    end games in ties under normal circumstances.
    """
    rows = []
    for game in games:
        if game["status"]["abstractGameState"] != "Final":
            continue

        home, away = game["teams"]["home"], game["teams"]["away"]
        this_side, other_side = (home, away) if home["team"]["id"] == team_id else (away, home)

        rows.append({
            "date": game["officialDate"],
            "opponent": other_side["team"]["name"],
            "win": 1 if this_side.get("isWinner") else 0,
            "gamesPlayed": 1,
        })

    if not rows:
        raise ValueError(f"No completed games found for team ID {team_id}")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Computed rather than trusting the API's leagueRecord.pct string,
    # which is pre-rounded to 3 decimals (same reasoning as FIP's
    # self-computed cumulative).
    df["cumulative"] = df["win"].cumsum() / df["gamesPlayed"].cumsum()

    return df


def _rolling_value_for_stat(df: pd.DataFrame, stat_key: str, window: int) -> pd.Series:
    config = get_stat_config(stat_key)
    if "composite_of" in config:
        return sum(_rolling_value_for_stat(df, component, window) for component in config["composite_of"])

    numerator = _weighted_numerator(df, config["numerator_fields"])
    denominator = _denominator(df, config["denominator_fields"])

    rolling_numerator = numerator.rolling(window=window).sum()
    rolling_denominator = denominator.rolling(window=window).sum()

    return (rolling_numerator / rolling_denominator) * config["multiplier"] + config.get("constant", 0.0)


def add_rolling_stat(df: pd.DataFrame, stat_key: str, window: int) -> pd.DataFrame:
    """Add a rolling stat column computed from summed numerator/denominator
    fields over the window (not by averaging per-game rates, which is
    mathematically invalid for a rate stat)."""
    df["rolling"] = _rolling_value_for_stat(df, stat_key, window)
    return df


def compute_game_value(df: pd.DataFrame, stat_key: str) -> pd.Series:
    """This game's own value of stat_key (not season-cumulative or
    rolling), computed from that row's own numerator/denominator fields."""
    config = get_stat_config(stat_key)
    if "composite_of" in config:
        return sum(compute_game_value(df, component) for component in config["composite_of"])
    numerator = _weighted_numerator(df, config["numerator_fields"])
    denominator = _denominator(df, config["denominator_fields"])
    return (numerator / denominator) * config["multiplier"] + config.get("constant", 0.0)


def format_stat_table(df: pd.DataFrame, stat_key: str) -> str:
    """Render the per-game data behind the plot (date, opponent, this
    game's own value, season-cumulative value, rolling value) as a plain
    aligned text table."""
    game_col = f"game_{stat_key}"
    season_col = f"season_{stat_key}"
    rolling_col = f"rolling_{stat_key}"

    table = df[["date", "opponent"]].copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    table[game_col] = compute_game_value(df, stat_key).round(3)
    table[season_col] = df["cumulative"].round(3)
    table[rolling_col] = df["rolling"].round(3)

    return table.to_string(index=False)


def _finish_plot(save_path: str | None) -> None:
    """Save the current figure if a path was given, otherwise display it.
    Saving must happen before plt.show(), since plt.show() blocks until
    the window is closed and can tear down the figure afterward."""
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_stat(
    df: pd.DataFrame,
    player_name: str,
    season: int,
    window: int,
    stat_key: str,
    save_path: str | None = None,
) -> None:
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
    _finish_plot(save_path)


def _rolling_by_date(df: pd.DataFrame) -> pd.Series:
    """A "rolling" series indexed by date, collapsed to the last game on
    any date with more than one (doubleheaders for a team, or -- in
    principle -- a player who appeared twice in one day). reindex()
    requires a unique index, and df is already sorted by date, so
    keeping the last occurrence per date is both correct (it reflects
    the rolling value as of the end of that date) and required."""
    s = df.set_index("date")["rolling"]
    return s[~s.index.duplicated(keep="last")]


def _reindexed_rolling_diff(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Compute player1's rolling value minus player2's, reindexed onto the
    union of both players' game dates. A player's rolling value holds
    (forward-fill) between their games until their next one, since the
    two players are rarely on the same game schedule."""
    s1 = _rolling_by_date(df1)
    s2 = _rolling_by_date(df2)
    all_dates = s1.index.union(s2.index).sort_values()
    diff = s1.reindex(all_dates).ffill() - s2.reindex(all_dates).ffill()
    return pd.DataFrame({"date": all_dates, "diff": diff.to_numpy()})


def _draw_diff_panel(
    ax: Axes, df1: pd.DataFrame, name1: str, df2: pd.DataFrame, name2: str,
    color1: str, color2: str, label: str,
) -> None:
    diff_df = _reindexed_rolling_diff(df1, df2)
    ax.fill_between(diff_df["date"], diff_df["diff"], 0,
                     where=(diff_df["diff"] >= 0), color=color1, alpha=0.3, interpolate=True)
    ax.fill_between(diff_df["date"], diff_df["diff"], 0,
                     where=(diff_df["diff"] < 0), color=color2, alpha=0.3, interpolate=True)
    ax.plot(diff_df["date"], diff_df["diff"], color="black", linewidth=1)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_ylabel(f"{label} diff\n({name1} − {name2})")
    ax.set_xlabel("Date")
    ax.tick_params(axis="x", rotation=45)


def plot_stat_comparison(
    df1: pd.DataFrame,
    name1: str,
    df2: pd.DataFrame,
    name2: str,
    season: int,
    window: int,
    stat_key: str,
    save_path: str | None = None,
    show_cumulative: bool = False,
    layout: str = "overlay",
    show_diff: bool = False,
) -> None:
    """Render a two-player comparison plot.

    layout: "overlay" (default, both players on one axes, matches the
    original comparison view), "stacked" (one axes per player, stacked
    vertically), "side-by-side" (one axes per player, side by side), or
    "chefs-special" (side-by-side layout with cumulative lines and the
    diff panel forced on, regardless of show_cumulative/show_diff).

    show_cumulative additionally draws each player's season-cumulative
    line (dashed, lower alpha) alongside their rolling line.

    show_diff adds a panel below showing player1's rolling value minus
    player2's (see _reindexed_rolling_diff).
    """
    if layout not in COMPARISON_LAYOUTS:
        raise ValueError(f"Unknown layout '{layout}'. Choose from: {', '.join(COMPARISON_LAYOUTS)}")

    if layout == "chefs-special":
        layout = "side-by-side"
        show_cumulative = True
        show_diff = True

    config = get_stat_config(stat_key)
    label = config["label"]
    color1, color2 = "crimson", "steelblue"

    extra_height = 2.5 if show_diff else 0

    if layout == "stacked":
        fig = plt.figure(figsize=(11, 8 + extra_height))
        gs = fig.add_gridspec(2 + (1 if show_diff else 0), 1,
                               height_ratios=[3, 3] + ([2] if show_diff else []))
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        panels = [(ax1, df1, name1, color1), (ax2, df2, name2, color2)]
        diff_ax = fig.add_subplot(gs[2], sharex=ax1) if show_diff else None
    elif layout == "side-by-side":
        fig = plt.figure(figsize=(13, 5 + extra_height))
        gs = fig.add_gridspec(1 + (1 if show_diff else 0), 2,
                               height_ratios=[3] + ([2] if show_diff else []))
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        panels = [(ax1, df1, name1, color1), (ax2, df2, name2, color2)]
        diff_ax = fig.add_subplot(gs[1, :]) if show_diff else None
    else:
        fig = plt.figure(figsize=(11, 5 + extra_height))
        gs = fig.add_gridspec(1 + (1 if show_diff else 0), 1,
                               height_ratios=[3] + ([2] if show_diff else []))
        ax = fig.add_subplot(gs[0])
        panels = [(ax, df1, name1, color1), (ax, df2, name2, color2)]
        diff_ax = fig.add_subplot(gs[1], sharex=ax) if show_diff else None

    for ax, df, name, color in panels:
        line_label = name if layout == "overlay" else f"Rolling {window}-game {label}"
        ax.plot(df["date"], df["rolling"], label=line_label, color=color, linewidth=2)
        if show_cumulative:
            # Hide the same early-season warm-up period the rolling line
            # already omits (fewer than `window` games), since a rate stat
            # computed from a handful of games can be wildly noisy and
            # dwarf the rest of the season on a shared axis.
            cumulative = df["cumulative"].copy()
            cumulative.iloc[: window - 1] = float("nan")
            cumulative_label = f"{name} season cumulative" if layout == "overlay" else f"Season cumulative {label}"
            ax.plot(df["date"], cumulative, label=cumulative_label,
                    color=color, linewidth=1.5, linestyle="--", alpha=0.4)

    if layout == "overlay":
        ax.set_ylabel(label)
        ax.set_xlabel("Date")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(f"{name1} vs {name2} — {label} Rolling {window}-Game Average ({season} Season)")
    else:
        for ax, _, name, _ in panels:
            ax.set_ylabel(label)
            ax.legend()
            ax.tick_params(axis="x", rotation=45)
            ax.set_title(name)
        panels[-1][0].set_xlabel("Date")
        fig.suptitle(f"{label} Rolling {window}-Game Average ({season} Season)")

    if diff_ax is not None:
        _draw_diff_panel(diff_ax, df1, name1, df2, name2, color1, color2, label)

    plt.tight_layout()
    _finish_plot(save_path)
