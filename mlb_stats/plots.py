from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


def build_standings_dataframe(team_records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten a division's standings into a display-ready DataFrame,
    sorted best-to-worst. Unlike every other build_*_dataframe in this
    module, there's no per-game rolling/cumulative concept here --
    standings are a single current snapshot, so this just reshapes what
    the API already gives us rather than computing anything."""
    rows = [
        {
            "rank": int(tr["divisionRank"]),
            "team": tr["team"]["name"],
            "wins": tr["leagueRecord"]["wins"],
            "losses": tr["leagueRecord"]["losses"],
            "pct": float(tr["leagueRecord"]["pct"]),
            "games_back": tr.get("divisionGamesBack", "-"),
            "streak": tr.get("streak", {}).get("streakCode", ""),
        }
        for tr in team_records
    ]
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def filter_splits_by_date(
    splits: list[dict[str, Any]], start_date: str | None = None, end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Keep only the game-log splits within [start_date, end_date]
    (inclusive, ISO "YYYY-MM-DD" strings, either side optional). ISO
    date strings compare lexicographically in chronological order, so
    plain string comparison is correct here."""
    kept = [
        s for s in splits
        if (start_date is None or s["date"] >= start_date)
        and (end_date is None or s["date"] <= end_date)
    ]
    if not kept:
        window = f"{start_date or 'season start'} and {end_date or 'season end'}"
        raise ValueError(f"No games found between {window}")
    return kept


def build_pitch_dataframe(
    games: list[tuple[dict[str, Any], list[dict[str, Any]]]], pitcher_id: int,
) -> pd.DataFrame:
    """Flatten (game-log split, that game's pitches) pairs into one row
    per pitch thrown by pitcher_id: date, opponent, pitch_type, velo.

    Unlike the stat DataFrames, `date` stays an ISO string: the velocity
    chart treats each game as a categorical column rather than a point
    on a continuous time axis, so no date arithmetic is ever needed.
    Pitches with no tracked velocity are dropped."""
    rows = [
        {
            "date": split["date"],
            "opponent": split["opponent"]["name"],
            "pitch_type": pitch["pitch_type"],
            "velo": pitch["velo"],
        }
        for split, pitches in games
        for pitch in pitches
        if pitch["pitcher_id"] == pitcher_id and pitch["velo"] is not None
    ]

    if not rows:
        raise ValueError(f"No pitch data found for player ID {pitcher_id}")

    return pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)


def format_pitch_table(df: pd.DataFrame) -> str:
    """Render a per-game summary of the pitch velocities behind the
    chart (one row per game, not per pitch -- a full season is thousands
    of pitches)."""
    grouped = df.groupby("date", sort=True)
    table = pd.DataFrame({
        "date": grouped["date"].first(),
        "opponent": grouped["opponent"].first(),
        "pitches": grouped["velo"].count(),
        "max_velo": grouped["velo"].max().round(1),
        "median_velo": grouped["velo"].median().round(1),
        "min_velo": grouped["velo"].min().round(1),
    })
    return table.to_string(index=False)


DEFAULT_PITCH_TYPE = "Four-Seam Fastball"


def filter_pitch_type(df: pd.DataFrame, pitch_type: str) -> pd.DataFrame:
    """Keep only pitches whose type contains pitch_type (case-insensitive
    substring), so "fastball", "four-seam", or the full "Four-Seam
    Fastball" all select the same pitches."""
    return df[df["pitch_type"].str.contains(pitch_type, case=False, na=False)]


def pitch_velocity_by_game(df: pd.DataFrame, pitch_type: str, pitcher_name: str) -> pd.DataFrame:
    """Collapse a per-pitch DataFrame (from build_pitch_dataframe) to one
    row per game for a single pitch type: date, opponent, and the game's
    mean/min/max velocity and pitch count. This is the per-game series
    behind a velocity-comparison line -- the average is what the line
    plots, the min/max give the shaded spread band. Raises if the
    pitcher threw none of that pitch type (a fair comparison needs both
    sides to have it)."""
    filtered = filter_pitch_type(df, pitch_type)
    if filtered.empty:
        raise ValueError(f"No {pitch_type} pitches found for {pitcher_name}")

    grouped = filtered.groupby("date", sort=True)
    return pd.DataFrame({
        "date": grouped["date"].first(),
        "opponent": grouped["opponent"].first(),
        "count": grouped["velo"].count(),
        "mean": grouped["velo"].mean(),
        "min": grouped["velo"].min(),
        # Quartiles for the box-and-whisker view; min/max double as the
        # whisker fences (full extent, no hidden outliers).
        "q1": grouped["velo"].quantile(0.25),
        "median": grouped["velo"].median(),
        "q3": grouped["velo"].quantile(0.75),
        "max": grouped["velo"].max(),
    }).reset_index(drop=True)


def format_pitch_comparison_table(by_game: pd.DataFrame, pitch_type: str) -> str:
    """Render a pitcher's per-game velocity for one pitch type: date,
    opponent, pitch count, and mean/min/max velo (the mean is the line
    the comparison chart draws)."""
    table = by_game.rename(columns={
        "date": "date", "opponent": "opponent", "count": pitch_type.split()[0].lower() + "s",
        "mean": "avg_velo", "min": "min_velo", "max": "max_velo",
    }).copy()
    for column in ("avg_velo", "min_velo", "max_velo"):
        table[column] = table[column].round(1)
    return table.to_string(index=False)


def format_standings_table(df: pd.DataFrame) -> str:
    """Render a division's standings as a plain aligned text table."""
    table = df.rename(columns={
        "rank": "Rank", "team": "Team", "wins": "W", "losses": "L",
        "pct": "PCT", "games_back": "GB", "streak": "Streak",
    }).copy()
    table["PCT"] = table["PCT"].round(3)
    return table.to_string(index=False)


def _rolling_value_for_stat(df: pd.DataFrame, stat_key: str, window: int) -> pd.Series:
    config = get_stat_config(stat_key)
    if config.get("computation") == "war_approx":
        # WAR is a counting stat, not a rate: the rolling value is WAR
        # accumulated over the last N games (a rolling sum), not a
        # numerator/denominator recomputation.
        return df["war_game"].rolling(window=window).sum()
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
    if config.get("computation") == "war_approx":
        return df["war_game"]
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
        # Release the figure -- a one-shot CLI run wouldn't care, but the
        # test suite renders many in one process and pyplot keeps every
        # unclosed figure in memory ("More than 20 figures" warning).
        plt.close()
    else:
        plt.show()
        plt.close()


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


PITCH_BOX_STYLES = ("game", "type")


def plot_pitch_velocities(
    df: pd.DataFrame, player_name: str, season: int, save_path: str | None = None,
    box: str | None = None,
) -> None:
    """Render every pitch as a dot, one column of dots per game (jittered
    horizontally so same-speed pitches don't stack into a single point),
    colored by pitch type, with a dashed line tracing each game's max
    velocity across the stretch.

    box overlays box-and-whisker summaries on the (then dimmed) dots:
    "game" draws one box per game across all its pitches; "type" draws
    one box per pitch type within each game, offset side by side and
    color-matched to the dots. Whiskers cap at 1.5*IQR and outliers are
    hidden, so the game-max line still marks each game's true peak."""
    dates = sorted(df["date"].unique())
    x_by_date = {date: i for i, date in enumerate(dates)}
    # Fixed colors per pitch type (sorted) so the dots and the "type"
    # boxes share one palette rather than drifting between draws.
    types = sorted(df["pitch_type"].unique())
    palette = plt.get_cmap("tab10")
    color_by_type = {t: palette(i % 10) for i, t in enumerate(types)}

    fig_width = max(11.0, 1.1 * len(dates) + 3)
    plt.figure(figsize=(fig_width, 6))

    # Dots recede to background context when boxes carry the summary.
    dot_alpha = 0.22 if box else 0.6

    # Seeded so repeated runs of the same command produce the same image.
    rng = np.random.default_rng(0)
    for pitch_type in types:
        group = df[df["pitch_type"] == pitch_type]
        x = group["date"].map(x_by_date) + rng.uniform(-0.28, 0.28, len(group))
        plt.scatter(x, group["velo"], s=14, alpha=dot_alpha,
                    color=color_by_type[pitch_type], label=pitch_type)

    if box == "game":
        data = [df[df["date"] == d]["velo"].to_numpy() for d in dates]
        bp = plt.boxplot(data, positions=list(range(len(dates))), widths=0.5,
                         manage_ticks=False, showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightgray")
            patch.set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color("crimson")
    elif box == "type":
        n = max(len(types), 1)
        slot = 0.6 / n
        for j, pitch_type in enumerate(types):
            offset = (j - (n - 1) / 2) * slot
            positions, data = [], []
            for i, d in enumerate(dates):
                velos = df[(df["date"] == d) & (df["pitch_type"] == pitch_type)]["velo"].to_numpy()
                if len(velos):
                    positions.append(i + offset)
                    data.append(velos)
            if not data:
                continue
            bp = plt.boxplot(data, positions=positions, widths=slot * 0.8,
                             manage_ticks=False, showfliers=False, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(color_by_type[pitch_type])
                patch.set_alpha(0.5)
            for median in bp["medians"]:
                median.set_color("black")

    game_max = df.groupby("date")["velo"].max().reindex(dates)
    plt.plot(range(len(dates)), game_max, color="black", linewidth=1,
             linestyle="--", marker="_", markersize=14, label="Game max")

    plt.xticks(range(len(dates)), dates, rotation=45)
    plt.xlim(-0.5, len(dates) - 0.5)
    plt.xlabel("Game date")
    plt.ylabel("Velocity (mph)")
    plt.title(f"{player_name} — Pitch Velocities by Game ({season} Season)")
    # Outside the axes: the dot columns fill the plot area edge to edge,
    # so any in-plot legend position would cover data.
    plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    _finish_plot(save_path)


VELO_COMPARISON_LAYOUTS = ("stacked", "side-by-side", "overlay")


def _draw_velo_panel(
    ax: Axes, by_game: pd.DataFrame, color: str, label: str | None = None, box: bool = False,
) -> None:
    """One pitcher's per-game velocity on a real date axis so the trend
    reads as over-time. By default a shaded min-max band; with box, a
    box-and-whisker per game (quartiles, whiskers to the full min-max)
    in its place. Either way the mean is drawn on top as a marked line."""
    dates = pd.to_datetime(by_game["date"])
    if box:
        stats = [
            {"med": row["median"], "q1": row["q1"], "q3": row["q3"],
             "whislo": row["min"], "whishi": row["max"], "fliers": []}
            for _, row in by_game.iterrows()
        ]
        bp = ax.bxp(stats, positions=mdates.date2num(dates), widths=2.0,
                    patch_artist=True, showfliers=False, manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
        for element in ("whiskers", "caps", "medians"):
            for line in bp[element]:
                line.set_color(color)
        ax.xaxis_date()
    else:
        ax.fill_between(dates, by_game["min"], by_game["max"], color=color, alpha=0.15)
    # Track the median with boxes (it's the box's own centre line, so the
    # trend line stays centred on each box) and the mean with the band.
    center = by_game["median"] if box else by_game["mean"]
    ax.plot(dates, center, color=color, marker="o", markersize=4, linewidth=2, label=label)


def plot_pitch_velocity_comparison(
    by_game1: pd.DataFrame,
    name1: str,
    by_game2: pd.DataFrame,
    name2: str,
    season: int,
    pitch_type: str,
    layout: str = "stacked",
    save_path: str | None = None,
    box: bool = False,
) -> None:
    """Compare two pitchers' velocity for one pitch type over a season:
    each pitcher's per-game mean (marked line) over a shaded min-max
    band, or -- with box -- over a box-and-whisker per game in the
    band's place. Inputs are per-game frames from pitch_velocity_by_game.

    layout: "stacked" (one panel per pitcher, stacked, sharing the date
    axis so timelines line up), "side-by-side" (panels side by side
    sharing the velocity axis for direct scale comparison), or "overlay"
    (both on one axes)."""
    if layout not in VELO_COMPARISON_LAYOUTS:
        raise ValueError(f"Unknown velo layout '{layout}'. Choose from: {', '.join(VELO_COMPARISON_LAYOUTS)}")

    color1, color2 = "crimson", "steelblue"
    panels = [(by_game1, name1, color1), (by_game2, name2, color2)]
    suptitle = f"{pitch_type} Velocity by Game ({season} Season)"

    if layout == "overlay":
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for by_game, name, color in panels:
            _draw_velo_panel(ax, by_game, color, label=name, box=box)
        ax.set_ylabel("Velocity (mph)")
        ax.set_xlabel("Game date")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(f"{name1} vs {name2} — {suptitle}")
    elif layout == "side-by-side":
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
        for ax, (by_game, name, color) in zip(axes, panels):
            _draw_velo_panel(ax, by_game, color, box=box)
            ax.set_title(name)
            ax.set_xlabel("Game date")
            ax.tick_params(axis="x", rotation=45)
        axes[0].set_ylabel("Velocity (mph)")
        fig.suptitle(suptitle)
    else:  # stacked
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, sharey=True)
        for ax, (by_game, name, color) in zip(axes, panels):
            _draw_velo_panel(ax, by_game, color, box=box)
            ax.set_title(name)
            ax.set_ylabel("Velocity (mph)")
        axes[-1].set_xlabel("Game date")
        axes[-1].tick_params(axis="x", rotation=45)
        fig.suptitle(suptitle)

    plt.tight_layout()
    _finish_plot(save_path)


def plot_standings(df: pd.DataFrame, division_name: str, season: int, save_path: str | None = None) -> None:
    """Render a division's standings as a horizontal bar chart (win% per
    team, best team on top), annotated with each team's W-L record."""
    fig_height = 1.2 + 0.6 * len(df)
    plt.figure(figsize=(9, fig_height))

    # barh plots its first row at the bottom, so reverse the (already
    # rank-sorted) rows to put the division leader at the top instead.
    ordered = df.iloc[::-1]
    colors = ["crimson" if rank == 1 else "steelblue" for rank in ordered["rank"]]
    bars = plt.barh(ordered["team"], ordered["pct"], color=colors)

    max_pct = df["pct"].max()
    for bar, (_, row) in zip(bars, ordered.iterrows()):
        label = f"{row['wins']}-{row['losses']}  ({row['pct']:.3f})"
        plt.text(bar.get_width() + max_pct * 0.02, bar.get_y() + bar.get_height() / 2,
                  label, va="center", fontsize=9)

    plt.xlabel("Win%")
    plt.title(f"{division_name} Standings ({season} Season)")
    plt.xlim(0, max_pct * 1.3)
    plt.tight_layout()
    _finish_plot(save_path)
