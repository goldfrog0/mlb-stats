import argparse
import datetime
import re

import pandas as pd

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
)
from mlb_stats.plots import (
    COMPARISON_LAYOUTS,
    DEFAULT_PITCH_TYPE,
    VELO_COMPARISON_LAYOUTS,
    build_pitch_dataframe,
    build_stat_dataframe,
    build_standings_dataframe,
    build_team_win_dataframe,
    add_rolling_stat,
    filter_pitch_type,
    filter_splits_by_date,
    format_pitch_comparison_table,
    format_pitch_table,
    format_standings_table,
    format_stat_table,
    pitch_velocity_by_game,
    plot_pitch_velocities,
    plot_pitch_velocity_comparison,
    plot_standings,
    plot_stat,
    plot_stat_comparison,
)
from mlb_stats.stats import STAT_CONFIGS, get_stat_config
from mlb_stats.war import build_war_approx_dataframe, league_fip, league_woba, position_adjustment

DEFAULT_WINDOW = 5
CURRENT_YEAR = datetime.date.today().year
AUTO_SAVE = "__auto__"  # sentinel for --save passed with no filename


def _load_stat_dataframe(name: str, season: int, stat_key: str, window: int) -> tuple[pd.DataFrame, str]:
    """Look up a player or team, pull their game log/schedule for
    stat_key, and return the rolling-stat DataFrame alongside their
    resolved full name."""
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
    return df, full_name


def _load_pitch_dataframe(
    name: str, season: int, start_date: str | None, end_date: str | None,
) -> tuple[pd.DataFrame, str]:
    """Look up a pitcher, find their games in the date range from the
    pitching game log, and pull each game's play-by-play pitch data."""
    player_id, full_name = find_player(name)
    splits = get_game_log(player_id, season, "pitching")
    splits = filter_splits_by_date(splits, start_date, end_date)
    games = [(s, get_game_pitches(s["game"]["gamePk"])) for s in splits]
    df = build_pitch_dataframe(games, player_id)
    return df, full_name


def _filter_single_velo(df: pd.DataFrame, pitch_type: str, pitcher_name: str) -> pd.DataFrame:
    """Restrict a single pitcher's per-pitch DataFrame to one pitch type,
    raising with a clear message rather than plotting an empty chart."""
    filtered = filter_pitch_type(df, pitch_type)
    if filtered.empty:
        raise ValueError(f"No {pitch_type} pitches found for {pitcher_name}")
    return filtered


def _load_standings_dataframe(division_name: str, season: int) -> tuple[pd.DataFrame, str]:
    division_id, full_name = find_division(division_name)
    team_records = get_division_standings(division_id, season)
    df = build_standings_dataframe(team_records)
    return df, full_name


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _today_str() -> str:
    return datetime.date.today().isoformat()


def _auto_filename_single(player: str, stat: str, season: int, window: int) -> str:
    name = f"{_slugify(player)}_{stat}_{season}"
    if window != DEFAULT_WINDOW:
        name += f"_w{window}"
    name += f"_{_today_str()}"
    return f"{name}.png"


def _auto_filename_standings(division_name: str, season: int) -> str:
    return f"{_slugify(division_name)}_standings_{season}_{_today_str()}.png"


def _auto_filename_velo(player: str, season: int, start_date: str | None, end_date: str | None) -> str:
    name = f"{_slugify(player)}_velo_{season}"
    if start_date:
        name += f"_from{start_date}"
    if end_date:
        name += f"_to{end_date}"
    return f"{name}_{_today_str()}.png"


def _auto_filename_velo_compare(
    player1: str, player2: str, pitch_type: str, season: int, layout: str,
) -> str:
    name = f"{_slugify(player1)}_vs_{_slugify(player2)}_velo_{_slugify(pitch_type)}_{season}"
    if layout != "stacked":
        name += f"_{layout}"
    return f"{name}_{_today_str()}.png"


def _auto_filename_compare(
    player1: str, player2: str, stat: str, season: int, window: int,
    layout: str, show_cumulative: bool, show_diff: bool,
) -> str:
    name = f"{_slugify(player1)}_vs_{_slugify(player2)}_{stat}_{season}"
    if window != DEFAULT_WINDOW:
        name += f"_w{window}"
    if layout != "overlay":
        name += f"_{layout}"
    if show_cumulative:
        name += "_cumulative"
    if show_diff:
        name += "_diff"
    name += f"_{_today_str()}"
    return f"{name}.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mlb-stats",
        description="MLB player stat visualizer"
    )
    parser.add_argument("player", type=str, nargs="?", default=None,
                        help='Player name, e.g. "Shohei Ohtani" -- or a team name for team stats '
                             'like win_pct, e.g. "Los Angeles Dodgers". Not required with '
                             "--standings")
    parser.add_argument("player2", type=str, nargs="?", default=None,
                        help="Optional second player or team to compare against")
    parser.add_argument("--stat", type=str, default="era", choices=sorted(STAT_CONFIGS),
                        help="Stat to plot (default: era)")
    parser.add_argument("--standings", type=str, default=None, metavar="DIVISION",
                        help='Show a division\'s standings instead of plotting a player/team, '
                             'e.g. --standings "AL East". Ignores player/player2/--stat; '
                             "--season, --save, and --table still apply")
    parser.add_argument("--velo", action="store_true",
                        help="Plot every pitch's release velocity as a dot column per game, "
                             "colored by pitch type, with a line tracing each game's max velo. "
                             "The player must be a pitcher; --stat/--window don't apply. Give a "
                             "second pitcher to compare their per-game average velocity for one "
                             "pitch type (see --pitch-type) over the season, arranged by --layout. "
                             "--season, --save, --table, and --start-date/--end-date all work")
    parser.add_argument("--pitch-type", type=str, default=None, metavar="TYPE",
                        help='--velo only: restrict to one pitch type, matched case-insensitively '
                             'as a substring ("fastball", "four-seam", "slider", ...). In velo '
                             f'comparison mode the average line is for this type (default: '
                             f'"{DEFAULT_PITCH_TYPE}")')
    parser.add_argument("--start-date", type=str, default=None, metavar="YYYY-MM-DD",
                        help="--velo only: skip games before this date (inclusive)")
    parser.add_argument("--end-date", type=str, default=None, metavar="YYYY-MM-DD",
                        help="--velo only: skip games after this date (inclusive)")
    parser.add_argument("--season", type=int, default=CURRENT_YEAR, help=f"Season year (default: {CURRENT_YEAR})")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help=f"Rolling average window (default: {DEFAULT_WINDOW})")
    parser.add_argument("--save", type=str, nargs="?", const=AUTO_SAVE, default=None, metavar="FILE",
                        help="Save plot to file instead of displaying. Give a FILE to name it "
                             "yourself, or pass --save with no FILE to auto-generate one from "
                             "the player(s)/stat/season (and window/layout/diff/cumulative when "
                             "non-default), e.g. shohei-ohtani_era_2026.png")
    parser.add_argument("--table", action="store_true",
                        help="Also print the plotted data as a text table")
    parser.add_argument("--layout", type=str, default=None, choices=COMPARISON_LAYOUTS,
                        help="Comparison mode only: how to arrange the two players' charts. "
                             "Default: overlay for stat comparison, stacked for --velo "
                             "comparison (which has no overlay/chefs-special)")
    parser.add_argument("--show-cumulative", action="store_true",
                        help="Comparison mode only: also draw each player's season-cumulative "
                             "line (dashed)")
    parser.add_argument("--diff", action="store_true",
                        help="Comparison mode only: add a panel showing player 1's rolling "
                             "value minus player 2's")

    args = parser.parse_args()

    if not args.standings and not args.player:
        parser.error("player is required unless --standings is given")
    if (args.start_date or args.end_date) and not args.velo:
        parser.error("--start-date/--end-date only apply to --velo")
    if args.pitch_type and not args.velo:
        parser.error("--pitch-type only applies to --velo")

    # Layout defaults differ by mode; a velo comparison has no overlay-vs-
    # rolling-lines concept, so its default is stacked and chefs-special
    # (a stat-only layout) isn't valid.
    if args.layout is None:
        args.layout = "stacked" if (args.velo and args.player2) else "overlay"
    if args.velo and args.player2 and args.layout not in VELO_COMPARISON_LAYOUTS:
        parser.error(f"--velo comparison layout must be one of: {', '.join(VELO_COMPARISON_LAYOUTS)}")

    try:
        if args.standings:
            df, division_name = _load_standings_dataframe(args.standings, args.season)

            if args.table:
                print(f"\n{division_name}")
                print("-" * len(division_name))
                print(format_standings_table(df))

            save_path = args.save
            if save_path == AUTO_SAVE:
                save_path = _auto_filename_standings(division_name, args.season)

            plot_standings(df, division_name, args.season, save_path=save_path)
        elif args.velo and args.player2:
            pitch_type = args.pitch_type or DEFAULT_PITCH_TYPE
            df1, name1 = _load_pitch_dataframe(args.player, args.season, args.start_date, args.end_date)
            df2, name2 = _load_pitch_dataframe(args.player2, args.season, args.start_date, args.end_date)
            by_game1 = pitch_velocity_by_game(df1, pitch_type, name1)
            by_game2 = pitch_velocity_by_game(df2, pitch_type, name2)

            if args.table:
                for name, by_game in [(name1, by_game1), (name2, by_game2)]:
                    print(f"\n{name}")
                    print("-" * len(name))
                    print(format_pitch_comparison_table(by_game, pitch_type))

            save_path = args.save
            if save_path == AUTO_SAVE:
                save_path = _auto_filename_velo_compare(name1, name2, pitch_type, args.season, args.layout)

            plot_pitch_velocity_comparison(by_game1, name1, by_game2, name2, args.season,
                                            pitch_type, layout=args.layout, save_path=save_path)
        elif args.velo:
            df, full_name = _load_pitch_dataframe(args.player, args.season, args.start_date, args.end_date)
            if args.pitch_type:
                df = _filter_single_velo(df, args.pitch_type, full_name)

            if args.table:
                print(f"\n{full_name}")
                print("-" * len(full_name))
                print(format_pitch_table(df))

            save_path = args.save
            if save_path == AUTO_SAVE:
                save_path = _auto_filename_velo(full_name, args.season, args.start_date, args.end_date)

            plot_pitch_velocities(df, full_name, args.season, save_path=save_path)
        elif args.player2:
            df1, name1 = _load_stat_dataframe(args.player, args.season, args.stat, args.window)
            df2, name2 = _load_stat_dataframe(args.player2, args.season, args.stat, args.window)

            if args.table:
                for name, df in [(name1, df1), (name2, df2)]:
                    print(f"\n{name}")
                    print("-" * len(name))
                    print(format_stat_table(df, args.stat))

            save_path = args.save
            if save_path == AUTO_SAVE:
                save_path = _auto_filename_compare(
                    name1, name2, args.stat, args.season, args.window,
                    args.layout, args.show_cumulative, args.diff,
                )

            plot_stat_comparison(df1, name1, df2, name2, args.season, args.window, args.stat,
                                  save_path=save_path, show_cumulative=args.show_cumulative,
                                  layout=args.layout, show_diff=args.diff)
        else:
            df, full_name = _load_stat_dataframe(args.player, args.season, args.stat, args.window)

            if args.table:
                print(f"\n{full_name}")
                print("-" * len(full_name))
                print(format_stat_table(df, args.stat))

            save_path = args.save
            if save_path == AUTO_SAVE:
                save_path = _auto_filename_single(full_name, args.stat, args.season, args.window)

            plot_stat(df, full_name, args.season, args.window, args.stat, save_path=save_path)

    except ValueError as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
