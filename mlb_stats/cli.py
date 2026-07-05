import argparse

import pandas as pd

from mlb_stats.api import find_player, get_game_log
from mlb_stats.plots import (
    build_stat_dataframe,
    add_rolling_stat,
    format_stat_table,
    plot_stat,
    plot_stat_comparison,
)
from mlb_stats.stats import STAT_CONFIGS, get_stat_config


def _load_stat_dataframe(player_name: str, season: int, stat_key: str, window: int) -> tuple[pd.DataFrame, str]:
    """Look up a player, pull their game log for stat_key, and return the
    rolling-stat DataFrame alongside their resolved full name."""
    config = get_stat_config(stat_key)
    player_id, full_name = find_player(player_name)
    splits = get_game_log(player_id, season, config["group"])
    df = build_stat_dataframe(splits, stat_key)
    df = add_rolling_stat(df, stat_key, window)
    return df, full_name


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mlb-stats",
        description="MLB player stat visualizer"
    )
    parser.add_argument("player", type=str, help='Player name e.g. "Shohei Ohtani"')
    parser.add_argument("player2", type=str, nargs="?", default=None,
                        help='Optional second player to compare against e.g. "Clayton Kershaw"')
    parser.add_argument("--stat", type=str, default="era", choices=sorted(STAT_CONFIGS),
                        help="Stat to plot (default: era)")
    parser.add_argument("--season", type=int, default=2026, help="Season year (default: 2026)")
    parser.add_argument("--window", type=int, default=5, help="Rolling average window (default: 5)")
    parser.add_argument("--save", type=str, default=None, metavar="FILE",
                        help="Save plot to file instead of displaying e.g. ohtani.png")
    parser.add_argument("--table", action="store_true",
                        help="Also print the plotted data as a text table")

    args = parser.parse_args()

    try:
        if args.player2:
            df1, name1 = _load_stat_dataframe(args.player, args.season, args.stat, args.window)
            df2, name2 = _load_stat_dataframe(args.player2, args.season, args.stat, args.window)

            if args.table:
                for name, df in [(name1, df1), (name2, df2)]:
                    print(f"\n{name}")
                    print("-" * len(name))
                    print(format_stat_table(df, args.stat))

            plot_stat_comparison(df1, name1, df2, name2, args.season, args.window, args.stat,
                                  save_path=args.save)
        else:
            df, full_name = _load_stat_dataframe(args.player, args.season, args.stat, args.window)

            if args.table:
                print(f"\n{full_name}")
                print("-" * len(full_name))
                print(format_stat_table(df, args.stat))

            plot_stat(df, full_name, args.season, args.window, args.stat, save_path=args.save)

    except ValueError as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
