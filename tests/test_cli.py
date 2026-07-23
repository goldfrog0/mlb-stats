"""End-to-end tests for the mlb-stats command: main() runs for real
(argument parsing through chart rendering, on the Agg backend) with only
the two network-facing functions monkeypatched to return fixture data."""

import datetime

import pytest

import mlb_stats.cli as cli


class TestSlugify:
    def test_lowercases_and_hyphenates(self) -> None:
        assert cli._slugify("Shohei Ohtani") == "shohei-ohtani"

    def test_strips_punctuation_runs(self) -> None:
        assert cli._slugify("J.T. Realmuto Jr.") == "j-t-realmuto-jr"


class TestAutoFilenames:
    @pytest.fixture(autouse=True)
    def frozen_today(self, monkeypatch) -> None:
        monkeypatch.setattr(cli, "_today_str", lambda: "2026-07-14")

    def test_single_defaults(self) -> None:
        assert (
            cli._auto_filename_single("Shohei Ohtani", "era", 2026, cli.DEFAULT_WINDOW)
            == "shohei-ohtani_era_2026_2026-07-14.png"
        )

    def test_single_non_default_window_gets_suffix(self) -> None:
        assert (
            cli._auto_filename_single("Shohei Ohtani", "era", 2026, 8)
            == "shohei-ohtani_era_2026_w8_2026-07-14.png"
        )

    def test_compare_all_defaults(self) -> None:
        name = cli._auto_filename_compare(
            "Shohei Ohtani", "Paul Skenes", "era", 2026, cli.DEFAULT_WINDOW,
            layout="overlay", show_cumulative=False, show_diff=False,
        )
        assert name == "shohei-ohtani_vs_paul-skenes_era_2026_2026-07-14.png"

    def test_compare_non_defaults_all_appear_in_order(self) -> None:
        name = cli._auto_filename_compare(
            "Shohei Ohtani", "Paul Skenes", "era", 2026, 10,
            layout="stacked", show_cumulative=True, show_diff=True,
        )
        assert name == "shohei-ohtani_vs_paul-skenes_era_2026_w10_stacked_cumulative_diff_2026-07-14.png"


class TestCurrentYearDefault:
    def test_current_year_is_this_year_as_int(self) -> None:
        assert cli.CURRENT_YEAR == datetime.date.today().year
        assert isinstance(cli.CURRENT_YEAR, int)


@pytest.fixture
def fake_api(monkeypatch, pitching_splits):
    """Patch the CLI's imported network functions to serve fixture data."""
    monkeypatch.setattr(cli, "find_player", lambda name: (660271, "Test Pitcher"))
    monkeypatch.setattr(cli, "get_game_log", lambda pid, season, group: pitching_splits)


class TestMain:
    def test_single_player_saves_chart(self, fake_api, monkeypatch, tmp_path, capsys) -> None:
        out = tmp_path / "chart.png"
        monkeypatch.setattr("sys.argv", ["mlb-stats", "Test Pitcher", "--stat", "era", "--save", str(out)])
        cli.main()
        assert out.exists() and out.stat().st_size > 0
        assert f"Saved to {out}" in capsys.readouterr().out

    def test_comparison_saves_chart(self, fake_api, monkeypatch, tmp_path) -> None:
        out = tmp_path / "compare.png"
        monkeypatch.setattr("sys.argv", [
            "mlb-stats", "Test Pitcher", "Other Pitcher",
            "--stat", "era", "--layout", "stacked", "--diff", "--save", str(out),
        ])
        cli.main()
        assert out.exists() and out.stat().st_size > 0

    def test_table_prints_per_game_data(self, fake_api, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setattr("sys.argv", [
            "mlb-stats", "Test Pitcher", "--stat", "era", "--table", "--save", str(tmp_path / "t.png"),
        ])
        cli.main()
        out = capsys.readouterr().out
        assert "Test Pitcher" in out
        assert "game_era" in out
        assert "Opponent 6" in out

    def test_unknown_player_exits_1_with_message(self, monkeypatch, capsys) -> None:
        def raise_not_found(name):
            raise ValueError(f"No player found for '{name}'")

        monkeypatch.setattr(cli, "find_player", raise_not_found)
        monkeypatch.setattr("sys.argv", ["mlb-stats", "Zzznotaplayer"])
        with pytest.raises(SystemExit) as excinfo:
            cli.main()
        assert excinfo.value.code == 1
        assert "No player found for 'Zzznotaplayer'" in capsys.readouterr().out

    def test_unknown_stat_rejected_by_argparse(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.argv", ["mlb-stats", "Someone", "--stat", "nope"])
        with pytest.raises(SystemExit) as excinfo:
            cli.main()
        assert excinfo.value.code == 2  # argparse usage error

    def test_team_win_pct_saves_chart(self, monkeypatch, tmp_path, team_schedule_games) -> None:
        # A "team" group stat should route through find_team/get_team_schedule
        # instead of find_player/get_game_log, transparently to argparse --
        # same positional argument either way.
        monkeypatch.setattr(cli, "find_team", lambda name: (119, "Los Angeles Dodgers"))
        monkeypatch.setattr(cli, "get_team_schedule", lambda team_id, season: team_schedule_games)
        out = tmp_path / "team.png"
        monkeypatch.setattr("sys.argv", [
            "mlb-stats", "Los Angeles Dodgers", "--stat", "win_pct", "--table", "--save", str(out),
        ])
        cli.main()
        assert out.exists() and out.stat().st_size > 0

    def test_unknown_team_exits_1_with_message(self, monkeypatch, capsys) -> None:
        def raise_not_found(name):
            raise ValueError(f"No team found for '{name}'")

        monkeypatch.setattr(cli, "find_team", raise_not_found)
        monkeypatch.setattr("sys.argv", ["mlb-stats", "Zzznotateam", "--stat", "win_pct"])
        with pytest.raises(SystemExit) as excinfo:
            cli.main()
        assert excinfo.value.code == 1
        assert "No team found for 'Zzznotateam'" in capsys.readouterr().out

    def test_standings_saves_chart_and_prints_table(self, monkeypatch, tmp_path, division_team_records, capsys) -> None:
        monkeypatch.setattr(cli, "find_division", lambda name: (201, "American League East"))
        monkeypatch.setattr(cli, "get_division_standings", lambda division_id, season: division_team_records)
        out = tmp_path / "standings.png"
        monkeypatch.setattr("sys.argv", ["mlb-stats", "--standings", "AL East", "--table", "--save", str(out)])
        cli.main()
        assert out.exists() and out.stat().st_size > 0
        printed = capsys.readouterr().out
        assert "American League East" in printed
        assert "Rays" in printed

    def test_bwar_saves_chart_and_prints_table(
        self, monkeypatch, tmp_path, batting_splits, capsys,
    ) -> None:
        # League totals shaped like team season blobs; exact values don't
        # matter here, only that the pipeline runs them through league_woba.
        league = [{"atBats": 100, "hits": 25, "doubles": 5, "triples": 1, "homeRuns": 3,
                   "baseOnBalls": 10, "intentionalWalks": 1, "hitByPitch": 2, "sacFlies": 2}]
        monkeypatch.setattr(cli, "find_player", lambda name: (660271, "Test Batter"))
        monkeypatch.setattr(cli, "get_game_log", lambda pid, season, group: batting_splits)
        monkeypatch.setattr(cli, "get_league_team_stats", lambda season, group: league)
        monkeypatch.setattr(cli, "get_primary_position", lambda pid: "DH")
        out = tmp_path / "bwar.png"
        monkeypatch.setattr("sys.argv", ["mlb-stats", "Test Batter", "--stat", "bwar", "--table", "--save", str(out)])
        cli.main()
        assert out.exists() and out.stat().st_size > 0
        assert "game_bwar" in capsys.readouterr().out

    def test_pwar_saves_chart(self, fake_api, monkeypatch, tmp_path) -> None:
        league = [{"homeRuns": 20, "baseOnBalls": 60, "hitBatsmen": 10, "strikeOuts": 200,
                   "inningsPitched": "300.0"}]
        monkeypatch.setattr(cli, "get_league_team_stats", lambda season, group: league)
        out = tmp_path / "pwar.png"
        monkeypatch.setattr("sys.argv", ["mlb-stats", "Test Pitcher", "--stat", "pwar", "--save", str(out)])
        cli.main()
        assert out.exists() and out.stat().st_size > 0

    def test_velo_saves_chart_and_prints_summary_table(
        self, monkeypatch, tmp_path, velo_game_splits, game_pitches_by_pk, capsys,
    ) -> None:
        monkeypatch.setattr(cli, "find_player", lambda name: (694973, "Test Pitcher"))
        monkeypatch.setattr(cli, "get_game_log", lambda pid, season, group: velo_game_splits)
        monkeypatch.setattr(cli, "get_game_pitches", lambda pk: game_pitches_by_pk[pk])
        out = tmp_path / "velo.png"
        monkeypatch.setattr("sys.argv", ["mlb-stats", "Test Pitcher", "--velo", "--table", "--save", str(out)])
        cli.main()
        assert out.exists() and out.stat().st_size > 0
        printed = capsys.readouterr().out
        assert "max_velo" in printed
        assert "Opponent A" in printed

    def test_velo_date_range_narrows_the_games(
        self, monkeypatch, tmp_path, velo_game_splits, game_pitches_by_pk, capsys,
    ) -> None:
        monkeypatch.setattr(cli, "find_player", lambda name: (694973, "Test Pitcher"))
        monkeypatch.setattr(cli, "get_game_log", lambda pid, season, group: velo_game_splits)
        monkeypatch.setattr(cli, "get_game_pitches", lambda pk: game_pitches_by_pk[pk])
        monkeypatch.setattr("sys.argv", [
            "mlb-stats", "Test Pitcher", "--velo", "--start-date", "2026-06-02",
            "--table", "--save", str(tmp_path / "v.png"),
        ])
        cli.main()
        printed = capsys.readouterr().out
        assert "Opponent B" in printed
        assert "Opponent A" not in printed

    def test_date_range_without_velo_is_a_usage_error(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.argv", ["mlb-stats", "Someone", "--start-date", "2026-06-01"])
        with pytest.raises(SystemExit) as excinfo:
            cli.main()
        assert excinfo.value.code == 2

    def test_velo_comparison_saves_chart_and_prints_both_tables(
        self, monkeypatch, tmp_path, velo_game_splits, game_pitches_by_pk, capsys,
    ) -> None:
        names = iter(["Pitcher One", "Pitcher Two"])
        monkeypatch.setattr(cli, "find_player", lambda name: (694973, next(names)))
        monkeypatch.setattr(cli, "get_game_log", lambda pid, season, group: velo_game_splits)
        monkeypatch.setattr(cli, "get_game_pitches", lambda pk: game_pitches_by_pk[pk])
        out = tmp_path / "cmp.png"
        monkeypatch.setattr("sys.argv", [
            "mlb-stats", "One", "Two", "--velo", "--pitch-type", "fastball", "--table", "--save", str(out),
        ])
        cli.main()
        assert out.exists() and out.stat().st_size > 0
        printed = capsys.readouterr().out
        assert "Pitcher One" in printed and "Pitcher Two" in printed
        assert "avg_velo" in printed

    def test_velo_comparison_bad_layout_is_a_usage_error(self, monkeypatch) -> None:
        # chefs-special is a stat-comparison layout; velo doesn't allow it.
        monkeypatch.setattr("sys.argv", ["mlb-stats", "One", "Two", "--velo", "--layout", "chefs-special"])
        with pytest.raises(SystemExit) as excinfo:
            cli.main()
        assert excinfo.value.code == 2

    def test_pitch_type_without_velo_is_a_usage_error(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.argv", ["mlb-stats", "Someone", "--pitch-type", "fastball"])
        with pytest.raises(SystemExit) as excinfo:
            cli.main()
        assert excinfo.value.code == 2

    @pytest.mark.parametrize("box", ["game", "type"])
    def test_velo_box_saves_chart(
        self, monkeypatch, tmp_path, velo_game_splits, game_pitches_by_pk, box,
    ) -> None:
        monkeypatch.setattr(cli, "find_player", lambda name: (694973, "Test Pitcher"))
        monkeypatch.setattr(cli, "get_game_log", lambda pid, season, group: velo_game_splits)
        monkeypatch.setattr(cli, "get_game_pitches", lambda pk: game_pitches_by_pk[pk])
        out = tmp_path / f"box_{box}.png"
        monkeypatch.setattr("sys.argv", ["mlb-stats", "Test Pitcher", "--velo", "--box", box, "--save", str(out)])
        cli.main()
        assert out.exists() and out.stat().st_size > 0

    def test_box_without_velo_is_a_usage_error(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.argv", ["mlb-stats", "Someone", "--box", "game"])
        with pytest.raises(SystemExit) as excinfo:
            cli.main()
        assert excinfo.value.code == 2

    def test_box_with_comparison_is_a_usage_error(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.argv", ["mlb-stats", "One", "Two", "--velo", "--box", "game"])
        with pytest.raises(SystemExit) as excinfo:
            cli.main()
        assert excinfo.value.code == 2

    def test_no_player_and_no_standings_is_a_usage_error(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.argv", ["mlb-stats"])
        with pytest.raises(SystemExit) as excinfo:
            cli.main()
        assert excinfo.value.code == 2

    def test_unknown_division_exits_1_with_message(self, monkeypatch, capsys) -> None:
        def raise_not_found(name):
            raise ValueError(f"No division found for '{name}'")

        monkeypatch.setattr(cli, "find_division", raise_not_found)
        monkeypatch.setattr("sys.argv", ["mlb-stats", "--standings", "Zzz"])
        with pytest.raises(SystemExit) as excinfo:
            cli.main()
        assert excinfo.value.code == 1
        assert "No division found for 'Zzz'" in capsys.readouterr().out
