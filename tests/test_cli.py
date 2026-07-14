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
