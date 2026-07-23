"""Tests for the pitch-velocity feature: flattening a game's play-by-play
feed into pitches, date-range filtering of game-log splits, and shaping
(split, pitches) pairs into the per-pitch DataFrame behind the chart.
Uses velo_game_splits/game_pitches_by_pk from conftest.py."""

import pytest

import mlb_stats.api as api
from mlb_stats.plots import (
    build_pitch_dataframe,
    filter_pitch_type,
    filter_splits_by_date,
    format_pitch_comparison_table,
    format_pitch_table,
    pitch_velocity_by_game,
    plot_pitch_velocities,
    plot_pitch_velocity_comparison,
)

# Must match conftest.py's PITCHER_ID (the game_pitches_by_pk fixture
# attributes its pitches to this id). Duplicated rather than imported:
# `from tests.conftest import ...` only resolves when the repo root is
# on sys.path, which `python -m pytest` guarantees but the bare pytest
# binary (used in CI) does not.
PITCHER_ID = 694973


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self._payload


FEED_PAYLOAD = {
    "liveData": {"plays": {"allPlays": [
        {
            "matchup": {"pitcher": {"id": PITCHER_ID}},
            "playEvents": [
                {"isPitch": True,
                 "details": {"type": {"description": "Four-Seam Fastball"}},
                 "pitchData": {"startSpeed": 97.1}},
                # A non-pitch event (pickoff attempt, mound visit, ...)
                {"isPitch": False, "details": {}},
                # A pitch the tracking system missed: no type, no velocity
                {"isPitch": True, "details": {}, "pitchData": {}},
            ],
        },
        {
            "matchup": {"pitcher": {"id": 999}},
            "playEvents": [
                {"isPitch": True,
                 "details": {"type": {"description": "Slider"}},
                 "pitchData": {"startSpeed": 86.0}},
            ],
        },
    ]}},
}


class TestGetGamePitches:
    @pytest.fixture(autouse=True)
    def clean_cache(self):
        api.get_game_pitches.cache_clear()
        yield
        api.get_game_pitches.cache_clear()

    @pytest.fixture(autouse=True)
    def fake_feed_endpoint(self, monkeypatch):
        monkeypatch.setattr(api.requests, "get", lambda url, params=None: FakeResponse(FEED_PAYLOAD))

    def test_flattens_every_pitch_from_every_pitcher(self) -> None:
        pitches = api.get_game_pitches(123)
        assert pitches == [
            {"pitcher_id": PITCHER_ID, "pitch_type": "Four-Seam Fastball", "velo": 97.1},
            {"pitcher_id": PITCHER_ID, "pitch_type": "Unknown", "velo": None},
            {"pitcher_id": 999, "pitch_type": "Slider", "velo": 86.0},
        ]

    def test_non_pitch_events_are_skipped(self) -> None:
        # 4 play events in the feed, but only 3 are pitches.
        assert len(api.get_game_pitches(123)) == 3


class TestFilterSplitsByDate:
    def test_no_bounds_keeps_everything(self, velo_game_splits) -> None:
        assert filter_splits_by_date(velo_game_splits) == velo_game_splits

    def test_bounds_are_inclusive(self, velo_game_splits) -> None:
        kept = filter_splits_by_date(velo_game_splits, "2026-06-01", "2026-06-06")
        assert len(kept) == 2

    def test_start_only(self, velo_game_splits) -> None:
        kept = filter_splits_by_date(velo_game_splits, start_date="2026-06-02")
        assert [s["date"] for s in kept] == ["2026-06-06"]

    def test_end_only(self, velo_game_splits) -> None:
        kept = filter_splits_by_date(velo_game_splits, end_date="2026-06-05")
        assert [s["date"] for s in kept] == ["2026-06-01"]

    def test_empty_range_raises(self, velo_game_splits) -> None:
        with pytest.raises(ValueError, match="No games found between 2027-01-01 and season end"):
            filter_splits_by_date(velo_game_splits, start_date="2027-01-01")


class TestBuildPitchDataframe:
    @pytest.fixture
    def games(self, velo_game_splits, game_pitches_by_pk):
        return [(s, game_pitches_by_pk[s["game"]["gamePk"]]) for s in velo_game_splits]

    def test_one_row_per_tracked_pitch_by_the_pitcher(self, games) -> None:
        df = build_pitch_dataframe(games, PITCHER_ID)
        # 7 fixture pitches, minus one by another pitcher, minus one untracked.
        assert len(df) == 5
        assert set(df.columns) == {"date", "opponent", "pitch_type", "velo"}

    def test_carries_game_date_and_opponent_onto_each_pitch(self, games) -> None:
        df = build_pitch_dataframe(games, PITCHER_ID)
        assert list(df["date"].unique()) == ["2026-06-01", "2026-06-06"]
        assert list(df[df["date"] == "2026-06-06"]["opponent"].unique()) == ["Opponent B"]

    def test_velocities_preserved(self, games) -> None:
        df = build_pitch_dataframe(games, PITCHER_ID)
        assert list(df[df["date"] == "2026-06-01"]["velo"]) == [97.0, 95.0, 85.0]

    def test_no_pitches_by_this_pitcher_raises(self, games) -> None:
        with pytest.raises(ValueError, match="No pitch data found for player ID 12345"):
            build_pitch_dataframe(games, 12345)


class TestFormatPitchTable:
    def test_one_summary_row_per_game(self, velo_game_splits, game_pitches_by_pk) -> None:
        games = [(s, game_pitches_by_pk[s["game"]["gamePk"]]) for s in velo_game_splits]
        table = format_pitch_table(build_pitch_dataframe(games, PITCHER_ID))

        lines = table.splitlines()
        assert len(lines) == 3  # header + one row per game
        for expected in ("date", "opponent", "pitches", "max_velo", "median_velo", "min_velo"):
            assert expected in lines[0]
        # Game 111: 3 pitches at 97/95/85 -> max 97, median 95, min 85.
        assert "Opponent A" in lines[1]
        for value in ("3", "97.0", "95.0", "85.0"):
            assert value in lines[1]
        # Game 222: 2 pitches at 98.5/84 -> max 98.5, min 84.
        assert "Opponent B" in lines[2]
        for value in ("2", "98.5", "84.0"):
            assert value in lines[2]


class TestFilterPitchType:
    @pytest.fixture
    def df(self, velo_game_splits, game_pitches_by_pk):
        games = [(s, game_pitches_by_pk[s["game"]["gamePk"]]) for s in velo_game_splits]
        return build_pitch_dataframe(games, PITCHER_ID)

    def test_case_insensitive_substring_match(self, df) -> None:
        # Fixture fastballs are "Four-Seam Fastball"; each of these substrings selects them.
        for query in ("Four-Seam Fastball", "four-seam", "fastball", "SEAM"):
            assert list(filter_pitch_type(df, query)["velo"]) == [97.0, 95.0, 98.5]

    def test_non_matching_type_is_empty(self, df) -> None:
        assert filter_pitch_type(df, "knuckleball").empty


class TestPitchVelocityByGame:
    @pytest.fixture
    def df(self, velo_game_splits, game_pitches_by_pk):
        games = [(s, game_pitches_by_pk[s["game"]["gamePk"]]) for s in velo_game_splits]
        return build_pitch_dataframe(games, PITCHER_ID)

    def test_per_game_aggregates_for_one_type(self, df) -> None:
        by_game = pitch_velocity_by_game(df, "Four-Seam Fastball", "Test Pitcher")
        assert list(by_game["date"]) == ["2026-06-01", "2026-06-06"]
        # Game 1: two fastballs 97/95 -> count 2, mean 96, min 95, max 97.
        first = by_game.iloc[0]
        assert (first["count"], first["mean"], first["min"], first["max"]) == (2, 96.0, 95.0, 97.0)
        # Game 2: one fastball 98.5.
        second = by_game.iloc[1]
        assert (second["count"], second["mean"], second["min"], second["max"]) == (1, 98.5, 98.5, 98.5)

    def test_excludes_other_pitch_types(self, df) -> None:
        # The slider (85) and curveball (84) must not pull the fastball mean down.
        by_game = pitch_velocity_by_game(df, "fastball", "Test Pitcher")
        assert by_game["mean"].min() >= 96.0

    def test_no_pitches_of_type_raises(self, df) -> None:
        with pytest.raises(ValueError, match="No Splitter pitches found for Test Pitcher"):
            pitch_velocity_by_game(df, "Splitter", "Test Pitcher")


class TestFormatPitchComparisonTable:
    def test_per_game_rows_with_avg(self, velo_game_splits, game_pitches_by_pk) -> None:
        games = [(s, game_pitches_by_pk[s["game"]["gamePk"]]) for s in velo_game_splits]
        by_game = pitch_velocity_by_game(build_pitch_dataframe(games, PITCHER_ID),
                                         "Four-Seam Fastball", "Test Pitcher")
        table = format_pitch_comparison_table(by_game, "Four-Seam Fastball")
        lines = table.splitlines()
        assert len(lines) == 3  # header + two games
        assert "avg_velo" in lines[0]
        assert "96.0" in lines[1]  # game 1 fastball average


class TestPlotPitchVelocityComparison:
    @pytest.fixture
    def two_pitchers(self, velo_game_splits, game_pitches_by_pk):
        games = [(s, game_pitches_by_pk[s["game"]["gamePk"]]) for s in velo_game_splits]
        df = build_pitch_dataframe(games, PITCHER_ID)
        by_game = pitch_velocity_by_game(df, "Four-Seam Fastball", "A")
        return by_game

    @pytest.mark.parametrize("layout", ["stacked", "side-by-side", "overlay"])
    def test_renders_each_layout(self, two_pitchers, tmp_path, layout) -> None:
        out = tmp_path / f"{layout}.png"
        plot_pitch_velocity_comparison(two_pitchers, "A", two_pitchers, "B", 2026,
                                       "Four-Seam Fastball", layout=layout, save_path=str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_unknown_layout_raises(self, two_pitchers) -> None:
        with pytest.raises(ValueError, match="Unknown velo layout 'chefs-special'"):
            plot_pitch_velocity_comparison(two_pitchers, "A", two_pitchers, "B", 2026,
                                           "Four-Seam Fastball", layout="chefs-special")


class TestPlotPitchVelocities:
    @pytest.fixture
    def df(self, velo_game_splits, game_pitches_by_pk):
        games = [(s, game_pitches_by_pk[s["game"]["gamePk"]]) for s in velo_game_splits]
        return build_pitch_dataframe(games, PITCHER_ID)

    @pytest.mark.parametrize("box", [None, "game", "type"])
    def test_renders_each_box_style(self, df, tmp_path, box) -> None:
        out = tmp_path / f"velo_{box}.png"
        plot_pitch_velocities(df, "Test Pitcher", 2026, save_path=str(out), box=box)
        assert out.exists() and out.stat().st_size > 0
