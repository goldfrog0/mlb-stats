"""Data-pipeline tests: parsing raw API values, building DataFrames,
rolling windows, per-game values, and the two special-case stats
(FIP's weights/constant, OPS's composite sum). All expected numbers are
computed by hand from the fixture values in conftest.py."""

import pandas as pd
import pytest

from mlb_stats.plots import (
    _parse_innings_pitched,
    _reindexed_rolling_diff,
    _to_float,
    add_rolling_stat,
    build_stat_dataframe,
    compute_game_value,
    format_stat_table,
)


class TestParsing:
    def test_innings_pitched_is_box_score_notation_not_decimal(self) -> None:
        # "6.2" means 6 innings + 2 outs = 6 2/3, NOT 6.2
        assert _parse_innings_pitched("6.2") == pytest.approx(6 + 2 / 3)
        assert _parse_innings_pitched("6.1") == pytest.approx(6 + 1 / 3)
        assert _parse_innings_pitched("6.0") == 6.0
        assert _parse_innings_pitched("0.1") == pytest.approx(1 / 3)
        assert _parse_innings_pitched(None) == 0.0

    def test_to_float_handles_api_string_formats(self) -> None:
        assert _to_float(".253") == 0.253
        assert _to_float("1.79") == 1.79
        assert _to_float(".---") == 0.0  # API placeholder for undefined rates
        assert _to_float(None) == 0.0


class TestBuildStatDataframe:
    def test_columns_and_cumulative_from_api(self, pitching_splits) -> None:
        df = build_stat_dataframe(pitching_splits, "era")
        assert list(df["opponent"])[0] == "Opponent 1"
        assert list(df["cumulative"]) == [0.0, 0.75, 1.5, 2.25, 3.0, 3.67]

    def test_rows_sorted_by_date_even_if_api_returns_unordered(self, pitching_splits) -> None:
        df = build_stat_dataframe(list(reversed(pitching_splits)), "era")
        assert df["date"].is_monotonic_increasing

    def test_fip_cumulative_is_computed_locally(self, pitching_splits) -> None:
        # No cumulative FIP field exists in the API, so it's derived from
        # cumulative sums. Game 1: (13*0 + 3*(1+0) - 2*7)/6 + 3.10
        df = build_stat_dataframe(pitching_splits, "fip")
        assert df["cumulative"].iloc[0] == pytest.approx((3 - 14) / 6 + 3.10)
        # Games 1-2: (13*1 + 3*(3+0) - 2*13)/12 + 3.10
        assert df["cumulative"].iloc[1] == pytest.approx(-4 / 12 + 3.10)


class TestRolling:
    def test_first_window_minus_one_rows_are_nan(self, pitching_splits) -> None:
        df = add_rolling_stat(build_stat_dataframe(pitching_splits, "era"), "era", window=5)
        assert df["rolling"].iloc[:4].isna().all()
        assert df["rolling"].iloc[4:].notna().all()

    def test_rolling_era_sums_runs_and_innings_not_rates(self, pitching_splits) -> None:
        df = add_rolling_stat(build_stat_dataframe(pitching_splits, "era"), "era", window=5)
        # Games 1-5: 9 * (0+1+2+3+4) / 30 innings
        assert df["rolling"].iloc[4] == pytest.approx(3.0)
        # Games 2-6: 9 * (1+2+3+4+5) / (24 + 6 2/3) innings = 405/92
        assert df["rolling"].iloc[5] == pytest.approx(405 / 92)

    def test_rolling_avg(self, batting_splits) -> None:
        df = add_rolling_stat(build_stat_dataframe(batting_splits, "avg"), "avg", window=5)
        # Games 1-5: (1+2+0+3+1) hits / (4+4+3+5+4) at-bats
        assert df["rolling"].iloc[4] == pytest.approx(7 / 20)


class TestGameValue:
    def test_single_game_era(self, pitching_splits) -> None:
        df = build_stat_dataframe(pitching_splits, "era")
        game = compute_game_value(df, "era")
        assert game.iloc[0] == 0.0
        # Last game: 9 * 5 ER / 6 2/3 IP
        assert game.iloc[5] == pytest.approx(6.75)

    def test_fip_weights_and_constant(self, pitching_splits) -> None:
        df = build_stat_dataframe(pitching_splits, "fip")
        game = compute_game_value(df, "fip")
        # Game 1: (13*0 + 3*(1 BB + 0 HBP) - 2*7 K) / 6 IP + 3.10
        assert game.iloc[0] == pytest.approx((3 - 14) / 6 + 3.10)

    def test_ops_is_sum_of_obp_and_slg(self, batting_splits) -> None:
        df = build_stat_dataframe(batting_splits, "ops")
        game = compute_game_value(df, "ops")
        # Game 1: OBP (1+1+0)/(4+1+0+0) = .400, SLG 1/4 = .250
        assert game.iloc[0] == pytest.approx(0.650)
        obp = compute_game_value(build_stat_dataframe(batting_splits, "obp"), "obp")
        slg = compute_game_value(build_stat_dataframe(batting_splits, "slg"), "slg")
        assert game.iloc[3] == pytest.approx(obp.iloc[3] + slg.iloc[3])


class TestFormatStatTable:
    def test_table_has_game_season_rolling_columns(self, pitching_splits) -> None:
        df = add_rolling_stat(build_stat_dataframe(pitching_splits, "era"), "era", window=5)
        table = format_stat_table(df, "era")
        for expected in ("game_era", "season_era", "rolling_era", "2026-04-01", "Opponent 6"):
            assert expected in table

    def test_warmup_rows_show_nan_rolling(self, pitching_splits) -> None:
        df = add_rolling_stat(build_stat_dataframe(pitching_splits, "era"), "era", window=5)
        first_row = format_stat_table(df, "era").splitlines()[1]
        assert "NaN" in first_row


class TestReindexedRollingDiff:
    """MLB schedules include doubleheaders -- two games on the same
    calendar date -- which produces duplicate "date" values. That broke
    pandas' reindex() (it requires a unique index) the first time this
    was exercised for real, comparing two teams where one had played a
    doubleheader. A player who appeared twice in one day would hit the
    same bug, so this is tested at the shared function, not just for
    teams."""

    def _df(self, dates: list[str], rolling: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"date": pd.to_datetime(dates), "rolling": rolling})

    def test_duplicate_date_does_not_raise(self) -> None:
        df1 = self._df(["2026-04-01", "2026-04-02", "2026-04-02"], [1.0, 2.0, 3.0])
        df2 = self._df(["2026-04-01", "2026-04-02"], [0.5, 0.5])
        _reindexed_rolling_diff(df1, df2)  # must not raise

    def test_duplicate_date_uses_the_later_game(self) -> None:
        # Second game on 04-02 (rolling=3.0) should win over the first
        # (rolling=2.0), since it reflects the later, more complete state.
        df1 = self._df(["2026-04-01", "2026-04-02", "2026-04-02"], [1.0, 2.0, 3.0])
        df2 = self._df(["2026-04-01", "2026-04-02"], [0.5, 0.5])
        result = _reindexed_rolling_diff(df1, df2)
        row = result[result["date"] == pd.Timestamp("2026-04-02")]
        assert row["diff"].iloc[0] == pytest.approx(3.0 - 0.5)

    def test_no_duplicates_still_works(self) -> None:
        df1 = self._df(["2026-04-01", "2026-04-02"], [1.0, 2.0])
        df2 = self._df(["2026-04-01", "2026-04-02"], [0.5, 1.5])
        result = _reindexed_rolling_diff(df1, df2)
        assert list(result["diff"]) == pytest.approx([0.5, 0.5])
