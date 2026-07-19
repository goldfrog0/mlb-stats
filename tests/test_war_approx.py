"""Tests for approximate per-game WAR (mlb_stats/war.py): league
baselines aggregated from team totals, per-game batting/pitching WAR
values (hand-computed expectations), the positional adjustment, and the
rolling-SUM semantics these configs get from add_rolling_stat. (The
group=batting API regression this feature surfaced is covered in
tests/test_api_groups.py, which can live on master without war.py.)"""

import pytest

from mlb_stats.plots import add_rolling_stat, compute_game_value
from mlb_stats.war import (
    build_war_approx_dataframe,
    league_fip,
    league_woba,
    position_adjustment,
)


class TestLeagueBaselines:
    def test_league_woba_aggregates_before_dividing(self) -> None:
        # Two teams, hand-computed with the module's weights:
        # A: uBB 8, 1B 20, 2B 5, 3B 1, HR 4, HBP 2 -> num 40.652, den 113
        # B: uBB 5, 1B 7, 2B 2, HR 1, HBP 1        -> num 14.897, den 57
        # league = 55.549 / 170
        teams = [
            {"atBats": 100, "hits": 30, "doubles": 5, "triples": 1, "homeRuns": 4,
             "baseOnBalls": 10, "intentionalWalks": 2, "hitByPitch": 2, "sacFlies": 3},
            {"atBats": 50, "hits": 10, "doubles": 2, "triples": 0, "homeRuns": 1,
             "baseOnBalls": 5, "intentionalWalks": 0, "hitByPitch": 1, "sacFlies": 1},
        ]
        assert league_woba(teams) == pytest.approx(55.549 / 170)

    def test_league_fip_aggregates_before_dividing(self) -> None:
        # A: 13*20 + 3*(60+10) - 2*200 = 70 over 300 IP
        # B: 13*10 + 3*(30+5) - 2*150 = -65 over 150.1 (= 150 1/3) IP
        # league = 5 / 450⅓ + 3.10
        teams = [
            {"homeRuns": 20, "baseOnBalls": 60, "hitBatsmen": 10, "strikeOuts": 200,
             "inningsPitched": "300.0"},
            {"homeRuns": 10, "baseOnBalls": 30, "hitBatsmen": 5, "strikeOuts": 150,
             "inningsPitched": "150.1"},
        ]
        assert league_fip(teams) == pytest.approx(5 / (450 + 1 / 3) + 3.10)


class TestPositionAdjustment:
    def test_known_positions(self) -> None:
        assert position_adjustment("SS") == 7.5
        assert position_adjustment("DH") == -17.5
        assert position_adjustment("TWP") == -17.5  # two-way player bats as DH

    def test_unknown_position_is_neutral(self) -> None:
        assert position_adjustment("P") == 0.0
        assert position_adjustment("") == 0.0


class TestBattingWar:
    def test_game_value_hand_computed(self, batting_splits) -> None:
        # Game 1: 1 single + 1 uBB, wOBA den 5 -> wOBA (.689 + .882)/5.
        # wRAA = (.3142 - .310)/1.24 * 5 PA; replacement = 20*5/600;
        # no positional adjustment -> war = .01836.
        df = build_war_approx_dataframe(batting_splits, "batting", 0.310, pos_adj_per_600=0.0)
        assert df["war_game"].iloc[0] == pytest.approx(0.0183602, abs=1e-6)

    def test_positional_adjustment_shifts_every_game(self, batting_splits) -> None:
        # DH at -17.5/600 over 5 PA shifts game 1 by -0.1458333 runs
        # = -0.01458 WAR.
        neutral = build_war_approx_dataframe(batting_splits, "batting", 0.310, 0.0)
        dh = build_war_approx_dataframe(batting_splits, "batting", 0.310, -17.5)
        assert dh["war_game"].iloc[0] - neutral["war_game"].iloc[0] == pytest.approx(-0.0145833, abs=1e-6)

    def test_big_game_counts_more(self, batting_splits) -> None:
        # Game 4 (single + double + homer + HBP) should dwarf game 3 (0-for-3 + walk).
        df = build_war_approx_dataframe(batting_splits, "batting", 0.310, 0.0)
        assert df["war_game"].iloc[3] == pytest.approx(0.2656452, abs=1e-6)
        assert df["war_game"].iloc[3] > df["war_game"].iloc[2]

    def test_cumulative_is_running_sum(self, batting_splits) -> None:
        df = build_war_approx_dataframe(batting_splits, "batting", 0.310, 0.0)
        assert df["cumulative"].iloc[-1] == pytest.approx(df["war_game"].sum())


class TestPitchingWar:
    def test_game_value_hand_computed(self, pitching_splits) -> None:
        # Game 1: FIP numerator 3*1 - 2*7 = -11 over 6 IP -> game FIP 1.2667.
        # ((4.20 - 1.2667)/10 + 0.12) * 6/9 = .27556.
        df = build_war_approx_dataframe(pitching_splits, "pitching", 4.20)
        assert df["war_game"].iloc[0] == pytest.approx(0.2755556, abs=1e-6)

    def test_worse_fip_earns_less(self, pitching_splits) -> None:
        # Fixture games get progressively worse (more ER/HR); game 6
        # (1 HR, 2 BB, 1 HBP, 6 K over 6.2 IP) is worth less than game 1.
        df = build_war_approx_dataframe(pitching_splits, "pitching", 4.20)
        assert df["war_game"].iloc[5] < df["war_game"].iloc[0]


class TestRollingSemantics:
    def test_rolling_is_a_sum_not_a_rate(self, batting_splits) -> None:
        df = build_war_approx_dataframe(batting_splits, "batting", 0.310, 0.0)
        df = add_rolling_stat(df, "bwar", window=2)
        assert df["rolling"].iloc[0] != df["rolling"].iloc[0]  # NaN before the window fills
        expected = df["war_game"].iloc[0] + df["war_game"].iloc[1]
        assert df["rolling"].iloc[1] == pytest.approx(expected)

    def test_compute_game_value_passthrough(self, batting_splits) -> None:
        df = build_war_approx_dataframe(batting_splits, "batting", 0.310, 0.0)
        assert compute_game_value(df, "bwar").equals(df["war_game"])
