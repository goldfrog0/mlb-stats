"""Tests for the career-WAR feature: fetching a season's WAR from the
API's sabermetrics stats, resolving a player's debut year, and shaping
per-season values into the career DataFrame behind the chart. Uses
career_war_seasons from conftest.py."""

import pytest

import mlb_stats.api as api
from mlb_stats.plots import build_war_dataframe, format_war_table


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self._payload


class TestGetSeasonWar:
    @pytest.fixture(autouse=True)
    def clean_cache(self):
        api.get_season_war.cache_clear()
        yield
        api.get_season_war.cache_clear()

    def _patch(self, monkeypatch, payload):
        monkeypatch.setattr(api.requests, "get", lambda url, params=None: FakeResponse(payload))

    def test_returns_war_value(self, monkeypatch) -> None:
        self._patch(monkeypatch, {"stats": [{"splits": [{"stat": {"war": 3.14608, "woba": 0.4}}]}]})
        assert api.get_season_war(660271, 2026, "hitting") == pytest.approx(3.14608)

    def test_no_splits_is_none_not_an_error(self, monkeypatch) -> None:
        # A position player's pitching WAR, or a season not played.
        self._patch(monkeypatch, {"stats": [{"splits": []}]})
        assert api.get_season_war(660271, 2026, "pitching") is None

    def test_split_without_war_key_is_none(self, monkeypatch) -> None:
        self._patch(monkeypatch, {"stats": [{"splits": [{"stat": {"woba": 0.4}}]}]})
        assert api.get_season_war(660271, 2026, "hitting") is None


class TestGetDebutYear:
    @pytest.fixture(autouse=True)
    def clean_cache(self):
        api.get_debut_year.cache_clear()
        yield
        api.get_debut_year.cache_clear()

    def test_year_extracted_from_debut_date(self, monkeypatch) -> None:
        payload = {"people": [{"id": 660271, "mlbDebutDate": "2018-03-29"}]}
        monkeypatch.setattr(api.requests, "get", lambda url, params=None: FakeResponse(payload))
        assert api.get_debut_year(660271) == 2018

    def test_missing_debut_date_raises(self, monkeypatch) -> None:
        payload = {"people": [{"id": 123}]}
        monkeypatch.setattr(api.requests, "get", lambda url, params=None: FakeResponse(payload))
        with pytest.raises(ValueError, match="No MLB debut date found for player ID 123"):
            api.get_debut_year(123)


class TestBuildWarDataframe:
    def test_missed_seasons_dropped_not_zeroed(self, career_war_seasons) -> None:
        df = build_war_dataframe(career_war_seasons)
        # 2020 (both components None) must not appear at all.
        assert list(df["season"]) == [2018, 2019, 2021]

    def test_missing_component_counts_as_zero(self, career_war_seasons) -> None:
        df = build_war_dataframe(career_war_seasons)
        row_2019 = df[df["season"] == 2019].iloc[0]
        assert row_2019["pitching"] == 0.0
        assert row_2019["total"] == pytest.approx(1.6)

    def test_total_sums_components_including_mixed_signs(self, career_war_seasons) -> None:
        df = build_war_dataframe(career_war_seasons)
        assert list(df["total"]) == pytest.approx([3.8, 1.6, 1.5])

    def test_survives_unsorted_input(self, career_war_seasons) -> None:
        df = build_war_dataframe(list(reversed(career_war_seasons)))
        assert list(df["season"]) == [2018, 2019, 2021]

    def test_no_data_at_all_raises(self) -> None:
        seasons = [{"season": 2026, "batting": None, "pitching": None}]
        with pytest.raises(ValueError, match="No WAR data found in any season"):
            build_war_dataframe(seasons)


class TestFormatWarTable:
    def test_career_totals_row(self, career_war_seasons) -> None:
        table = format_war_table(build_war_dataframe(career_war_seasons))
        lines = table.splitlines()
        assert len(lines) == 5  # header + 3 seasons + career row
        for expected in ("Season", "Batting", "Pitching", "Total"):
            assert expected in lines[0]
        assert "Career" in lines[-1]
        for value in ("3.8", "3.1", "6.9"):
            assert value in lines[-1]
