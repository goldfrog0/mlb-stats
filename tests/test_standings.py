"""Tests for the standings feature: division lookup, fetching a
division's standings, and shaping them into a display-ready DataFrame.
Uses division_team_records from conftest.py (4 teams, already
rank-sorted, matching real AL East data verified against the live API:
Rays 56-38 .596, Yankees 54-42 .563, Red Sox 46-48 .489,
Blue Jays 45-51 .469)."""

import pytest

import mlb_stats.api as api
from mlb_stats.plots import build_standings_dataframe, format_standings_table


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self._payload


ALL_TEAMS_WITH_DIVISIONS = {
    "teams": [
        {"id": 139, "name": "Tampa Bay Rays", "division": {"id": 201, "name": "American League East"}},
        {"id": 147, "name": "New York Yankees", "division": {"id": 201, "name": "American League East"}},
        {"id": 108, "name": "Los Angeles Angels", "division": {"id": 200, "name": "American League West"}},
        {"id": 119, "name": "Los Angeles Dodgers", "division": {"id": 203, "name": "National League West"}},
        {"id": 121, "name": "New York Mets", "division": {"id": 204, "name": "National League East"}},
    ],
}


class TestFindDivision:
    @pytest.fixture(autouse=True)
    def clean_cache(self):
        api._all_teams.cache_clear()
        yield
        api._all_teams.cache_clear()

    @pytest.fixture(autouse=True)
    def fake_teams_endpoint(self, monkeypatch):
        monkeypatch.setattr(api.requests, "get", lambda url, params=None: FakeResponse(ALL_TEAMS_WITH_DIVISIONS))

    def test_al_alias_expands(self) -> None:
        assert api.find_division("AL East") == (201, "American League East")

    def test_nl_alias_expands(self) -> None:
        assert api.find_division("NL West") == (203, "National League West")

    def test_full_name_matches_unchanged(self) -> None:
        assert api.find_division("National League East") == (204, "National League East")

    def test_case_insensitive(self) -> None:
        assert api.find_division("al east") == (201, "American League East")

    def test_no_match_raises(self) -> None:
        with pytest.raises(ValueError, match="No division found for 'Zzz'"):
            api.find_division("Zzz")

    def test_ambiguous_match_picks_first_and_warns(self, capsys) -> None:
        # "West" matches both AL West and NL West.
        division_id, name = api.find_division("West")
        assert (division_id, name) in [(200, "American League West"), (203, "National League West")]
        assert "Multiple matches" in capsys.readouterr().out


class TestGetDivisionStandings:
    @pytest.fixture(autouse=True)
    def clean_cache(self):
        api.get_division_standings.cache_clear()
        yield
        api.get_division_standings.cache_clear()

    def test_returns_matching_division_records(self, monkeypatch, division_team_records) -> None:
        payload = {"records": [
            {"division": {"id": 202}, "teamRecords": [{"placeholder": "wrong division"}]},
            {"division": {"id": 201}, "teamRecords": division_team_records},
        ]}
        monkeypatch.setattr(api.requests, "get", lambda url, params=None: FakeResponse(payload))
        result = api.get_division_standings(201, 2026)
        assert result == division_team_records

    def test_division_not_in_response_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(api.requests, "get", lambda url, params=None: FakeResponse({"records": []}))
        with pytest.raises(ValueError, match="No standings found for division ID 201"):
            api.get_division_standings(201, 2026)


class TestBuildStandingsDataframe:
    def test_sorted_best_to_worst(self, division_team_records) -> None:
        df = build_standings_dataframe(division_team_records)
        assert list(df["team"]) == ["Rays", "Yankees", "Red Sox", "Blue Jays"]
        assert list(df["rank"]) == [1, 2, 3, 4]

    def test_record_and_pct(self, division_team_records) -> None:
        df = build_standings_dataframe(division_team_records)
        leader = df.iloc[0]
        assert (leader["wins"], leader["losses"]) == (56, 38)
        assert leader["pct"] == pytest.approx(0.596)

    def test_games_back_and_streak(self, division_team_records) -> None:
        df = build_standings_dataframe(division_team_records)
        assert df.iloc[0]["games_back"] == "-"
        assert df.iloc[1]["games_back"] == "3.0"
        assert df.iloc[0]["streak"] == "L1"

    def test_survives_unsorted_input(self, division_team_records) -> None:
        # The API already returns rank-sorted records, but don't rely on
        # callers preserving that order.
        shuffled = [division_team_records[2], division_team_records[0], division_team_records[3],
                    division_team_records[1]]
        df = build_standings_dataframe(shuffled)
        assert list(df["rank"]) == [1, 2, 3, 4]


class TestFormatStandingsTable:
    def test_contains_expected_columns_and_values(self, division_team_records) -> None:
        table = format_standings_table(build_standings_dataframe(division_team_records))
        for expected in ("Rank", "Team", "W", "L", "PCT", "GB", "Streak", "Rays", "0.596", "L1"):
            assert expected in table
