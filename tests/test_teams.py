"""Tests for the team-level features: team lookup, schedule fetching,
and flattening a schedule into the same DataFrame shape build_stat_dataframe
produces (so win_pct rolls up through the same generic code as any
player stat). Uses team_schedule_games from conftest.py, which produces
the win/loss sequence W L W W L L (3-3) across 6 completed games plus
one filtered-out future game."""

import pytest

import mlb_stats.api as api
from mlb_stats.plots import add_rolling_stat, build_team_win_dataframe, compute_game_value


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self._payload


ALL_TEAMS_PAYLOAD = {
    "teams": [
        {
            "id": 119, "name": "Los Angeles Dodgers", "teamName": "Dodgers",
            "locationName": "Los Angeles", "abbreviation": "LAD",
        },
        {
            "id": 137, "name": "San Francisco Giants", "teamName": "Giants",
            "locationName": "San Francisco", "abbreviation": "SF",
        },
        {
            "id": 147, "name": "New York Yankees", "teamName": "Yankees",
            "locationName": "New York", "abbreviation": "NYY",
        },
        {
            "id": 121, "name": "New York Mets", "teamName": "Mets",
            "locationName": "New York", "abbreviation": "NYM",
        },
    ],
}


class TestFindTeam:
    @pytest.fixture(autouse=True)
    def clean_cache(self):
        api._all_teams.cache_clear()
        yield
        api._all_teams.cache_clear()

    @pytest.fixture(autouse=True)
    def fake_teams_endpoint(self, monkeypatch):
        monkeypatch.setattr(api.requests, "get", lambda url, params=None: FakeResponse(ALL_TEAMS_PAYLOAD))

    def test_matches_by_partial_full_name(self) -> None:
        assert api.find_team("dodgers") == (119, "Los Angeles Dodgers")

    def test_matches_by_city(self) -> None:
        assert api.find_team("san francisco") == (137, "San Francisco Giants")

    def test_matches_by_abbreviation_exact(self) -> None:
        assert api.find_team("SF") == (137, "San Francisco Giants")

    def test_case_insensitive(self) -> None:
        assert api.find_team("DODGERS") == (119, "Los Angeles Dodgers")

    def test_no_match_raises(self) -> None:
        with pytest.raises(ValueError, match="No team found for 'Zzznotateam'"):
            api.find_team("Zzznotateam")

    def test_ambiguous_match_picks_first(self, capsys) -> None:
        # Both NY teams match "new york" -- takes the first, matching
        # find_player's "multiple matches" behavior.
        team_id, name = api.find_team("new york")
        assert (team_id, name) in [(147, "New York Yankees"), (121, "New York Mets")]
        assert "Multiple matches" in capsys.readouterr().out

    def test_team_list_is_cached(self) -> None:
        calls = []
        import mlb_stats.api as api_module

        original_get = api_module.requests.get

        def counting_get(url, params=None):
            calls.append(url)
            return FakeResponse(ALL_TEAMS_PAYLOAD)

        api_module.requests.get = counting_get
        try:
            api.find_team("dodgers")
            api.find_team("giants")  # different query, same underlying team list
        finally:
            api_module.requests.get = original_get
        assert len(calls) == 1


class TestGetTeamSchedule:
    @pytest.fixture(autouse=True)
    def clean_cache(self):
        api.get_team_schedule.cache_clear()
        yield
        api.get_team_schedule.cache_clear()

    def test_flattens_date_grouped_games(self, monkeypatch, team_schedule_games) -> None:
        payload = {"dates": [
            {"games": [team_schedule_games[0]]},
            {"games": [team_schedule_games[1], team_schedule_games[2]]},
        ]}
        monkeypatch.setattr(api.requests, "get", lambda url, params=None: FakeResponse(payload))
        games = api.get_team_schedule(119, 2026)
        assert len(games) == 3

    def test_no_games_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(api.requests, "get", lambda url, params=None: FakeResponse({"dates": []}))
        with pytest.raises(ValueError, match="No schedule found for team ID 119"):
            api.get_team_schedule(119, 2026)


class TestBuildTeamWinDataframe:
    def test_filters_to_completed_games_only(self, team_schedule_games) -> None:
        df = build_team_win_dataframe(team_schedule_games, team_id=119)
        assert len(df) == 6  # the 7th (future) game is excluded

    def test_win_loss_sequence(self, team_schedule_games) -> None:
        df = build_team_win_dataframe(team_schedule_games, team_id=119)
        assert list(df["win"]) == [1, 0, 1, 1, 0, 0]

    def test_opponent_is_always_the_other_team(self, team_schedule_games) -> None:
        df = build_team_win_dataframe(team_schedule_games, team_id=119)
        assert (df["opponent"] == "San Francisco Giants").all()

    def test_cumulative_win_pct(self, team_schedule_games) -> None:
        df = build_team_win_dataframe(team_schedule_games, team_id=119)
        assert list(df["cumulative"]) == pytest.approx([1.0, 0.5, 2 / 3, 0.75, 0.6, 0.5])

    def test_rolling_win_pct(self, team_schedule_games) -> None:
        df = add_rolling_stat(build_team_win_dataframe(team_schedule_games, team_id=119), "win_pct", window=5)
        assert df["rolling"].iloc[:4].isna().all()
        assert df["rolling"].iloc[4] == pytest.approx(0.6)   # games 1-5: 3 wins / 5
        assert df["rolling"].iloc[5] == pytest.approx(0.4)   # games 2-6: 2 wins / 5

    def test_game_value_is_one_or_zero(self, team_schedule_games) -> None:
        df = build_team_win_dataframe(team_schedule_games, team_id=119)
        game = compute_game_value(df, "win_pct")
        assert list(game) == [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]

    def test_no_completed_games_raises(self) -> None:
        future_only = [{
            "officialDate": "2026-04-01",
            "status": {"abstractGameState": "Preview"},
            "teams": {
                "home": {"team": {"id": 119, "name": "Los Angeles Dodgers"}},
                "away": {"team": {"id": 137, "name": "San Francisco Giants"}},
            },
        }]
        with pytest.raises(ValueError, match="No completed games found for team ID 119"):
            build_team_win_dataframe(future_only, team_id=119)
