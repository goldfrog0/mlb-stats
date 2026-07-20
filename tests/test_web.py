"""Web backend tests via FastAPI's TestClient -- no server process, no
network: the two API-facing functions are monkeypatched to serve
fixture data, everything else (routing, validation, serialization,
static files) runs for real."""

import pytest
from fastapi.testclient import TestClient

import mlb_stats.web as web

client = TestClient(web.app)


@pytest.fixture
def fake_api(monkeypatch, pitching_splits):
    monkeypatch.setattr(web, "find_player", lambda name: (660271, f"Resolved {name}"))
    monkeypatch.setattr(web, "get_game_log", lambda pid, season, group: pitching_splits)


class TestStaticFrontend:
    def test_index_served_at_root(self) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "MLB Player" in resp.text


class TestListStats:
    def test_all_registered_stats_with_label_and_group(self) -> None:
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["era"] == {"label": "ERA", "group": "pitching"}
        assert stats["ops"] == {"label": "OPS", "group": "batting"}
        assert stats["win_pct"] == {"label": "Win%", "group": "team"}


class TestPlayerEndpoint:
    def test_response_shape(self, fake_api) -> None:
        resp = client.get("/api/player", params={"name": "Someone", "stat": "era"})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["name"] == "Resolved Someone"
        assert payload["label"] == "ERA"
        assert len(payload["data"]) == 6
        assert set(payload["data"][0]) == {"date", "opponent", "game", "cumulative", "rolling"}

    def test_nan_rolling_values_serialize_as_null(self, fake_api) -> None:
        # The first window-1 games have no rolling value; JSON must carry
        # null there, not NaN (which is invalid JSON).
        data = client.get("/api/player", params={"name": "X", "window": 5}).json()["data"]
        assert [r["rolling"] for r in data[:4]] == [None] * 4
        assert data[4]["rolling"] == pytest.approx(3.0)

    def test_season_param_is_optional(self, fake_api) -> None:
        # Omitting season falls back to the current year server-side.
        resp = client.get("/api/player", params={"name": "Someone"})
        assert resp.status_code == 200

    def test_unknown_player_is_404_with_detail(self, monkeypatch) -> None:
        def raise_not_found(name):
            raise ValueError(f"No player found for '{name}'")

        monkeypatch.setattr(web, "find_player", raise_not_found)
        resp = client.get("/api/player", params={"name": "Zzz"})
        assert resp.status_code == 404
        assert resp.json()["detail"] == "No player found for 'Zzz'"


class TestTeamStat:
    @pytest.fixture
    def fake_team_api(self, monkeypatch, team_schedule_games):
        monkeypatch.setattr(web, "find_team", lambda name: (119, f"Resolved {name}"))
        monkeypatch.setattr(web, "get_team_schedule", lambda team_id, season: team_schedule_games)

    def test_player_endpoint_routes_team_stat_through_team_lookup(self, fake_team_api) -> None:
        resp = client.get("/api/player", params={"name": "Dodgers", "stat": "win_pct"})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["name"] == "Resolved Dodgers"
        assert payload["label"] == "Win%"
        assert len(payload["data"]) == 6  # future game excluded

    def test_unknown_team_is_404(self, monkeypatch) -> None:
        def raise_not_found(name):
            raise ValueError(f"No team found for '{name}'")

        monkeypatch.setattr(web, "find_team", raise_not_found)
        resp = client.get("/api/player", params={"name": "Zzz", "stat": "win_pct"})
        assert resp.status_code == 404
        assert resp.json()["detail"] == "No team found for 'Zzz'"

    def test_compare_two_teams(self, fake_team_api) -> None:
        resp = client.get(
            "/api/compare",
            params={"player1": "Dodgers", "player2": "Giants", "stat": "win_pct"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert len(payload["player1"]["data"]) == 6
        assert len(payload["player2"]["data"]) == 6


class TestCareerWarEndpoint:
    @pytest.fixture
    def fake_war_api(self, monkeypatch, career_war_seasons):
        war_by_season_group = {
            (s["season"], group): s[key]
            for s in career_war_seasons
            for group, key in [("hitting", "batting"), ("pitching", "pitching")]
        }
        monkeypatch.setattr(web, "find_player", lambda name: (660271, f"Resolved {name}"))
        monkeypatch.setattr(web, "get_debut_year", lambda pid: 2018)
        monkeypatch.setattr(web, "get_season_war",
                            lambda pid, season, group: war_by_season_group.get((season, group)))

    def test_response_shape(self, fake_war_api) -> None:
        resp = client.get("/api/career-war", params={"name": "Someone"})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["name"] == "Resolved Someone"
        # The fixture's fully-missed 2020 season must be dropped.
        assert [s["season"] for s in payload["seasons"]] == [2018, 2019, 2021]
        assert payload["seasons"][0] == {
            "season": 2018, "batting": 2.7, "pitching": 1.1, "total": pytest.approx(3.8),
        }

    def test_unknown_player_is_404(self, monkeypatch) -> None:
        def raise_not_found(name):
            raise ValueError(f"No player found for '{name}'")

        monkeypatch.setattr(web, "find_player", raise_not_found)
        resp = client.get("/api/career-war", params={"name": "Zzz"})
        assert resp.status_code == 404

    def test_missing_name_is_a_validation_error(self) -> None:
        resp = client.get("/api/career-war")
        assert resp.status_code == 422


class TestWarApproxStat:
    @pytest.fixture
    def fake_war_api(self, monkeypatch, batting_splits):
        league = [{"atBats": 100, "hits": 25, "doubles": 5, "triples": 1, "homeRuns": 3,
                   "baseOnBalls": 10, "intentionalWalks": 1, "hitByPitch": 2, "sacFlies": 2}]
        monkeypatch.setattr(web, "find_player", lambda name: (660271, f"Resolved {name}"))
        monkeypatch.setattr(web, "get_game_log", lambda pid, season, group: batting_splits)
        monkeypatch.setattr(web, "get_league_team_stats", lambda season, group: league)
        monkeypatch.setattr(web, "get_primary_position", lambda pid: "DH")

    def test_bwar_flows_through_the_player_endpoint(self, fake_war_api) -> None:
        resp = client.get("/api/player", params={"name": "Someone", "stat": "bwar"})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["label"] == "Batting WAR (approx)"
        assert len(payload["data"]) == 6
        # Standard serialized shape, so the frontend needs no changes.
        assert set(payload["data"][0]) == {"date", "opponent", "game", "cumulative", "rolling"}

    def test_bwar_listed_in_stats(self) -> None:
        stats = client.get("/api/stats").json()
        assert stats["bwar"] == {"label": "Batting WAR (approx)", "group": "batting"}
        assert stats["pwar"] == {"label": "Pitching WAR (approx)", "group": "pitching"}


class TestPitchVelocitiesEndpoint:
    @pytest.fixture
    def fake_velo_api(self, monkeypatch, velo_game_splits, game_pitches_by_pk):
        monkeypatch.setattr(web, "find_player", lambda name: (694973, f"Resolved {name}"))
        monkeypatch.setattr(web, "get_game_log", lambda pid, season, group: velo_game_splits)
        monkeypatch.setattr(web, "get_game_pitches", lambda pk: game_pitches_by_pk[pk])

    def test_response_shape(self, fake_velo_api) -> None:
        resp = client.get("/api/pitch-velocities", params={"name": "Someone"})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["name"] == "Resolved Someone"
        # 7 fixture pitches, minus one by another pitcher, minus one untracked.
        assert len(payload["pitches"]) == 5
        assert payload["pitches"][0] == {
            "date": "2026-06-01", "opponent": "Opponent A",
            "pitch_type": "Four-Seam Fastball", "velo": 97.0,
        }

    def test_date_range_narrows_the_games(self, fake_velo_api) -> None:
        resp = client.get("/api/pitch-velocities", params={"name": "Someone", "start": "2026-06-02"})
        assert resp.status_code == 200
        pitches = resp.json()["pitches"]
        assert {p["date"] for p in pitches} == {"2026-06-06"}

    def test_empty_date_range_is_404(self, fake_velo_api) -> None:
        resp = client.get("/api/pitch-velocities", params={"name": "Someone", "start": "2027-01-01"})
        assert resp.status_code == 404
        assert "No games found" in resp.json()["detail"]

    def test_unknown_player_is_404(self, monkeypatch) -> None:
        def raise_not_found(name):
            raise ValueError(f"No player found for '{name}'")

        monkeypatch.setattr(web, "find_player", raise_not_found)
        resp = client.get("/api/pitch-velocities", params={"name": "Zzz"})
        assert resp.status_code == 404

    def test_missing_name_is_a_validation_error(self) -> None:
        resp = client.get("/api/pitch-velocities")
        assert resp.status_code == 422


class TestStandingsEndpoint:
    @pytest.fixture
    def fake_standings_api(self, monkeypatch, division_team_records):
        monkeypatch.setattr(web, "find_division", lambda name: (201, "American League East"))
        monkeypatch.setattr(web, "get_division_standings", lambda division_id, season: division_team_records)

    def test_response_shape(self, fake_standings_api) -> None:
        resp = client.get("/api/standings", params={"division": "AL East"})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["division"] == "American League East"
        assert len(payload["teams"]) == 4
        assert payload["teams"][0]["team"] == "Rays"
        assert payload["teams"][0]["pct"] == pytest.approx(0.596)

    def test_season_param_is_optional(self, fake_standings_api) -> None:
        resp = client.get("/api/standings", params={"division": "AL East"})
        assert resp.status_code == 200

    def test_unknown_division_is_404(self, monkeypatch) -> None:
        def raise_not_found(name):
            raise ValueError(f"No division found for '{name}'")

        monkeypatch.setattr(web, "find_division", raise_not_found)
        resp = client.get("/api/standings", params={"division": "Zzz"})
        assert resp.status_code == 404
        assert resp.json()["detail"] == "No division found for 'Zzz'"

    def test_missing_division_param_is_a_validation_error(self) -> None:
        resp = client.get("/api/standings")
        assert resp.status_code == 422


class TestCompareEndpoint:
    def test_both_players_resolved(self, fake_api) -> None:
        resp = client.get("/api/compare", params={"player1": "A", "player2": "B", "stat": "era"})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["player1"]["name"] == "Resolved A"
        assert payload["player2"]["name"] == "Resolved B"
        assert len(payload["player1"]["data"]) == 6

    def test_missing_player2_is_a_validation_error(self, fake_api) -> None:
        resp = client.get("/api/compare", params={"player1": "A"})
        assert resp.status_code == 422  # FastAPI validation, not a crash


class TestSearchPlayersEndpoint:
    def test_returns_matches(self, monkeypatch) -> None:
        monkeypatch.setattr(
            web, "search_players",
            lambda query: [{"id": 660271, "name": "Shohei Ohtani"}],
        )
        resp = client.get("/api/search-players", params={"q": "Sho"})
        assert resp.status_code == 200
        assert resp.json() == [{"id": 660271, "name": "Shohei Ohtani"}]

    def test_short_query_returns_empty_without_calling_search(self, monkeypatch) -> None:
        called = []
        monkeypatch.setattr(web, "search_players", lambda query: called.append(query) or [])
        resp = client.get("/api/search-players", params={"q": "S"})
        assert resp.status_code == 200
        assert resp.json() == []
        assert called == []  # never reached search_players -- rejected before the network call

    def test_missing_query_is_a_validation_error(self) -> None:
        resp = client.get("/api/search-players")
        assert resp.status_code == 422
