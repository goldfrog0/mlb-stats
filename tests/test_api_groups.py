"""Regression tests for the API's stat-group naming.

The MLB API's name for the batting group is "hitting" -- an
unrecognized group is silently ignored and the player's DEFAULT group
is returned instead. That default happens to be the hitting log for
pure batters (so group=batting appeared to work), but the PITCHING log
for a two-way player: Ohtani's batting stats were quietly computed
from his pitching appearances until get_game_log learned to
translate."""

import pytest

import mlb_stats.api as api


class TestGameLogGroupTranslation:
    @pytest.fixture(autouse=True)
    def clean_cache(self):
        api.get_game_log.cache_clear()
        yield
        api.get_game_log.cache_clear()

    def _capture_params(self, monkeypatch, splits):
        sent = {}

        def fake_get(url, params=None):
            sent.update(params)

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"stats": [{"splits": splits}]}

            return R()

        monkeypatch.setattr(api.requests, "get", fake_get)
        return sent

    def test_batting_translated_to_hitting(self, monkeypatch, batting_splits) -> None:
        sent = self._capture_params(monkeypatch, batting_splits)
        api.get_game_log(660271, 2026, "batting")
        assert sent["group"] == "hitting"

    def test_pitching_passed_through_unchanged(self, monkeypatch, pitching_splits) -> None:
        sent = self._capture_params(monkeypatch, pitching_splits)
        api.get_game_log(660271, 2026, "pitching")
        assert sent["group"] == "pitching"
