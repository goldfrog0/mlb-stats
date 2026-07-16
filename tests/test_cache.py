"""Tests for the TTL cache decorator and its application to the API
layer. Time is controlled by monkeypatching the monotonic clock the
cache module uses, so nothing here sleeps."""

from typing import Any

import pytest

import mlb_stats.api as api
import mlb_stats.cache


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(mlb_stats.cache, "monotonic", c)
    return c


class TestTtlCacheDecorator:
    def test_second_call_within_ttl_is_cached(self, clock) -> None:
        calls = []

        @mlb_stats.cache.ttl_cache(ttl_seconds=60)
        def fetch(x: int) -> int:
            calls.append(x)
            return x * 2

        assert fetch(3) == 6
        assert fetch(3) == 6
        assert calls == [3]

    def test_expires_after_ttl(self, clock) -> None:
        calls = []

        @mlb_stats.cache.ttl_cache(ttl_seconds=60)
        def fetch(x: int) -> int:
            calls.append(x)
            return x * 2

        fetch(3)
        clock.now += 61
        fetch(3)
        assert calls == [3, 3]

    def test_distinct_args_are_distinct_entries(self, clock) -> None:
        calls = []

        @mlb_stats.cache.ttl_cache(ttl_seconds=60)
        def fetch(x: int) -> int:
            calls.append(x)
            return x

        fetch(1)
        fetch(2)
        fetch(1)
        assert calls == [1, 2]

    def test_exceptions_are_not_cached(self, clock) -> None:
        calls = []

        @mlb_stats.cache.ttl_cache(ttl_seconds=60)
        def fetch(x: int) -> int:
            calls.append(x)
            if len(calls) == 1:
                raise ValueError("transient")
            return x

        with pytest.raises(ValueError):
            fetch(1)
        assert fetch(1) == 1  # retried, not replayed from cache
        assert calls == [1, 1]

    def test_maxsize_evicts_oldest(self, clock) -> None:
        calls = []

        @mlb_stats.cache.ttl_cache(ttl_seconds=60, maxsize=2)
        def fetch(x: int) -> int:
            calls.append(x)
            return x

        fetch(1)
        fetch(2)
        fetch(3)  # evicts 1
        fetch(3)  # still cached
        fetch(1)  # was evicted -> refetch
        assert calls == [1, 2, 3, 1]

    def test_cache_clear(self, clock) -> None:
        calls = []

        @mlb_stats.cache.ttl_cache(ttl_seconds=60)
        def fetch(x: int) -> int:
            calls.append(x)
            return x

        fetch(1)
        fetch.cache_clear()
        fetch(1)
        assert calls == [1, 1]


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


class TestApiCaching:
    """The api functions should hit the network once per unique lookup."""

    @pytest.fixture(autouse=True)
    def clean_caches(self):
        api.find_player.cache_clear()
        api.get_game_log.cache_clear()
        yield
        api.find_player.cache_clear()
        api.get_game_log.cache_clear()

    def test_find_player_cached(self, monkeypatch) -> None:
        http_calls = []

        def fake_get(url, params=None):
            http_calls.append(url)
            return FakeResponse({"people": [{"id": 660271, "fullName": "Shohei Ohtani"}]})

        monkeypatch.setattr(api.requests, "get", fake_get)
        assert api.find_player("Shohei Ohtani") == (660271, "Shohei Ohtani")
        assert api.find_player("Shohei Ohtani") == (660271, "Shohei Ohtani")
        assert len(http_calls) == 1

    def test_get_game_log_cached_per_key(self, monkeypatch, pitching_splits) -> None:
        http_calls = []

        def fake_get(url, params=None):
            http_calls.append((url, params["season"]))
            return FakeResponse({"stats": [{"splits": pitching_splits}]})

        monkeypatch.setattr(api.requests, "get", fake_get)
        api.get_game_log(660271, 2026, "pitching")
        api.get_game_log(660271, 2026, "pitching")  # cached
        api.get_game_log(660271, 2025, "pitching")  # different season -> fetch
        assert len(http_calls) == 2

    def test_failed_lookup_not_cached(self, monkeypatch) -> None:
        http_calls = []

        def fake_get(url, params=None):
            http_calls.append(url)
            return FakeResponse({"people": []})

        monkeypatch.setattr(api.requests, "get", fake_get)
        for _ in range(2):
            with pytest.raises(ValueError, match="No player found"):
                api.find_player("Zzznotaplayer")
        assert len(http_calls) == 2  # retried both times, never cached
