"""Small in-process TTL cache decorator.

Used to avoid re-fetching from the MLB Stats API on repeat requests --
which matters for the web app, where a long-running worker serves many
requests (the CLI process only lives for one command). Each process
keeps its own cache, so under gunicorn every worker caches
independently; that's fine for this app's traffic and avoids any shared
state.

Thread-safe: the web endpoints are sync functions, which FastAPI runs
in a threadpool, so multiple threads can hit the cache concurrently.
"""

import threading
from collections.abc import Callable
from functools import wraps
from time import monotonic
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def ttl_cache(ttl_seconds: float, maxsize: int = 256) -> Callable[[F], F]:
    """Cache a function's return values for ttl_seconds.

    Exceptions are never cached -- a failed call is retried on the next
    request. Entries beyond maxsize are evicted oldest-first. The
    wrapped function gains a cache_clear() method (like lru_cache).
    """

    def decorator(func: F) -> F:
        cache: dict[tuple[Any, ...], tuple[float, Any]] = {}
        lock = threading.Lock()

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = args + tuple(sorted(kwargs.items()))
            now = monotonic()

            with lock:
                hit = cache.get(key)
                if hit is not None and hit[0] > now:
                    return hit[1]

            # Deliberately outside the lock: holding it during a network
            # call would serialize every request through the cache. The
            # tradeoff is that two concurrent identical requests may both
            # fetch -- harmless, the second result just wins.
            value = func(*args, **kwargs)

            with lock:
                cache[key] = (now + ttl_seconds, value)
                for stale in [k for k, (expires, _) in cache.items() if expires <= now]:
                    del cache[stale]
                while len(cache) > maxsize:
                    del cache[next(iter(cache))]

            return value

        def cache_clear() -> None:
            with lock:
                cache.clear()

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
