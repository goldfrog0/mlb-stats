from typing import Any

import requests

from mlb_stats.cache import ttl_cache

BASE_URL = "https://statsapi.mlb.com/api/v1"

# Player identities are stable, so lookups can be cached for a long time.
# Game logs gain a new entry whenever a game finishes, so they get a short
# TTL: a just-completed game shows up within this window. Failed calls are
# never cached (see ttl_cache).
PLAYER_TTL_SECONDS = 24 * 60 * 60
GAME_LOG_TTL_SECONDS = 15 * 60


@ttl_cache(PLAYER_TTL_SECONDS)
def find_player(name: str) -> tuple[int, str]:
    """Search for a player by name. Returns (player_id, full_name) or raises."""
    resp = requests.get(f"{BASE_URL}/people/search", params={"names": name})
    resp.raise_for_status()
    people = resp.json().get("people", [])

    if not people:
        raise ValueError(f"No player found for '{name}'")

    if len(people) > 1:
        print(f"Multiple matches, using: {people[0]['fullName']}")

    return people[0]["id"], people[0]["fullName"]


@ttl_cache(PLAYER_TTL_SECONDS)
def search_players(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Search for players by partial name match, for autocomplete. Unlike
    find_player, an empty result is a normal state (the user just hasn't
    typed a matching name yet) -- returns [] rather than raising.
    Filtered to active players only, since the app only has current-season
    data and a retired/minor-league player would just be a dead end."""
    resp = requests.get(f"{BASE_URL}/people/search", params={"names": query})
    resp.raise_for_status()
    people = resp.json().get("people", [])
    active = [p for p in people if p.get("active")]
    return [{"id": p["id"], "name": p["fullName"]} for p in active[:limit]]


@ttl_cache(GAME_LOG_TTL_SECONDS)
def get_game_log(player_id: int, season: int, group: str) -> list[dict[str, Any]]:
    """Fetch per-game stats for a player in a given season and stat group
    (e.g. "pitching" or "batting")."""
    params: dict[str, str | int] = {"stats": "gameLog", "group": group, "season": season}
    resp = requests.get(f"{BASE_URL}/people/{player_id}/stats", params=params)
    resp.raise_for_status()
    data = resp.json()

    stats = data.get("stats") or [{}]
    splits = stats[0].get("splits", [])
    if not splits:
        raise ValueError(f"No {group} data found for player ID {player_id} in {season}")

    return splits


@ttl_cache(PLAYER_TTL_SECONDS)
def _all_teams() -> list[dict[str, Any]]:
    resp = requests.get(f"{BASE_URL}/teams", params={"sportId": 1})
    resp.raise_for_status()
    return resp.json().get("teams", [])


def find_team(name: str) -> tuple[int, str]:
    """Search for a team by (partial, case-insensitive) name, e.g. "dodgers"
    or "los angeles". Returns (team_id, full_name) or raises. The /teams
    endpoint has no server-side search -- there are only 30 teams, so this
    fetches the full list (cached) and matches client-side against the
    full name, club name, city, and abbreviation."""
    query = name.strip().lower()
    matches = [
        t for t in _all_teams()
        if query in t["name"].lower()
        or query in t.get("teamName", "").lower()
        or query in t.get("locationName", "").lower()
        or query == t.get("abbreviation", "").lower()
    ]

    if not matches:
        raise ValueError(f"No team found for '{name}'")

    if len(matches) > 1:
        print(f"Multiple matches, using: {matches[0]['name']}")

    return matches[0]["id"], matches[0]["name"]


@ttl_cache(GAME_LOG_TTL_SECONDS)
def get_team_schedule(team_id: int, season: int) -> list[dict[str, Any]]:
    """Fetch a team's regular-season schedule/results for a season, as a
    flat list of games (flattened from the API's date-grouped shape)."""
    params: dict[str, str | int] = {"sportId": 1, "teamId": team_id, "season": season, "gameType": "R"}
    resp = requests.get(f"{BASE_URL}/schedule", params=params)
    resp.raise_for_status()
    data = resp.json()

    games = [game for date_entry in data.get("dates", []) for game in date_entry.get("games", [])]
    if not games:
        raise ValueError(f"No schedule found for team ID {team_id} in {season}")

    return games
