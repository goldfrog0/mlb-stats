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
    (e.g. "pitching" or "batting").

    The API's name for the batting group is "hitting" -- an unrecognized
    group is silently ignored and the player's default group is returned
    instead, which happens to be the hitting log for pure batters (so
    "batting" appeared to work) but the PITCHING log for a two-way
    player. Translate rather than renaming the app-wide group, which is
    user-facing in labels and docs."""
    api_group = {"batting": "hitting"}.get(group, group)
    params: dict[str, str | int] = {"stats": "gameLog", "group": api_group, "season": season}
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


def _normalize_division_query(query: str) -> str:
    """"AL East" -> "american league east" so it substring-matches the
    API's full division names ("American League East"). Only the first
    word gets expanded, so "national league" typed out already works
    unchanged."""
    words = query.strip().lower().split()
    aliases = {"al": "american league", "nl": "national league"}
    if words and words[0] in aliases:
        words[0] = aliases[words[0]]
    return " ".join(words)


def find_division(name: str) -> tuple[int, str]:
    """Search for a division by (partial, case-insensitive) name, e.g.
    "AL East", "National League Central", or "west". Returns
    (division_id, full_name) or raises. Derived from the cached team
    list (each team carries its division), rather than a separate
    endpoint call."""
    query = _normalize_division_query(name)
    matched_by_id: dict[int, str] = {}
    for t in _all_teams():
        division = t["division"]
        if query in division["name"].lower():
            matched_by_id.setdefault(division["id"], division["name"])

    if not matched_by_id:
        raise ValueError(f"No division found for '{name}'")

    if len(matched_by_id) > 1:
        first_name = next(iter(matched_by_id.values()))
        print(f"Multiple matches, using: {first_name}")

    division_id, division_name = next(iter(matched_by_id.items()))
    return division_id, division_name


@ttl_cache(GAME_LOG_TTL_SECONDS)
def get_division_standings(division_id: int, season: int) -> list[dict[str, Any]]:
    """Fetch a division's current standings for a season: one record per
    team (record, rank, games back, streak), in the order the API
    returns them (already ranked best-to-worst)."""
    params: dict[str, str | int] = {"leagueId": "103,104", "season": season}
    resp = requests.get(f"{BASE_URL}/standings", params=params)
    resp.raise_for_status()
    data = resp.json()

    for division_record in data.get("records", []):
        if division_record.get("division", {}).get("id") == division_id:
            team_records = division_record.get("teamRecords", [])
            if not team_records:
                raise ValueError(f"No standings found for division ID {division_id} in {season}")
            return team_records

    raise ValueError(f"No standings found for division ID {division_id} in {season}")
