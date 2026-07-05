from typing import Any

import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"


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
