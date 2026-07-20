# MLB Stats

[![CI](https://github.com/goldfrog0/mlb-stats/actions/workflows/ci.yml/badge.svg)](https://github.com/goldfrog0/mlb-stats/actions/workflows/ci.yml)

Look up an MLB player or team, pull their game-by-game stat log or
schedule from the public [MLB Stats API](https://statsapi.mlb.com),
and plot a chosen stat over time with a rolling average — for one
player/team, or two compared side by side. Available as both a
command-line tool and a browser UI.

## Setup

Requires Python 3.

```bash
git clone https://github.com/goldfrog0/mlb-stats.git
cd mlb-stats

python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the `mlb-stats` CLI command in editable
mode, so any changes you make to the files in `mlb_stats/` take effect
immediately without reinstalling.

## Usage

### CLI

```bash
mlb-stats "Shohei Ohtani"
mlb-stats "Shohei Ohtani" "Paul Skenes" --stat era --layout stacked --diff
mlb-stats "Los Angeles Dodgers" --stat win_pct
mlb-stats --standings "AL East" --table
mlb-stats "Paul Skenes" --velo --start-date 2026-06-01 --end-date 2026-06-30
mlb-stats "Shohei Ohtani" --stat bwar   # approximate per-game WAR
```

See [HOW_TO_USE.txt](HOW_TO_USE.txt) for the full list of stats,
options, and examples.

### Browser UI (development)

```bash
uvicorn mlb_stats.web:app --reload
```

Then open http://127.0.0.1:8000. `--reload` picks up source changes
automatically, which is what you want while developing.

### Browser UI (production)

For real serving, run under gunicorn, which supervises a pool of
uvicorn workers (restarting any that die or hang) and load-balances
across them:

```bash
gunicorn mlb_stats.web:app
```

Settings live in [gunicorn.conf.py](gunicorn.conf.py) (picked up
automatically) and can be overridden with environment variables:

```bash
HOST=0.0.0.0 PORT=8080 WEB_CONCURRENCY=4 gunicorn mlb_stats.web:app
```

By default it binds to `127.0.0.1:8000` — the right posture behind a
reverse proxy like nginx or caddy, which is how you'd want TLS and
compression handled in a real deployment. Set `HOST=0.0.0.0` to expose
it directly on your network instead. Worker count defaults to
`2 × CPU cores + 1`; override with `WEB_CONCURRENCY`. Note that
gunicorn is Unix-only (Linux/macOS).

### Caching

MLB API responses are cached in-process (`mlb_stats/cache.py`): player
lookups for 24 hours (IDs are stable), game logs for 15 minutes (so a
just-finished game shows up within that window), and per-game pitch
data for 24 hours (a completed game's pitches never change). Repeat plots — or
tweaks like changing the rolling window, which reuse the same game
log — skip the network entirely. Failed lookups are never cached. Each
gunicorn worker keeps its own cache, so a fresh worker starts cold.

## Project layout

```
mlb_stats/
├── stats.py    # Registry of supported stats (add a new stat here)
├── api.py      # MLB Stats API calls (players and teams)
├── plots.py    # Data shaping + matplotlib charting (used by the CLI)
├── war.py      # Approximate per-game WAR (bwar/pwar stats)
├── cli.py      # CLI entry point (mlb-stats command)
├── web.py      # FastAPI backend (JSON endpoints for the browser UI)
└── static/     # Browser UI frontend (HTML/CSS/JS, Plotly.js)
tests/            # pytest suite (see Development below)
gunicorn.conf.py  # Production server config (workers, bind, logging)
```

## Development

Dev tooling is separate from the runtime dependencies:

```bash
pip install -r requirements-dev.txt
```

### Running the tests

```bash
pytest
```

The suite runs entirely offline and takes a couple of seconds — no MLB
API calls are made. Tests use synthetic game logs shaped exactly like
real API responses (built in `tests/conftest.py`, with small
hand-checkable numbers), and the two network-facing functions
(`find_player`, `get_game_log`) are monkeypatched wherever a test
exercises code that would otherwise hit the network.

What lives where:

| File | Covers |
| --- | --- |
| `tests/test_plots.py` | The data pipeline: innings-pitched box-score parsing ("6.2" = 6⅔), rolling windows summing counts (not averaging rates), per-game values, FIP's weights/constant, OPS as a composite |
| `tests/test_stats.py` | Stat-registry consistency, so a malformed new entry fails a test instead of crashing at runtime |
| `tests/test_teams.py` | Team lookup (partial/city/abbreviation matching), schedule fetching, and flattening a schedule into win/loss + cumulative win% -- including the doubleheader (duplicate-date) regression |
| `tests/test_standings.py` | Division lookup (AL/NL alias expansion, ambiguous matches), fetching a division's standings, and shaping them into a display-ready DataFrame |
| `tests/test_velo.py` | Pitch velocities: flattening a game's play-by-play feed into pitches, date-range filtering, and the per-pitch DataFrame (dropping other pitchers' and untracked pitches) |
| `tests/test_war_approx.py` | Approximate WAR: league wOBA/FIP baselines aggregated from team totals, hand-computed per-game batting/pitching WAR, positional adjustments, rolling-sum semantics |
| `tests/test_api_groups.py` | The group=batting→hitting API translation regression (a two-way player's batting stats silently coming from their pitching log) |
| `tests/test_cache.py` | The TTL cache: expiry, eviction, errors never cached, and that the api layer really does hit the network only once per unique lookup |
| `tests/test_cli.py` | The `mlb-stats` command end to end: argument parsing, chart files actually written, `--table` output, auto-generated filenames, exit codes on errors |
| `tests/test_web.py` | The FastAPI endpoints via `TestClient` (no server needed): JSON shapes, NaN→null serialization, 404s, validation errors, player-search autocomplete, the static frontend |

Useful variations:

```bash
pytest -v                  # list every test as it runs
pytest tests/test_cli.py   # one file
pytest -k rolling          # tests matching a keyword
```

### CI

[GitHub Actions](.github/workflows/ci.yml) runs ruff, mypy, and the
full test suite on every push and pull request against `master`, on
Python 3.11 (the floor — `NotRequired` in `stats.py` needs it) and
3.13.

When adding a stat or feature, the usual pattern is: extend the fixture
data in `tests/conftest.py` if new fields are needed, then assert
against values you computed by hand — not values re-derived with the
same code under test.

### Type checking and linting

```bash
mypy mlb_stats --ignore-missing-imports
ruff check mlb_stats tests
```

## Disclaimer

Not affiliated with, endorsed by, or sponsored by MLB or MLB Advanced
Media. Data is pulled from the unofficial MLB Stats API
(statsapi.mlb.com), which is undocumented and not officially supported
for third-party use.
