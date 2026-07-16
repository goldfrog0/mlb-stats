# MLB Stats

Look up an MLB player, pull their game-by-game stat log from the public
[MLB Stats API](https://statsapi.mlb.com), and plot a chosen stat over
time with a rolling average — for one player, or two players compared
side by side. Available as both a command-line tool and a browser UI.

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
just-finished game shows up within that window). Repeat plots — or
tweaks like changing the rolling window, which reuse the same game
log — skip the network entirely. Failed lookups are never cached. Each
gunicorn worker keeps its own cache, so a fresh worker starts cold.

## Project layout

```
mlb_stats/
├── stats.py    # Registry of supported stats (add a new stat here)
├── api.py      # MLB Stats API calls
├── plots.py    # Data shaping + matplotlib charting (used by the CLI)
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
| `tests/test_cache.py` | The TTL cache: expiry, eviction, errors never cached, and that the api layer really does hit the network only once per unique lookup |
| `tests/test_cli.py` | The `mlb-stats` command end to end: argument parsing, chart files actually written, `--table` output, auto-generated filenames, exit codes on errors |
| `tests/test_web.py` | The FastAPI endpoints via `TestClient` (no server needed): JSON shapes, NaN→null serialization, 404s, validation errors, the static frontend |

Useful variations:

```bash
pytest -v                  # list every test as it runs
pytest tests/test_cli.py   # one file
pytest -k rolling          # tests matching a keyword
```

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
