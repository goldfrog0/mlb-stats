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

### Browser UI

```bash
uvicorn mlb_stats.web:app --reload
```

Then open http://127.0.0.1:8000.

## Project layout

```
mlb_stats/
├── stats.py    # Registry of supported stats (add a new stat here)
├── api.py      # MLB Stats API calls
├── plots.py    # Data shaping + matplotlib charting (used by the CLI)
├── cli.py      # CLI entry point (mlb-stats command)
├── web.py      # FastAPI backend (JSON endpoints for the browser UI)
└── static/     # Browser UI frontend (HTML/CSS/JS, Plotly.js)
```

## Development

```bash
pip install mypy
mypy mlb_stats --ignore-missing-imports
```

## Disclaimer

Not affiliated with, endorsed by, or sponsored by MLB or MLB Advanced
Media. Data is pulled from the unofficial MLB Stats API
(statsapi.mlb.com), which is undocumented and not officially supported
for third-party use.
