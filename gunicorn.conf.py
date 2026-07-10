"""Gunicorn configuration for production serving.

Gunicorn is a battle-tested process manager: it supervises multiple
worker processes (restarting any that die or hang) and load-balances
incoming connections across them. FastAPI is an ASGI app, so the
workers themselves are uvicorn workers -- gunicorn manages processes,
uvicorn speaks ASGI inside each one.

Run from the project root:

    gunicorn mlb_stats.web:app

(this file is picked up automatically by name). Configuration is
overridable via environment variables:

    HOST=0.0.0.0 PORT=8000 WEB_CONCURRENCY=4 gunicorn mlb_stats.web:app

For development, keep using uvicorn directly instead -- it has
auto-reload, which gunicorn setups don't do well:

    uvicorn mlb_stats.web:app --reload
"""

import multiprocessing
import os

worker_class = "uvicorn_worker.UvicornWorker"

# Common sizing rule of thumb for mostly-I/O-bound services. Override
# with WEB_CONCURRENCY when the host's CPU count doesn't reflect what
# you actually want (e.g. small containers).
workers = int(os.environ.get("WEB_CONCURRENCY", multiprocessing.cpu_count() * 2 + 1))

# Default to localhost-only, the right posture behind a reverse proxy
# (nginx/caddy). Set HOST=0.0.0.0 to expose directly on the LAN.
bind = f"{os.environ.get('HOST', '127.0.0.1')}:{os.environ.get('PORT', '8000')}"

# Log to stdout/stderr so a process supervisor (systemd, Docker) owns
# log routing rather than gunicorn writing files.
accesslog = "-"
errorlog = "-"

# Kill and replace a worker stuck longer than this (seconds). The MLB
# API normally answers in well under a second; a hung upstream call
# shouldn't pin a worker forever.
timeout = 30

# Recycle workers periodically (with jitter so they don't all restart
# at once) as cheap insurance against slow memory leaks in long-running
# processes.
max_requests = 1000
max_requests_jitter = 100
