"""uvicorn entry point.

Run locally:
    uv run uvicorn autopilot.api.main:app --host 0.0.0.0 --port 8000

Or via docker-compose:
    docker compose up --build
"""
from __future__ import annotations

from autopilot.api.app import create_app
from autopilot.api.state import AppState

state = AppState.from_paths()
app = create_app(state)
