"""uvicorn entry point.

Run locally:
    uv run uvicorn autopilot.api.main:app --host 0.0.0.0 --port 8000

Or via docker-compose:
    docker compose up --build
"""
from __future__ import annotations

from dotenv import load_dotenv

# Load .env at process start so OPENAI_API_KEY / ANTHROPIC_API_KEY are
# available to the providers when send_request is first invoked. Safe to
# call even when .env is absent.
load_dotenv()

from autopilot.api.app import create_app  # noqa: E402
from autopilot.api.state import AppState  # noqa: E402

state = AppState.from_paths()
app = create_app(state)
