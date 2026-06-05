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

# Warm the embeddings model if the verifier is using semantic scoring.
# First call would otherwise download the model (~80 MB) on the first
# user request and add multi-second latency. Done synchronously here so
# the server doesn't accept traffic until the model is ready.
try:
    from autopilot.verifier import Verifier as _V
    _verifier = state.verifier
    if getattr(_verifier, "_scoring_method", None) == "semantic":
        from autopilot import embeddings
        embeddings.warmup()
except Exception as _e:  # pragma: no cover
    # Don't block startup if the model can't be loaded; the verifier
    # already falls back to exact_match on import failure.
    import logging
    logging.warning("embeddings warmup skipped: %s: %s", type(_e).__name__, _e)
