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

# If the verifier is configured for semantic scoring AND the optional
# `sentence-transformers` extra is installed, warm the model now so the
# first user request doesn't pay the ~80 MB download + load cost.
# If the extra is NOT installed (e.g., on a memory-constrained PaaS like
# Render's free tier), the import will fail here; we log it and continue.
# The verifier itself handles the runtime fallback to exact_match scoring -
# see autopilot.verifier._score and test_semantic_falls_back_to_exact_match.
try:
    _verifier = state.verifier
    if getattr(_verifier, "_scoring_method", None) == "semantic":
        from autopilot import embeddings
        embeddings.warmup()
except Exception as _e:  # pragma: no cover
    import logging
    logging.warning(
        "embeddings warmup skipped (%s: %s) - verifier will fall back to exact_match",
        type(_e).__name__, _e,
    )
