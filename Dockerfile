FROM python:3.11-slim

# uv binary
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app

# Install deps first for layer caching. NOTE: we deliberately do NOT install
# the `semantic` extra here so the image stays small enough for free-tier
# hosts (Render: 512 MB RAM, Vercel: 250 MB function size). With the extra
# enabled the image would balloon ~1.5 GB just from torch + transformers.
# autopilot.verifier transparently falls back to exact_match scoring when
# sentence-transformers isn't available - the API works identically, just
# with the token-overlap verifier instead of cosine similarity.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# Add config + dataset + scripts + chat UI
COPY config ./config
COPY data/prompts_labeled.jsonl ./data/prompts_labeled.jsonl
COPY scripts ./scripts
COPY static ./static

# Train the classifier at build time so the image is self-contained
RUN uv run python scripts/train_classifier.py

# Render (and most PaaS hosts) inject a dynamic PORT env var. Default to
# 8000 for local `docker run` use. Shell-form CMD so $PORT expands at
# container start, not at image build.
ENV PORT=8000
EXPOSE 8000
CMD uv run uvicorn autopilot.api.main:app --host 0.0.0.0 --port "${PORT}"
