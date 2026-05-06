FROM python:3.11-slim

# uv binary
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app

# Install deps first for layer caching
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# Add config + dataset + scripts
COPY config ./config
COPY data/prompts_labeled.jsonl ./data/prompts_labeled.jsonl
COPY scripts ./scripts

# Train the classifier at build time so the image is self-contained
RUN uv run python scripts/train_classifier.py

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "autopilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
