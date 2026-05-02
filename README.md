# LLM Cost Autopilot

Routes LLM requests to the cheapest model that can handle them at acceptable quality.

## Setup

```bash
uv sync
cp .env.example .env  # add your OPENAI_API_KEY
uv run pytest         # runs unit tests (no API calls)
uv run pytest -m integration  # runs real OpenAI smoke test
```
