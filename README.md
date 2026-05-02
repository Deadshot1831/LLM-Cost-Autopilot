# LLM Cost Autopilot

Routes LLM requests to the cheapest model that can handle them at acceptable quality.

## Status

- [x] **Phase 1**: Unified model interface (OpenAI + mocked Anthropic/Ollama)
- [ ] Phase 2: Complexity classifier
- [ ] Phase 3: Async quality verification
- [ ] Phase 4: Logging + dashboard
- [ ] Phase 5: FastAPI service
- [ ] Phase 6: Portfolio polish

## Setup

```bash
uv sync
cp .env.example .env  # add OPENAI_API_KEY
```

## Usage

```python
import asyncio
from autopilot.client import send_request
from autopilot.registry import load_registry

registry = load_registry("config/models.yaml")
cfg = registry.get("gpt-4o-mini")
response = asyncio.run(send_request("Hello!", cfg))
print(response.text, response.cost)
```

## Tests

```bash
uv run pytest                          # unit tests, no API calls (30 tests)
uv run pytest -m integration           # real OpenAI smoke test (needs OPENAI_API_KEY)
uv run python scripts/run_baseline.py  # comparison table across all providers
```

## Architecture (Phase 1)

- `src/autopilot/models.py` — `ModelConfig`, `Response`, `ComplexityTier` dataclasses
- `src/autopilot/registry.py` — YAML-backed model registry (`config/models.yaml`)
- `src/autopilot/providers/` — one file per provider, all implementing the `Provider` protocol
- `src/autopilot/client.py` — `send_request(prompt, config)` dispatcher (the only public entry point)

Anthropic and Ollama providers are deterministic mocks for Phase 1; they are
swapped for real implementations in later phases (Anthropic in Phase 3 when
the verifier needs a second opinion; Ollama whenever the local model is set up).
