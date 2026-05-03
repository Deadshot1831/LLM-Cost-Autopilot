# LLM Cost Autopilot

Routes LLM requests to the cheapest model that can handle them at acceptable quality.

## Status

- [x] **Phase 1**: Unified model interface (OpenAI + mocked Anthropic/Ollama)
- [x] **Phase 2**: Complexity classifier + tier-to-model routing (92.9% test accuracy)
- [ ] Phase 3: Async quality verification
- [ ] Phase 4: Logging + dashboard
- [ ] Phase 5: FastAPI service
- [ ] Phase 6: Portfolio polish

## Setup

```bash
uv sync
cp .env.example .env                      # add OPENAI_API_KEY
uv run python scripts/train_classifier.py # produces models/classifier.joblib
```

## Usage

### Direct send_request (Phase 1)

```python
import asyncio
from autopilot.client import send_request
from autopilot.registry import load_registry

registry = load_registry("config/models.yaml")
cfg = registry.get("gpt-4o-mini")
response = asyncio.run(send_request("Hello!", cfg))
print(response.text, response.cost)
```

### Routed request (Phase 2)

```python
import asyncio
from autopilot.classifier import ComplexityClassifier
from autopilot.registry import load_registry
from autopilot.router import Router
from autopilot.routing import load_routing_config

router = Router(
    classifier=ComplexityClassifier.load("models/classifier.joblib"),
    routing=load_routing_config("config/routing.yaml"),
    registry=load_registry("config/models.yaml"),
)
result = asyncio.run(router.route_request("What is 2 + 2?"))
print(result.tier, result.response.model_id, result.response.cost)
print(result.routing_reason)
```

## Tests + Scripts

```bash
uv run pytest                              # unit tests, no API calls (59 tests)
uv run pytest -m integration               # real OpenAI smoke test (needs OPENAI_API_KEY)
uv run python scripts/run_baseline.py      # cost/latency comparison across providers
uv run python scripts/train_classifier.py  # train + persist the complexity classifier
uv run python scripts/evaluate_routing.py  # end-to-end routing demo (needs OPENAI_API_KEY)
```

## Architecture

**Phase 1 (unified interface)**
- `src/autopilot/models.py` — `ModelConfig`, `Response`, `ComplexityTier` dataclasses
- `src/autopilot/registry.py` — YAML-backed model registry (`config/models.yaml`)
- `src/autopilot/providers/` — one file per provider, all implementing the `Provider` protocol
- `src/autopilot/client.py` — `send_request(prompt, config)` dispatcher

**Phase 2 (classifier + routing)**
- `src/autopilot/features.py` — extracts numeric features from prompts (token count, instruction verbs, constraints, has-context, output format)
- `src/autopilot/dataset.py` — JSONL loader for labeled prompts
- `src/autopilot/classifier.py` — TF-IDF + LogisticRegression complexity classifier with `joblib` persistence
- `src/autopilot/routing.py` — tier-to-model map (`config/routing.yaml`)
- `src/autopilot/router.py` — `Router.route_request(prompt)` glues classifier + routing + `send_request` into a `RoutedResponse`

Anthropic and Ollama providers are deterministic mocks for Phase 1; they are
swapped for real implementations in later phases (Anthropic in Phase 3 when
the verifier needs a second opinion; Ollama whenever the local model is set up).
