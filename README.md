# LLM Cost Autopilot

Routes LLM requests to the cheapest model that can handle them at acceptable quality.

## Status

- [x] **Phase 1**: Unified model interface (OpenAI + Anthropic + mocked Ollama)
- [x] **Phase 2**: Complexity classifier + tier-to-model routing (92.9% test accuracy)
- [x] **Phase 3**: Async quality verification + auto-escalation + retrain feedback loop
- [x] **Phase 4**: SQLite logging + Streamlit cost dashboard
- [ ] Phase 5: FastAPI service
- [ ] Phase 6: Portfolio polish

## Setup

```bash
uv sync
cp .env.example .env                       # add OPENAI_API_KEY (and ANTHROPIC_API_KEY if you have one)
uv run python scripts/train_classifier.py  # produces models/classifier.joblib
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

### Verifying router (Phase 3)

```python
import asyncio
from autopilot.verifier import Verifier
from autopilot.verifying_router import VerifyingRouter

verifier = Verifier(reference_cfg=registry.get("gpt-4o"))
vr = VerifyingRouter(
    base_router=router,
    verifier=verifier,
    failure_log_path="data/routing_failures.jsonl",
)
result = asyncio.run(vr.route_request("Summarize this article."))
print(result.final_response.text)
print(result.escalation.escalated, result.escalation.reason)
```

### Logging + dashboard (Phase 4)

```python
import asyncio
from autopilot.db import open_db
from autopilot.logging_router import LoggingRouter

conn = open_db("data/autopilot.db")
lr = LoggingRouter(verifying_router=vr, conn=conn)
asyncio.run(lr.route_request("Summarize this article."))
# then in another shell:
#   ./scripts/run_dashboard.sh
```

## Tests + Scripts

```bash
uv run pytest                                  # unit tests, no API calls (98 tests)
uv run pytest -m integration                   # real OpenAI smoke test (needs OPENAI_API_KEY)
uv run python scripts/run_baseline.py          # cost/latency comparison across providers
uv run python scripts/train_classifier.py      # train + persist the complexity classifier
uv run python scripts/evaluate_routing.py      # end-to-end routing demo (needs OPENAI_API_KEY)
uv run python scripts/run_verification_demo.py # routed + verified + savings table
uv run python scripts/retrain_from_failures.py # promote failed prompts and retrain
uv run python scripts/load_test.py -n 30       # populate the dashboard database
./scripts/run_dashboard.sh                     # launch Streamlit dashboard
```

## Architecture

**Phase 1 (unified interface)**
- `src/autopilot/models.py` — `ModelConfig`, `Response`, `ComplexityTier` dataclasses
- `src/autopilot/registry.py` — YAML-backed model registry (`config/models.yaml`)
- `src/autopilot/providers/` — one file per provider, all implementing the `Provider` protocol. OpenAI and Anthropic use real SDKs when their API keys are set; Ollama is mocked
- `src/autopilot/client.py` — `send_request(prompt, config)` dispatcher

**Phase 2 (classifier + routing)**
- `src/autopilot/features.py` — extracts numeric features from prompts (token count, instruction verbs, constraints, has-context, output format)
- `src/autopilot/dataset.py` — JSONL loader for labeled prompts
- `src/autopilot/classifier.py` — TF-IDF + LogisticRegression complexity classifier with `joblib` persistence
- `src/autopilot/routing.py` — tier-to-model map (`config/routing.yaml`)
- `src/autopilot/router.py` — `Router.route_request(prompt)` glues classifier + routing + `send_request` into a `RoutedResponse`

**Phase 3 (verification + escalation)**
- `src/autopilot/quality.py` — `QualityVerdict`, `VerdictResult`, exact-match (Jaccard) scoring
- `src/autopilot/verifier.py` — `Verifier` calls the reference model and scores agreement (exact-match for short prompts, LLM-as-judge for long prompts)
- `src/autopilot/escalation.py` — `escalate_on_fail` (swap candidate for reference) + `log_failure` (append to JSONL)
- `src/autopilot/verifying_router.py` — `VerifyingRouter` wraps a base `Router` with verify + escalate + log
- `config/verification.yaml` — reference model id, judge prompt template, sample rate
- `data/routing_failures.jsonl` — append-only log; `scripts/retrain_from_failures.py` consumes it

**Phase 4 (logging + dashboard)**
- `src/autopilot/db.py` — SQLite schema, `RequestRecord`, insert/query helpers
- `src/autopilot/logging_router.py` — `LoggingRouter` wraps `VerifyingRouter` and persists every request
- `dashboard/app.py` — Streamlit page: cost-savings headline + routing/verdict/escalation charts + recent-requests table
- `scripts/load_test.py` — populates `data/autopilot.db` with N seeded prompts so the dashboard has data
- `scripts/run_dashboard.sh` — `uv run streamlit run dashboard/app.py`
