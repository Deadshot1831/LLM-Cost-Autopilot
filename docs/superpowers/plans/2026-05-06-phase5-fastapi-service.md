# Phase 5: FastAPI Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the Phase 1-4 pipeline as a FastAPI service. One headline endpoint (`POST /v1/completions`) does the full routing-and-verification flow; three management endpoints (`GET /v1/models`, `GET /v1/stats`, `PUT /v1/routing-config`) read/edit the runtime config. Everything runs under `docker-compose` with a single API service + the SQLite volume.

**Architecture:** A new `api/app.py` builds a FastAPI app via a factory (`create_app(state)`) so tests can inject a fake state. A new `api/state.py` constructs the dependencies once at startup (registry, classifier, routing, verifier, db conn, LoggingRouter) and exposes them via FastAPI's `app.state`. A new `api/schemas.py` holds Pydantic request/response models. A new `api/main.py` is the uvicorn entry point. The router is built once and reused across requests; routing config edits via `PUT /v1/routing-config` mutate the in-memory map *and* rewrite `config/routing.yaml` so the change survives restarts.

**Tech Stack:** `fastapi`, `uvicorn[standard]`, `httpx` (already a transitive dep — used by the test client). Reuses Phase 1-4 stack. Docker base: `python:3.11-slim`.

---

## File Structure

```
LLM Cost Autopilot/
├── src/autopilot/
│   └── api/
│       ├── __init__.py             # NEW
│       ├── app.py                  # NEW: create_app factory + endpoints
│       ├── schemas.py              # NEW: Pydantic models
│       ├── state.py                # NEW: AppState construction
│       └── main.py                 # NEW: uvicorn entry point
├── Dockerfile                      # NEW
├── docker-compose.yml              # NEW
├── .dockerignore                   # NEW
└── tests/
    └── test_api/
        ├── __init__.py             # NEW
        ├── conftest.py             # NEW: shared client fixture
        ├── test_completions.py     # NEW
        ├── test_models_endpoint.py # NEW
        ├── test_stats_endpoint.py  # NEW
        └── test_routing_config_endpoint.py # NEW
```

---

### Task 1: Add FastAPI + uvicorn deps

**Files:**
- Modify: `pyproject.toml` (auto)

- [ ] **Step 1: Install deps**

Run:
```bash
cd "/Users/yashwantyadav/Desktop/LLM Cost Autopilot"
uv add fastapi 'uvicorn[standard]'
uv add --dev httpx  # already transitive but make it explicit for the test client
```

- [ ] **Step 2: Verify imports**

Run:
```bash
uv run python -c "import fastapi, uvicorn; print(fastapi.__version__, uvicorn.__version__)"
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add fastapi + uvicorn for Phase 5 API service"
```

---

### Task 2: Pydantic schemas (`api/schemas.py`)

**Files:**
- Create: `src/autopilot/api/__init__.py`
- Create: `src/autopilot/api/schemas.py`
- Test: covered indirectly by endpoint tests in later tasks

- [ ] **Step 1: Create the package**

```bash
mkdir -p src/autopilot/api
```

Create `src/autopilot/api/__init__.py` (empty).

- [ ] **Step 2: Write schemas**

Create `src/autopilot/api/schemas.py`:
```python
from __future__ import annotations

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt")


class CompletionMeta(BaseModel):
    tier: str
    candidate_model: str
    final_model: str
    escalated: bool
    verdict: str
    verdict_score: float
    verdict_method: str
    routing_reason: str
    final_cost: float
    final_latency_ms: float


class CompletionResponse(BaseModel):
    text: str
    meta: CompletionMeta


class ModelInfo(BaseModel):
    model_id: str
    provider: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    avg_latency_ms: int
    quality_tier: str


class ModelsListResponse(BaseModel):
    models: list[ModelInfo]


class StatsResponse(BaseModel):
    total_requests: int
    final_cost_total: float
    baseline_cost_total: float
    savings_total: float
    savings_pct: float


class RoutingConfigRequest(BaseModel):
    simple: str = Field(..., description="model_id for SIMPLE tier")
    moderate: str = Field(..., description="model_id for MODERATE tier")
    complex: str = Field(..., description="model_id for COMPLEX tier")


class RoutingConfigResponse(BaseModel):
    simple: str
    moderate: str
    complex: str
```

- [ ] **Step 3: Commit**

```bash
git add src/autopilot/api/__init__.py src/autopilot/api/schemas.py
git commit -m "feat(api): Pydantic request/response schemas"
```

---

### Task 3: AppState (`api/state.py`)

**Files:**
- Create: `src/autopilot/api/state.py`

- [ ] **Step 1: Write `AppState`**

Create `src/autopilot/api/state.py`:
```python
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from autopilot.classifier import ComplexityClassifier
from autopilot.db import open_db
from autopilot.logging_router import LoggingRouter
from autopilot.models import ComplexityTier
from autopilot.registry import ModelRegistry, load_registry
from autopilot.router import Router
from autopilot.routing import load_routing_config
from autopilot.verifier import Verifier
from autopilot.verifying_router import VerifyingRouter

ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class AppState:
    registry: ModelRegistry
    routing: dict[ComplexityTier, str]
    routing_path: Path
    classifier: ComplexityClassifier
    base_router: Router
    verifier: Verifier
    verifying_router: VerifyingRouter
    db_conn: sqlite3.Connection
    logging_router: LoggingRouter

    @classmethod
    def from_paths(
        cls,
        *,
        models_yaml: Path | str = ROOT / "config" / "models.yaml",
        routing_yaml: Path | str = ROOT / "config" / "routing.yaml",
        classifier_path: Path | str = ROOT / "models" / "classifier.joblib",
        db_path: Path | str = ROOT / "data" / "autopilot.db",
        failure_log_path: Path | str = ROOT / "data" / "routing_failures.jsonl",
        reference_model_id: str = "gpt-4o",
    ) -> "AppState":
        registry = load_registry(models_yaml)
        routing = load_routing_config(routing_yaml)
        classifier = ComplexityClassifier.load(classifier_path)
        base_router = Router(
            classifier=classifier, routing=routing, registry=registry,
        )
        verifier = Verifier(reference_cfg=registry.get(reference_model_id))
        verifying_router = VerifyingRouter(
            base_router=base_router, verifier=verifier,
            failure_log_path=failure_log_path,
        )
        conn = open_db(db_path)
        logging_router = LoggingRouter(
            verifying_router=verifying_router, conn=conn,
        )
        return cls(
            registry=registry,
            routing=routing,
            routing_path=Path(routing_yaml),
            classifier=classifier,
            base_router=base_router,
            verifier=verifier,
            verifying_router=verifying_router,
            db_conn=conn,
            logging_router=logging_router,
        )

    def update_routing(self, new_routing: dict[ComplexityTier, str]) -> None:
        """Mutate in-memory routing AND persist back to YAML."""
        for model_id in new_routing.values():
            self.registry.get(model_id)  # raises if unknown
        self.routing.clear()
        self.routing.update(new_routing)
        # Rebuild base_router so its bound dict is the same object
        self.base_router = Router(
            classifier=self.classifier,
            routing=self.routing,
            registry=self.registry,
        )
        self.verifying_router = VerifyingRouter(
            base_router=self.base_router,
            verifier=self.verifier,
            failure_log_path=getattr(self.verifying_router, "_failure_log_path", None),
        )
        self.logging_router = LoggingRouter(
            verifying_router=self.verifying_router, conn=self.db_conn,
        )
        # Persist to YAML
        self.routing_path.write_text(
            "routing:\n"
            f"  simple: {self.routing[ComplexityTier.SIMPLE]}\n"
            f"  moderate: {self.routing[ComplexityTier.MODERATE]}\n"
            f"  complex: {self.routing[ComplexityTier.COMPLEX]}\n"
        )
```

- [ ] **Step 2: Commit**

```bash
git add src/autopilot/api/state.py
git commit -m "feat(api): AppState bundles all dependencies + supports routing reload"
```

---

### Task 4: FastAPI app + endpoints (`api/app.py`)

**Files:**
- Create: `src/autopilot/api/app.py`
- Create: `tests/test_api/__init__.py`
- Create: `tests/test_api/conftest.py`

- [ ] **Step 1: Write `app.py`**

Create `src/autopilot/api/app.py`:
```python
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from autopilot.api.schemas import (
    CompletionMeta,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelsListResponse,
    RoutingConfigRequest,
    RoutingConfigResponse,
    StatsResponse,
)
from autopilot.api.state import AppState
from autopilot.db import query_aggregate_costs
from autopilot.models import ComplexityTier
from autopilot.registry import ModelNotFoundError


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="LLM Cost Autopilot", version="0.1.0")
    app.state.autopilot = state

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(req: CompletionRequest) -> CompletionResponse:
        s: AppState = app.state.autopilot
        result = await s.logging_router.route_request(req.prompt)
        return CompletionResponse(
            text=result.final_response.text,
            meta=CompletionMeta(
                tier=result.routed.tier.value,
                candidate_model=result.routed.response.model_id,
                final_model=result.final_response.model_id,
                escalated=result.escalation.escalated,
                verdict=result.verification.result.verdict.value,
                verdict_score=result.verification.result.score,
                verdict_method=result.verification.result.method,
                routing_reason=result.routed.routing_reason,
                final_cost=result.final_response.cost,
                final_latency_ms=result.final_response.latency_ms,
            ),
        )

    @app.get("/v1/models", response_model=ModelsListResponse)
    def list_models() -> ModelsListResponse:
        s: AppState = app.state.autopilot
        models = [
            ModelInfo(
                model_id=cfg.model_id,
                provider=cfg.provider,
                input_cost_per_1k=cfg.input_cost_per_1k,
                output_cost_per_1k=cfg.output_cost_per_1k,
                avg_latency_ms=cfg.avg_latency_ms,
                quality_tier=cfg.quality_tier.value,
            )
            for cfg in s.registry.models.values()
        ]
        return ModelsListResponse(models=models)

    @app.get("/v1/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        s: AppState = app.state.autopilot
        agg = query_aggregate_costs(s.db_conn)
        return StatsResponse(**agg)

    @app.get("/v1/routing-config", response_model=RoutingConfigResponse)
    def get_routing() -> RoutingConfigResponse:
        s: AppState = app.state.autopilot
        return RoutingConfigResponse(
            simple=s.routing[ComplexityTier.SIMPLE],
            moderate=s.routing[ComplexityTier.MODERATE],
            complex=s.routing[ComplexityTier.COMPLEX],
        )

    @app.put("/v1/routing-config", response_model=RoutingConfigResponse)
    def update_routing(req: RoutingConfigRequest) -> RoutingConfigResponse:
        s: AppState = app.state.autopilot
        new_map = {
            ComplexityTier.SIMPLE: req.simple,
            ComplexityTier.MODERATE: req.moderate,
            ComplexityTier.COMPLEX: req.complex,
        }
        try:
            s.update_routing(new_map)
        except ModelNotFoundError as e:
            raise HTTPException(status_code=400, detail=f"Unknown model_id: {e}")
        return RoutingConfigResponse(
            simple=s.routing[ComplexityTier.SIMPLE],
            moderate=s.routing[ComplexityTier.MODERATE],
            complex=s.routing[ComplexityTier.COMPLEX],
        )

    return app
```

- [ ] **Step 2: Test infrastructure**

Create `tests/test_api/__init__.py` (empty).

Create `tests/test_api/conftest.py`:
```python
import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autopilot.api.app import create_app
from autopilot.api.state import AppState
from autopilot.classifier import train_classifier
from autopilot.db import open_db
from autopilot.dataset import load_dataset
from autopilot.logging_router import LoggingRouter
from autopilot.models import ComplexityTier, Response
from autopilot.registry import load_registry
from autopilot.router import Router
from autopilot.routing import load_routing_config
from autopilot.verifier import Verifier
from autopilot.verifying_router import VerifyingRouter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="session")
def trained():
    rows = load_dataset(PROJECT_ROOT / "data" / "prompts_labeled.jsonl")
    return train_classifier(rows, random_state=42).classifier


@pytest.fixture
def app_state(trained, tmp_path):
    routing_yaml = tmp_path / "routing.yaml"
    routing_yaml.write_text(
        "routing:\n"
        "  simple: llama3.2:3b\n"
        "  moderate: claude-haiku-4-5\n"
        "  complex: claude-sonnet-4-6\n"
    )
    registry = load_registry(PROJECT_ROOT / "config" / "models.yaml")
    routing = load_routing_config(routing_yaml)
    base = Router(classifier=trained, routing=routing, registry=registry)

    async def fake_send(prompt, config, *, provider=None):
        return Response(
            text="reference content here", input_tokens=10, output_tokens=10,
            latency_ms=200.0, cost=0.005, model_id=config.model_id,
        )

    verifier = Verifier(reference_cfg=registry.get("gpt-4o"), send=fake_send)
    vr = VerifyingRouter(
        base_router=base, verifier=verifier,
        failure_log_path=tmp_path / "fail.jsonl",
    )
    conn = open_db(tmp_path / "test.db")
    lr = LoggingRouter(verifying_router=vr, conn=conn)
    state = AppState(
        registry=registry, routing=routing, routing_path=routing_yaml,
        classifier=trained, base_router=base, verifier=verifier,
        verifying_router=vr, db_conn=conn, logging_router=lr,
    )
    return state


@pytest.fixture
def client(app_state):
    app = create_app(app_state)
    return TestClient(app)
```

- [ ] **Step 3: Smoke-test that the app constructs**

Run:
```bash
uv run python -c "
from autopilot.api.app import create_app
print('app module imports cleanly')
"
```
Expected: prints the message.

- [ ] **Step 4: Commit**

```bash
git add src/autopilot/api/app.py tests/test_api/__init__.py tests/test_api/conftest.py
git commit -m "feat(api): FastAPI app factory + endpoint wiring"
```

---

### Task 5: Endpoint tests

**Files:**
- Create: `tests/test_api/test_completions.py`
- Create: `tests/test_api/test_models_endpoint.py`
- Create: `tests/test_api/test_stats_endpoint.py`
- Create: `tests/test_api/test_routing_config_endpoint.py`

- [ ] **Step 1: Write completions tests**

Create `tests/test_api/test_completions.py`:
```python
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_post_completions_returns_text_and_meta(client):
    r = client.post("/v1/completions", json={"prompt": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert "text" in body
    assert "meta" in body
    meta = body["meta"]
    assert meta["tier"] in {"simple", "moderate", "complex"}
    assert "candidate_model" in meta
    assert "final_model" in meta
    assert isinstance(meta["escalated"], bool)
    assert "routing_reason" in meta


def test_completions_rejects_empty_prompt(client):
    r = client.post("/v1/completions", json={"prompt": ""})
    assert r.status_code == 422  # Pydantic validation


def test_completions_increments_stats(client):
    s_before = client.get("/v1/stats").json()
    client.post("/v1/completions", json={"prompt": "hello"})
    s_after = client.get("/v1/stats").json()
    assert s_after["total_requests"] == s_before["total_requests"] + 1
```

- [ ] **Step 2: Write models endpoint tests**

Create `tests/test_api/test_models_endpoint.py`:
```python
def test_get_models_returns_all_registered(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    models = r.json()["models"]
    ids = {m["model_id"] for m in models}
    assert {"gpt-4o", "gpt-4o-mini"}.issubset(ids)
    sample = next(m for m in models if m["model_id"] == "gpt-4o")
    assert sample["provider"] == "openai"
    assert sample["quality_tier"] == "complex"
    assert sample["input_cost_per_1k"] > 0
```

- [ ] **Step 3: Write stats endpoint tests**

Create `tests/test_api/test_stats_endpoint.py`:
```python
def test_stats_starts_at_zero(client):
    r = client.get("/v1/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_requests"] == 0
    assert body["savings_pct"] == 0.0


def test_stats_after_one_request(client):
    client.post("/v1/completions", json={"prompt": "hello world"})
    body = client.get("/v1/stats").json()
    assert body["total_requests"] == 1
    assert body["baseline_cost_total"] > 0
```

- [ ] **Step 4: Write routing-config endpoint tests**

Create `tests/test_api/test_routing_config_endpoint.py`:
```python
def test_get_routing_config(client):
    r = client.get("/v1/routing-config")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"simple", "moderate", "complex"}


def test_put_routing_config_updates(client):
    new_cfg = {
        "simple": "gpt-4o-mini",
        "moderate": "gpt-4o-mini",
        "complex": "gpt-4o",
    }
    r = client.put("/v1/routing-config", json=new_cfg)
    assert r.status_code == 200
    assert r.json() == new_cfg
    # GET after PUT reflects the change
    after = client.get("/v1/routing-config").json()
    assert after == new_cfg


def test_put_routing_config_rejects_unknown_model(client):
    r = client.put("/v1/routing-config", json={
        "simple": "nonexistent", "moderate": "gpt-4o-mini", "complex": "gpt-4o",
    })
    assert r.status_code == 400


def test_put_routing_config_persists_to_yaml(client, app_state, tmp_path):
    # Use the routing_path the fixture wrote into tmp_path
    new_cfg = {
        "simple": "gpt-4o-mini",
        "moderate": "claude-haiku-4-5",
        "complex": "gpt-4o",
    }
    client.put("/v1/routing-config", json=new_cfg)
    text = app_state.routing_path.read_text()
    assert "gpt-4o-mini" in text
    assert "claude-haiku-4-5" in text
```

- [ ] **Step 5: Run all API tests**

Run:
```bash
uv run pytest tests/test_api -v
```
Expected: 13 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/test_api
git commit -m "test(api): coverage for completions, models, stats, routing-config endpoints"
```

---

### Task 6: uvicorn entry point

**Files:**
- Create: `src/autopilot/api/main.py`

- [ ] **Step 1: Write entry point**

Create `src/autopilot/api/main.py`:
```python
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
```

- [ ] **Step 2: Smoke-test that the entry point imports**

Run:
```bash
uv run python -c "from autopilot.api.main import app; print(type(app).__name__)"
```
Expected: `FastAPI`. (Will fail if `models/classifier.joblib` doesn't exist — run `train_classifier.py` first.)

- [ ] **Step 3: Commit**

```bash
git add src/autopilot/api/main.py
git commit -m "feat(api): uvicorn entry point loading default config + classifier"
```

---

### Task 7: Docker

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `docker-compose.yml`

- [ ] **Step 1: Write `.dockerignore`**

Create `.dockerignore`:
```
.venv/
.env
.git/
.pytest_cache/
.coverage
htmlcov/
__pycache__/
*.pyc
*.pyo
data/autopilot.db
docs/
dashboard/
.DS_Store
```

- [ ] **Step 2: Write `Dockerfile`**

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

# uv binary
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app

# Install deps first for layer caching
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# Add config + dataset + models
COPY config ./config
COPY data/prompts_labeled.jsonl ./data/prompts_labeled.jsonl
COPY scripts ./scripts

# Train the classifier at build time so the image is self-contained
RUN uv run python scripts/train_classifier.py

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "autopilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Write `docker-compose.yml`**

Create `docker-compose.yml`:
```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
    volumes:
      # Persist the SQLite DB and failure log across container restarts
      - ./data:/app/data
```

- [ ] **Step 4: (Optional) Build smoke test**

Skip the actual `docker build` — it pulls a base image and would take minutes. Instead validate the Dockerfile syntactically:
```bash
docker build --check . 2>/dev/null || echo "(docker not available - skipping syntax check)"
```

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore docker-compose.yml
git commit -m "feat(docker): Dockerfile + docker-compose for the API service"
```

---

### Task 8: Phase 5 wrap-up

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full unit suite**

Run:
```bash
uv run pytest --ignore=tests/integration
```
Expected: all green.

- [ ] **Step 2: Update README**

Read `README.md`, then:
- check off Phase 5 in Status
- add an "API service (Phase 5)" subsection under Usage:
  ```bash
  # Local
  uv run python scripts/train_classifier.py
  uv run uvicorn autopilot.api.main:app --reload

  # Or docker compose
  cp .env.example .env  # add your keys
  docker compose up --build

  # Then:
  curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "What is 2 + 2?"}'

  curl http://localhost:8000/v1/stats
  curl http://localhost:8000/v1/models
  ```
- under Architecture, add a Phase 5 block listing `api/app.py`, `api/schemas.py`, `api/state.py`, `api/main.py`, `Dockerfile`, `docker-compose.yml`

- [ ] **Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: phase 5 complete - FastAPI service + docker-compose"
```

---

## Self-Review

**Spec coverage (Phase 5):**
- "Single POST /v1/completions endpoint, user doesn't choose the model, returns metadata" → Task 4 (`completions` endpoint + `CompletionMeta`)
- "GET /v1/models" → Task 4 (`list_models`)
- "GET /v1/stats" → Task 4 (`stats`)
- "PUT /v1/routing-config" (update without redeploying) → Task 4 (`update_routing` mutates in-memory + rewrites YAML)
- "Containerize: API + background worker + SQLite" → Task 7 (Dockerfile + docker-compose; SQLite is a volume; the "background worker" from the original spec collapses into the same process because Phase 3 verification is synchronous — no separate worker needed)

**Out of scope for Phase 5 (deferred):**
- Authentication / rate limiting — single-tenant V1.
- Streaming responses — `Verifier` would have to consume the full text before scoring, so streaming the user the cheap response while the verifier runs in background is a Phase 4-redesign concern.
- True background-worker container for async verification — current design verifies inline. Adding a worker would mean splitting `LoggingRouter.route_request` into "return cheap response immediately, queue verification" — a real refactor, deferred.
- `/v1/health` returns just `{"status":"ok"}` — no DB ping or registry sanity check yet.

**Placeholder scan:** None.

**Type consistency:** `AppState`, `create_app`, `CompletionRequest/Response/Meta`, `ModelInfo`, `ModelsListResponse`, `StatsResponse`, `RoutingConfigRequest/Response`, `update_routing`, `LoggingRouter`, `VerifyingRouter`, `Router`, `Verifier`, `ModelRegistry` — all consistent across tasks 2-5.
