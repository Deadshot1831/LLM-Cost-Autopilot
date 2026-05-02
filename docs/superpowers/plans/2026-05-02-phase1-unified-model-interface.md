# Phase 1: Unified Model Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a provider-agnostic LLM client with a model registry, unified `send_request` interface, and standardized `Response` objects — verified against real OpenAI and mocked Anthropic/Ollama providers.

**Architecture:** A `ModelConfig` dataclass holds per-model metadata (provider, ID, costs, latency, quality tier). A `Provider` protocol is implemented by `OpenAIProvider`, `AnthropicProvider`, and `OllamaProvider` (the latter two return deterministic mock responses for now). A single `send_request(prompt, model_config)` function dispatches to the right provider, normalizes the result into a `Response` dataclass (text, input/output tokens, latency, cost, model_id), and is the only public entry point for callers.

**Tech Stack:** Python 3.11+, `uv` for env/deps, `pytest` + `pytest-asyncio` for tests, `openai` SDK, `respx`/`pytest-httpx` for HTTP mocking, `pydantic` for config validation, `pyyaml` for the registry file.

---

## File Structure

```
LLM Cost Autopilot/
├── pyproject.toml                       # uv project config, deps, pytest config
├── .python-version                      # 3.11
├── .gitignore                           # python, venv, .env, *.db
├── .env.example                         # OPENAI_API_KEY=sk-...
├── README.md                            # one-paragraph project intro
├── src/
│   └── autopilot/
│       ├── __init__.py
│       ├── models.py                    # ModelConfig, Response, ComplexityTier dataclasses
│       ├── registry.py                  # load_registry() from YAML, get_model()
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py                  # Provider protocol
│       │   ├── openai_provider.py       # real OpenAI implementation
│       │   ├── anthropic_provider.py    # mock implementation (Phase 1)
│       │   └── ollama_provider.py       # mock implementation (Phase 1)
│       └── client.py                    # send_request() dispatcher
├── config/
│   └── models.yaml                      # registry: 5 models with real pricing
└── tests/
    ├── __init__.py
    ├── conftest.py                      # shared fixtures
    ├── test_models.py                   # ModelConfig/Response dataclass tests
    ├── test_registry.py                 # YAML loading tests
    ├── test_providers/
    │   ├── __init__.py
    │   ├── test_openai_provider.py      # mocked HTTP tests
    │   ├── test_anthropic_provider.py   # mock provider tests
    │   └── test_ollama_provider.py      # mock provider tests
    ├── test_client.py                   # dispatcher tests
    └── integration/
        └── test_baseline_prompts.py     # 10-prompt smoke test (real OpenAI, opt-in)
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `README.md`
- Create: `src/autopilot/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Initialize uv project**

Run:
```bash
cd "/Users/yashwantyadav/Desktop/LLM Cost Autopilot"
uv init --package --python 3.11 --name autopilot .
```

Expected: creates `pyproject.toml`, `.python-version`, `src/autopilot/__init__.py`, and `README.md`.

- [ ] **Step 2: Add runtime and dev dependencies**

Run:
```bash
uv add openai pydantic pyyaml python-dotenv
uv add --dev pytest pytest-asyncio pytest-cov respx pytest-httpx
```

Expected: dependencies appear in `pyproject.toml`, `uv.lock` is created.

- [ ] **Step 3: Configure pytest in pyproject.toml**

Append to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "integration: real-API tests (require OPENAI_API_KEY, opt-in via -m integration)",
]
addopts = "-ra --strict-markers"
```

- [ ] **Step 4: Write `.gitignore`**

Create `.gitignore` with:
```
__pycache__/
*.py[cod]
.venv/
.env
*.db
*.sqlite
.pytest_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
.DS_Store
```

- [ ] **Step 5: Write `.env.example`**

Create `.env.example`:
```
OPENAI_API_KEY=sk-replace-me
```

- [ ] **Step 6: Write minimal README.md**

Overwrite `README.md` with:
```markdown
# LLM Cost Autopilot

Routes LLM requests to the cheapest model that can handle them at acceptable quality.

## Setup
```
uv sync
cp .env.example .env  # add your OPENAI_API_KEY
uv run pytest         # runs unit tests (no API calls)
uv run pytest -m integration  # runs real OpenAI smoke test
```
```

- [ ] **Step 7: Create empty test infrastructure**

Create `tests/__init__.py` (empty) and `tests/conftest.py`:
```python
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT
```

- [ ] **Step 8: Verify the toolchain works**

Run:
```bash
uv run pytest --collect-only
```
Expected: `collected 0 items` (no errors). If it errors, fix before continuing.

- [ ] **Step 9: Initial commit**

```bash
cd "/Users/yashwantyadav/Desktop/LLM Cost Autopilot"
git init
git add .
git commit -m "chore: scaffold uv project with pytest config"
```

---

### Task 2: Core Dataclasses (`models.py`)

**Files:**
- Create: `src/autopilot/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for `ComplexityTier`, `ModelConfig`, `Response`**

Create `tests/test_models.py`:
```python
import pytest

from autopilot.models import ComplexityTier, ModelConfig, Response


class TestComplexityTier:
    def test_three_tiers_exist(self):
        assert ComplexityTier.SIMPLE.value == "simple"
        assert ComplexityTier.MODERATE.value == "moderate"
        assert ComplexityTier.COMPLEX.value == "complex"

    def test_ordering(self):
        assert ComplexityTier.SIMPLE < ComplexityTier.MODERATE
        assert ComplexityTier.MODERATE < ComplexityTier.COMPLEX


class TestModelConfig:
    def test_construction(self):
        cfg = ModelConfig(
            provider="openai",
            model_id="gpt-4o-mini",
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006,
            avg_latency_ms=400,
            quality_tier=ComplexityTier.MODERATE,
        )
        assert cfg.provider == "openai"
        assert cfg.model_id == "gpt-4o-mini"

    def test_compute_cost(self):
        cfg = ModelConfig(
            provider="openai",
            model_id="gpt-4o-mini",
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006,
            avg_latency_ms=400,
            quality_tier=ComplexityTier.MODERATE,
        )
        # 1000 input tokens, 500 output tokens
        cost = cfg.compute_cost(input_tokens=1000, output_tokens=500)
        assert cost == pytest.approx(0.00015 + 0.0003)

    def test_negative_costs_rejected(self):
        with pytest.raises(ValueError):
            ModelConfig(
                provider="openai",
                model_id="x",
                input_cost_per_1k=-1,
                output_cost_per_1k=0.0,
                avg_latency_ms=100,
                quality_tier=ComplexityTier.SIMPLE,
            )


class TestResponse:
    def test_construction(self):
        r = Response(
            text="hello",
            input_tokens=10,
            output_tokens=5,
            latency_ms=123.4,
            cost=0.0001,
            model_id="gpt-4o-mini",
        )
        assert r.text == "hello"
        assert r.total_tokens == 15
```

- [ ] **Step 2: Run tests — expect failure**

Run:
```bash
uv run pytest tests/test_models.py -v
```
Expected: `ModuleNotFoundError: No module named 'autopilot.models'`.

- [ ] **Step 3: Implement `models.py`**

Create `src/autopilot/models.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import total_ordering


@total_ordering
class ComplexityTier(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

    def _rank(self) -> int:
        return {"simple": 0, "moderate": 1, "complex": 2}[self.value]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ComplexityTier):
            return NotImplemented
        return self._rank() < other._rank()


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_id: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    avg_latency_ms: int
    quality_tier: ComplexityTier

    def __post_init__(self) -> None:
        if self.input_cost_per_1k < 0 or self.output_cost_per_1k < 0:
            raise ValueError("Costs must be non-negative")
        if self.avg_latency_ms < 0:
            raise ValueError("Latency must be non-negative")

    def compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1000.0) * self.input_cost_per_1k
            + (output_tokens / 1000.0) * self.output_cost_per_1k
        )


@dataclass(frozen=True)
class Response:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost: float
    model_id: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
uv run pytest tests/test_models.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/models.py tests/test_models.py
git commit -m "feat(models): add ModelConfig, Response, ComplexityTier dataclasses"
```

---

### Task 3: Model Registry (YAML loader)

**Files:**
- Create: `config/models.yaml`
- Create: `src/autopilot/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the registry YAML with real pricing (May 2026)**

Create `config/models.yaml`:
```yaml
# Pricing as of 2026-05-02. Verify against provider docs before production use.
models:
  - provider: openai
    model_id: gpt-4o
    input_cost_per_1k: 0.0025
    output_cost_per_1k: 0.01
    avg_latency_ms: 800
    quality_tier: complex

  - provider: openai
    model_id: gpt-4o-mini
    input_cost_per_1k: 0.00015
    output_cost_per_1k: 0.0006
    avg_latency_ms: 400
    quality_tier: moderate

  - provider: anthropic
    model_id: claude-sonnet-4-6
    input_cost_per_1k: 0.003
    output_cost_per_1k: 0.015
    avg_latency_ms: 700
    quality_tier: complex

  - provider: anthropic
    model_id: claude-haiku-4-5
    input_cost_per_1k: 0.0008
    output_cost_per_1k: 0.004
    avg_latency_ms: 350
    quality_tier: moderate

  - provider: ollama
    model_id: llama3.2:3b
    input_cost_per_1k: 0.0
    output_cost_per_1k: 0.0
    avg_latency_ms: 1500
    quality_tier: simple
```

- [ ] **Step 2: Write failing tests for the registry**

Create `tests/test_registry.py`:
```python
from pathlib import Path

import pytest

from autopilot.models import ComplexityTier
from autopilot.registry import ModelNotFoundError, load_registry


@pytest.fixture
def registry_path(project_root: Path) -> Path:
    return project_root / "config" / "models.yaml"


class TestLoadRegistry:
    def test_loads_all_five_models(self, registry_path: Path):
        registry = load_registry(registry_path)
        assert len(registry) == 5

    def test_lookup_by_model_id(self, registry_path: Path):
        registry = load_registry(registry_path)
        cfg = registry.get("gpt-4o-mini")
        assert cfg.provider == "openai"
        assert cfg.quality_tier == ComplexityTier.MODERATE

    def test_unknown_model_raises(self, registry_path: Path):
        registry = load_registry(registry_path)
        with pytest.raises(ModelNotFoundError):
            registry.get("does-not-exist")

    def test_list_models(self, registry_path: Path):
        registry = load_registry(registry_path)
        ids = registry.list_ids()
        assert "gpt-4o" in ids
        assert "llama3.2:3b" in ids

    def test_filter_by_tier(self, registry_path: Path):
        registry = load_registry(registry_path)
        complex_models = registry.by_tier(ComplexityTier.COMPLEX)
        ids = {m.model_id for m in complex_models}
        assert ids == {"gpt-4o", "claude-sonnet-4-6"}
```

- [ ] **Step 3: Run tests — expect failure**

Run:
```bash
uv run pytest tests/test_registry.py -v
```
Expected: `ModuleNotFoundError: No module named 'autopilot.registry'`.

- [ ] **Step 4: Implement `registry.py`**

Create `src/autopilot/registry.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from autopilot.models import ComplexityTier, ModelConfig


class ModelNotFoundError(KeyError):
    pass


@dataclass(frozen=True)
class ModelRegistry:
    models: dict[str, ModelConfig] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.models)

    def get(self, model_id: str) -> ModelConfig:
        try:
            return self.models[model_id]
        except KeyError as e:
            raise ModelNotFoundError(model_id) from e

    def list_ids(self) -> list[str]:
        return list(self.models.keys())

    def by_tier(self, tier: ComplexityTier) -> list[ModelConfig]:
        return [m for m in self.models.values() if m.quality_tier == tier]


def load_registry(path: Path | str) -> ModelRegistry:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not raw or "models" not in raw:
        raise ValueError(f"Registry {path} has no 'models' key")
    models: dict[str, ModelConfig] = {}
    for entry in raw["models"]:
        cfg = ModelConfig(
            provider=entry["provider"],
            model_id=entry["model_id"],
            input_cost_per_1k=float(entry["input_cost_per_1k"]),
            output_cost_per_1k=float(entry["output_cost_per_1k"]),
            avg_latency_ms=int(entry["avg_latency_ms"]),
            quality_tier=ComplexityTier(entry["quality_tier"]),
        )
        if cfg.model_id in models:
            raise ValueError(f"Duplicate model_id: {cfg.model_id}")
        models[cfg.model_id] = cfg
    return ModelRegistry(models=models)
```

- [ ] **Step 5: Run tests — expect pass**

Run:
```bash
uv run pytest tests/test_registry.py -v
```
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add config/models.yaml src/autopilot/registry.py tests/test_registry.py
git commit -m "feat(registry): YAML-backed model registry with tier lookup"
```

---

### Task 4: Provider Protocol (`providers/base.py`)

**Files:**
- Create: `src/autopilot/providers/__init__.py`
- Create: `src/autopilot/providers/base.py`
- Create: `tests/test_providers/__init__.py`
- Test: `tests/test_providers/test_base.py`

- [ ] **Step 1: Write failing test for the Provider protocol**

Create `tests/test_providers/__init__.py` (empty).

Create `tests/test_providers/test_base.py`:
```python
from autopilot.models import ComplexityTier, ModelConfig, Response
from autopilot.providers.base import Provider


class _FakeProvider:
    name = "fake"

    async def complete(self, prompt: str, config: ModelConfig) -> Response:
        return Response(
            text="ok",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1.0,
            cost=0.0,
            model_id=config.model_id,
        )


def test_fake_satisfies_protocol():
    fake: Provider = _FakeProvider()
    assert fake.name == "fake"


async def test_fake_returns_response():
    fake = _FakeProvider()
    cfg = ModelConfig(
        provider="fake",
        model_id="x",
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        avg_latency_ms=1,
        quality_tier=ComplexityTier.SIMPLE,
    )
    r = await fake.complete("hi", cfg)
    assert r.text == "ok"
```

- [ ] **Step 2: Run test — expect failure**

Run:
```bash
uv run pytest tests/test_providers/test_base.py -v
```
Expected: import error for `autopilot.providers.base`.

- [ ] **Step 3: Implement the Provider protocol**

Create `src/autopilot/providers/__init__.py` (empty).

Create `src/autopilot/providers/base.py`:
```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

from autopilot.models import ModelConfig, Response


@runtime_checkable
class Provider(Protocol):
    name: str

    async def complete(self, prompt: str, config: ModelConfig) -> Response: ...
```

- [ ] **Step 4: Run test — expect pass**

Run:
```bash
uv run pytest tests/test_providers/test_base.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/providers/__init__.py src/autopilot/providers/base.py tests/test_providers/__init__.py tests/test_providers/test_base.py
git commit -m "feat(providers): define Provider protocol"
```

---

### Task 5: Mock Anthropic Provider

**Files:**
- Create: `src/autopilot/providers/anthropic_provider.py`
- Test: `tests/test_providers/test_anthropic_provider.py`

- [ ] **Step 1: Write failing tests for the mock Anthropic provider**

Create `tests/test_providers/test_anthropic_provider.py`:
```python
from autopilot.models import ComplexityTier, ModelConfig
from autopilot.providers.anthropic_provider import AnthropicProvider


def _cfg() -> ModelConfig:
    return ModelConfig(
        provider="anthropic",
        model_id="claude-haiku-4-5",
        input_cost_per_1k=0.0008,
        output_cost_per_1k=0.004,
        avg_latency_ms=350,
        quality_tier=ComplexityTier.MODERATE,
    )


async def test_returns_deterministic_mock_response():
    provider = AnthropicProvider()
    r = await provider.complete("Say hello", _cfg())
    assert r.text.startswith("[mock anthropic")
    assert r.model_id == "claude-haiku-4-5"


async def test_cost_is_computed_from_token_counts():
    provider = AnthropicProvider()
    r = await provider.complete("hello world", _cfg())
    expected = _cfg().compute_cost(r.input_tokens, r.output_tokens)
    assert r.cost == expected


async def test_input_tokens_track_prompt_length():
    provider = AnthropicProvider()
    short = await provider.complete("hi", _cfg())
    long = await provider.complete("hi " * 100, _cfg())
    assert long.input_tokens > short.input_tokens


async def test_provider_name():
    assert AnthropicProvider().name == "anthropic"
```

- [ ] **Step 2: Run tests — expect failure**

Run:
```bash
uv run pytest tests/test_providers/test_anthropic_provider.py -v
```
Expected: import error.

- [ ] **Step 3: Implement the mock Anthropic provider**

Create `src/autopilot/providers/anthropic_provider.py`:
```python
from __future__ import annotations

import time

from autopilot.models import ModelConfig, Response


def _approx_tokens(text: str) -> int:
    # rough heuristic: ~4 chars per token; +1 to avoid zero
    return max(1, len(text) // 4)


class AnthropicProvider:
    name = "anthropic"

    async def complete(self, prompt: str, config: ModelConfig) -> Response:
        start = time.perf_counter()
        text = f"[mock anthropic:{config.model_id}] response to: {prompt[:60]}"
        input_tokens = _approx_tokens(prompt)
        output_tokens = _approx_tokens(text)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return Response(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost=config.compute_cost(input_tokens, output_tokens),
            model_id=config.model_id,
        )
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
uv run pytest tests/test_providers/test_anthropic_provider.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/providers/anthropic_provider.py tests/test_providers/test_anthropic_provider.py
git commit -m "feat(providers): mock Anthropic provider for Phase 1"
```

---

### Task 6: Mock Ollama Provider

**Files:**
- Create: `src/autopilot/providers/ollama_provider.py`
- Test: `tests/test_providers/test_ollama_provider.py`

- [ ] **Step 1: Write failing tests for the mock Ollama provider**

Create `tests/test_providers/test_ollama_provider.py`:
```python
from autopilot.models import ComplexityTier, ModelConfig
from autopilot.providers.ollama_provider import OllamaProvider


def _cfg() -> ModelConfig:
    return ModelConfig(
        provider="ollama",
        model_id="llama3.2:3b",
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        avg_latency_ms=1500,
        quality_tier=ComplexityTier.SIMPLE,
    )


async def test_returns_zero_cost():
    provider = OllamaProvider()
    r = await provider.complete("hello", _cfg())
    assert r.cost == 0.0


async def test_response_includes_model_id():
    provider = OllamaProvider()
    r = await provider.complete("hello", _cfg())
    assert r.model_id == "llama3.2:3b"
    assert "[mock ollama" in r.text


async def test_provider_name():
    assert OllamaProvider().name == "ollama"
```

- [ ] **Step 2: Run tests — expect failure**

Run:
```bash
uv run pytest tests/test_providers/test_ollama_provider.py -v
```
Expected: import error.

- [ ] **Step 3: Implement the mock Ollama provider**

Create `src/autopilot/providers/ollama_provider.py`:
```python
from __future__ import annotations

import time

from autopilot.models import ModelConfig, Response


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class OllamaProvider:
    name = "ollama"

    async def complete(self, prompt: str, config: ModelConfig) -> Response:
        start = time.perf_counter()
        text = f"[mock ollama:{config.model_id}] response to: {prompt[:60]}"
        input_tokens = _approx_tokens(prompt)
        output_tokens = _approx_tokens(text)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return Response(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost=config.compute_cost(input_tokens, output_tokens),
            model_id=config.model_id,
        )
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
uv run pytest tests/test_providers/test_ollama_provider.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/providers/ollama_provider.py tests/test_providers/test_ollama_provider.py
git commit -m "feat(providers): mock Ollama provider for Phase 1"
```

---

### Task 7: Real OpenAI Provider (HTTP-mocked tests)

**Files:**
- Create: `src/autopilot/providers/openai_provider.py`
- Test: `tests/test_providers/test_openai_provider.py`

- [ ] **Step 1: Write failing tests using `respx` to mock the OpenAI HTTP API**

Create `tests/test_providers/test_openai_provider.py`:
```python
import httpx
import pytest
import respx

from autopilot.models import ComplexityTier, ModelConfig
from autopilot.providers.openai_provider import OpenAIProvider


def _cfg() -> ModelConfig:
    return ModelConfig(
        provider="openai",
        model_id="gpt-4o-mini",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        avg_latency_ms=400,
        quality_tier=ComplexityTier.MODERATE,
    )


@pytest.fixture
def mock_openai_response():
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1714600000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello, world!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    }


@respx.mock
async def test_completes_against_mocked_openai(mock_openai_response):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=mock_openai_response)
    )
    provider = OpenAIProvider(api_key="sk-test")
    r = await provider.complete("Say hello", _cfg())
    assert r.text == "Hello, world!"
    assert r.input_tokens == 12
    assert r.output_tokens == 5
    assert r.model_id == "gpt-4o-mini"


@respx.mock
async def test_cost_is_computed_from_usage(mock_openai_response):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=mock_openai_response)
    )
    provider = OpenAIProvider(api_key="sk-test")
    r = await provider.complete("Say hello", _cfg())
    expected = _cfg().compute_cost(12, 5)
    assert r.cost == pytest.approx(expected)


@respx.mock
async def test_latency_is_measured(mock_openai_response):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=mock_openai_response)
    )
    provider = OpenAIProvider(api_key="sk-test")
    r = await provider.complete("Say hello", _cfg())
    assert r.latency_ms >= 0


@respx.mock
async def test_api_error_raises():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "boom"}})
    )
    provider = OpenAIProvider(api_key="sk-test")
    with pytest.raises(Exception):
        await provider.complete("hi", _cfg())


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAIProvider(api_key=None)


def test_provider_name():
    assert OpenAIProvider(api_key="sk-test").name == "openai"
```

- [ ] **Step 2: Run tests — expect failure**

Run:
```bash
uv run pytest tests/test_providers/test_openai_provider.py -v
```
Expected: import error.

- [ ] **Step 3: Implement the real OpenAI provider**

Create `src/autopilot/providers/openai_provider.py`:
```python
from __future__ import annotations

import os
import time

from openai import AsyncOpenAI

from autopilot.models import ModelConfig, Response


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key."
            )
        self._client = AsyncOpenAI(api_key=key)

    async def complete(self, prompt: str, config: ModelConfig) -> Response:
        start = time.perf_counter()
        completion = await self._client.chat.completions.create(
            model=config.model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = completion.choices[0].message.content or ""
        usage = completion.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return Response(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost=config.compute_cost(input_tokens, output_tokens),
            model_id=config.model_id,
        )
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
uv run pytest tests/test_providers/test_openai_provider.py -v
```
Expected: 6 passed. If `respx` does not intercept the AsyncOpenAI HTTP traffic, switch the OpenAIProvider to use a shared `httpx.AsyncClient` directly (POST `https://api.openai.com/v1/chat/completions` with bearer token) and re-run.

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/providers/openai_provider.py tests/test_providers/test_openai_provider.py
git commit -m "feat(providers): real OpenAI provider with HTTP-mocked tests"
```

---

### Task 8: `send_request` Dispatcher (`client.py`)

**Files:**
- Create: `src/autopilot/client.py`
- Test: `tests/test_client.py`

- [ ] **Step 1: Write failing tests for the dispatcher**

Create `tests/test_client.py`:
```python
import pytest

from autopilot.client import UnsupportedProviderError, send_request
from autopilot.models import ComplexityTier, ModelConfig


def _cfg(provider: str, model_id: str = "test-model") -> ModelConfig:
    return ModelConfig(
        provider=provider,
        model_id=model_id,
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.002,
        avg_latency_ms=100,
        quality_tier=ComplexityTier.SIMPLE,
    )


async def test_dispatches_to_anthropic_provider():
    cfg = _cfg("anthropic", "claude-haiku-4-5")
    r = await send_request("hello", cfg)
    assert r.model_id == "claude-haiku-4-5"
    assert "anthropic" in r.text


async def test_dispatches_to_ollama_provider():
    cfg = _cfg("ollama", "llama3.2:3b")
    r = await send_request("hello", cfg)
    assert r.model_id == "llama3.2:3b"
    assert "ollama" in r.text


async def test_unsupported_provider_raises():
    cfg = _cfg("unknown-provider")
    with pytest.raises(UnsupportedProviderError):
        await send_request("hello", cfg)


async def test_injected_provider_overrides_default(monkeypatch):
    from autopilot.providers.anthropic_provider import AnthropicProvider

    cfg = _cfg("anthropic", "claude-haiku-4-5")
    custom = AnthropicProvider()
    r = await send_request("hi", cfg, provider=custom)
    assert r.model_id == "claude-haiku-4-5"
```

- [ ] **Step 2: Run tests — expect failure**

Run:
```bash
uv run pytest tests/test_client.py -v
```
Expected: import error.

- [ ] **Step 3: Implement the dispatcher**

Create `src/autopilot/client.py`:
```python
from __future__ import annotations

from autopilot.models import ModelConfig, Response
from autopilot.providers.anthropic_provider import AnthropicProvider
from autopilot.providers.base import Provider
from autopilot.providers.ollama_provider import OllamaProvider
from autopilot.providers.openai_provider import OpenAIProvider


class UnsupportedProviderError(ValueError):
    pass


def _default_provider(name: str) -> Provider:
    if name == "openai":
        return OpenAIProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "ollama":
        return OllamaProvider()
    raise UnsupportedProviderError(f"Unknown provider: {name}")


async def send_request(
    prompt: str,
    config: ModelConfig,
    *,
    provider: Provider | None = None,
) -> Response:
    p = provider if provider is not None else _default_provider(config.provider)
    return await p.complete(prompt, config)
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
uv run pytest tests/test_client.py -v
```
Expected: 4 passed. Note: `test_dispatches_to_anthropic_provider` and `test_dispatches_to_ollama_provider` exercise the default-dispatch path, which only constructs OpenAIProvider lazily — so missing `OPENAI_API_KEY` in CI does not break them.

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/client.py tests/test_client.py
git commit -m "feat(client): unified send_request dispatcher across providers"
```

---

### Task 9: Baseline Integration Smoke Test (real OpenAI, opt-in)

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_baseline_prompts.py`
- Create: `scripts/run_baseline.py`

- [ ] **Step 1: Write the opt-in integration test**

Create `tests/integration/__init__.py` (empty).

Create `tests/integration/test_baseline_prompts.py`:
```python
import os
from pathlib import Path

import pytest

from autopilot.client import send_request
from autopilot.registry import load_registry

pytestmark = pytest.mark.integration


BASELINE_PROMPTS = [
    "What is 2 + 2?",
    "Translate 'hello' to French.",
    "Summarize: The quick brown fox jumps over the lazy dog.",
    "Extract the email from: Contact me at jane@example.com.",
    "Classify sentiment (pos/neg): I love this product.",
    "Write a haiku about autumn.",
    "List three benefits of exercise.",
    "Convert 100 km to miles.",
    "What language is 'Bonjour le monde'?",
    "Reverse the string 'autopilot'.",
]


@pytest.fixture(scope="module")
def registry(project_root: Path):
    return load_registry(project_root / "config" / "models.yaml")


@pytest.fixture(autouse=True)
def _require_openai_key():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


async def test_gpt4o_mini_handles_all_baseline_prompts(registry):
    cfg = registry.get("gpt-4o-mini")
    for prompt in BASELINE_PROMPTS:
        r = await send_request(prompt, cfg)
        assert r.text.strip(), f"Empty response for prompt: {prompt!r}"
        assert r.input_tokens > 0
        assert r.output_tokens > 0
        assert r.cost > 0
```

- [ ] **Step 2: Run integration test — expect 1 passed (or skipped if no key)**

Run:
```bash
uv run pytest tests/integration -m integration -v
```
Expected: with `OPENAI_API_KEY` set, 1 passed. Without it, 1 skipped. If a model returns 4xx, fix the registry pricing/model_id and rerun.

- [ ] **Step 3: Write a baseline-comparison CLI script**

Create `scripts/run_baseline.py`:
```python
"""Send the 10 baseline prompts to all registered models and print a comparison table."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from autopilot.client import send_request
from autopilot.registry import load_registry

load_dotenv()

PROMPTS = [
    "What is 2 + 2?",
    "Translate 'hello' to French.",
    "Summarize: The quick brown fox jumps over the lazy dog.",
    "Extract the email from: Contact me at jane@example.com.",
    "Classify sentiment (pos/neg): I love this product.",
    "Write a haiku about autumn.",
    "List three benefits of exercise.",
    "Convert 100 km to miles.",
    "What language is 'Bonjour le monde'?",
    "Reverse the string 'autopilot'.",
]


async def _run_one(prompt: str, cfg, idx: int) -> dict:
    r = await send_request(prompt, cfg)
    return {
        "idx": idx,
        "model": cfg.model_id,
        "cost": r.cost,
        "latency_ms": round(r.latency_ms, 1),
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "preview": r.text[:60].replace("\n", " "),
    }


async def main() -> None:
    registry = load_registry(Path(__file__).resolve().parent.parent / "config" / "models.yaml")
    rows: list[dict] = []
    for cfg in registry.models.values():
        # Skip OpenAI if no key — keeps the script runnable without all providers.
        if cfg.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            print(f"[skip] {cfg.model_id} — OPENAI_API_KEY not set")
            continue
        for i, prompt in enumerate(PROMPTS, start=1):
            rows.append(await _run_one(prompt, cfg, i))

    print(f"{'idx':>3} {'model':<24} {'cost':>10} {'lat_ms':>8} {'in':>5} {'out':>5}  preview")
    print("-" * 100)
    for row in rows:
        print(
            f"{row['idx']:>3} {row['model']:<24} {row['cost']:>10.6f} "
            f"{row['latency_ms']:>8} {row['input_tokens']:>5} {row['output_tokens']:>5}  {row['preview']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run the baseline script (smoke check)**

Run:
```bash
uv run python scripts/run_baseline.py
```
Expected: a table with rows for the two mocked providers (anthropic, ollama) and — if `OPENAI_API_KEY` is set — rows for both OpenAI models. No exceptions.

- [ ] **Step 5: Run the full unit suite to confirm nothing regressed**

Run:
```bash
uv run pytest -v --ignore=tests/integration
```
Expected: all unit tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/integration scripts/run_baseline.py
git commit -m "feat(baseline): 10-prompt integration smoke test + comparison script"
```

---

### Task 10: Phase 1 Wrap-Up

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with Phase 1 status and usage**

Overwrite `README.md`:
```markdown
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
uv run pytest                      # unit tests, no API calls
uv run pytest -m integration       # real OpenAI smoke test (needs OPENAI_API_KEY)
uv run python scripts/run_baseline.py  # comparison table across all providers
```

## Architecture (Phase 1)

- `src/autopilot/models.py` — `ModelConfig`, `Response`, `ComplexityTier` dataclasses
- `src/autopilot/registry.py` — YAML-backed model registry (`config/models.yaml`)
- `src/autopilot/providers/` — one file per provider, all implementing the `Provider` protocol
- `src/autopilot/client.py` — `send_request(prompt, config)` dispatcher (the only public entry point)
```

- [ ] **Step 2: Run the full suite one final time**

Run:
```bash
uv run pytest -v --ignore=tests/integration
```
Expected: all passing.

- [ ] **Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: phase 1 complete — unified model interface"
```

---

## Self-Review

**Spec coverage (Phase 1):**
- "Create a model registry: ModelConfig dataclass with provider, model ID, costs, latency, quality tier" → Tasks 2, 3
- "Populate with real pricing for GPT-4o, GPT-4o-mini, Claude Sonnet, Claude Haiku, local Llama" → Task 3 (`config/models.yaml`)
- "Single `send_request(prompt, model_config)` function with unified interface, returns standardized Response" → Tasks 2, 8
- "Test every provider: 10 prompts, log outputs/costs/latencies" → Task 9 (integration test + baseline script)

**Out of scope for Phase 1 (deferred):**
- Real Anthropic provider (mocked here; real wiring in Phase 3 when verifier needs it)
- Real Ollama provider (mocked here; user does not have Ollama installed)
- Streaming responses (not in spec)
- Retry/backoff logic (not needed for Phase 1 baseline)

**Placeholder scan:** No "TBD" / "implement later" / "add error handling" left in the plan.

**Type consistency:** `ModelConfig`, `Response`, `ComplexityTier`, `Provider`, `ModelRegistry`, `send_request`, `_default_provider`, `compute_cost`, `complete` — checked across tasks 2–9, names/signatures match.
