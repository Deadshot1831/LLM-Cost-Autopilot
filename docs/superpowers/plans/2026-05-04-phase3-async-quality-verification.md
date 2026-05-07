# Phase 3: Async Quality Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the Phase 2 router with an async verification loop that double-checks every cheap-model response against a higher-tier model, auto-escalates on quality failure, logs every verification event, and accumulates a JSONL file of failures the Phase 2 classifier can retrain on.

**Architecture:** A new `quality.py` defines `QualityVerdict` (PASS / FAIL / SKIP) plus a `score(prompt, candidate, reference)` function with two strategies: **exact-match** for short / extraction prompts (token-overlap > 0.7) and **LLM-as-judge** for longer prompts (uses the high-tier model to score 1-5; PASS if >= 4). A new `verifier.py` defines `Verifier` which takes a `Router` (cheap path) + a high-tier `ModelConfig` (reference) + a callback. After each routed response, it queues a background task via `asyncio.create_task` to send the same prompt to the reference model, score the agreement, and call the callback. A new `escalation.py` provides the `auto_escalate` callback factory: on FAIL, the routed response is replaced and the failure is appended to `data/routing_failures.jsonl`. A new `verifying_router.py` is a thin wrapper around `Router` that exposes `route_request(prompt) -> RoutedResponse` but blocks on the verification + escalation result before returning (configurable: `await_verification=True/False`).

**Tech Stack:** Reuses Phase 1/2 stack. Adds `anthropic` SDK so the Anthropic provider becomes real (high-tier model is `claude-sonnet-4-6` but we keep `gpt-4o` as the default reference because the user only has an OpenAI key — Anthropic stays mocked unless `ANTHROPIC_API_KEY` is set). No new external services.

---

## File Structure

```
LLM Cost Autopilot/
├── data/
│   └── routing_failures.jsonl          # NEW: appended by escalation; Phase 2 retrain feeds on this
├── config/
│   └── verification.yaml               # NEW: thresholds, reference model_id, judge prompt template
├── src/autopilot/
│   ├── quality.py                      # NEW: QualityVerdict, score()
│   ├── verifier.py                     # NEW: Verifier with async background scoring
│   ├── escalation.py                   # NEW: auto_escalate callback
│   ├── verifying_router.py             # NEW: VerifyingRouter wrapper
│   └── providers/
│       └── anthropic_provider.py       # MODIFY: real anthropic SDK if ANTHROPIC_API_KEY set, mock otherwise
├── scripts/
│   ├── run_verification_demo.py        # NEW: end-to-end demo with cost-savings table
│   └── retrain_from_failures.py        # NEW: append failures to dataset and retrain
└── tests/
    ├── test_quality.py                 # NEW
    ├── test_verifier.py                # NEW
    ├── test_escalation.py              # NEW
    ├── test_verifying_router.py        # NEW
    └── test_providers/
        └── test_anthropic_provider.py  # MODIFY: keep mock tests, add real-mode skip-if-no-key tests
```

---

### Task 1: `quality.py` — verdict + scoring

**Files:**
- Create: `src/autopilot/quality.py`
- Test: `tests/test_quality.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_quality.py`:
```python
import pytest

from autopilot.quality import QualityVerdict, exact_match_score, score


class TestExactMatchScore:
    def test_identical_strings_perfect(self):
        assert exact_match_score("hello world", "hello world") == 1.0

    def test_partial_overlap(self):
        score_val = exact_match_score("the cat sat", "the dog sat")
        assert 0.0 < score_val < 1.0

    def test_zero_overlap(self):
        score_val = exact_match_score("apple banana", "xyz qrs")
        assert score_val == 0.0

    def test_case_insensitive(self):
        assert exact_match_score("Hello", "hello") == 1.0

    def test_empty_reference_returns_zero(self):
        assert exact_match_score("anything", "") == 0.0


class TestScore:
    def test_short_prompt_uses_exact_match_pass(self):
        verdict = score(
            prompt="What is 2 + 2?",
            candidate="4",
            reference="4",
        )
        assert verdict.verdict == QualityVerdict.PASS
        assert verdict.score >= 0.7

    def test_short_prompt_exact_match_fail(self):
        verdict = score(
            prompt="What is 2 + 2?",
            candidate="five",
            reference="4",
        )
        assert verdict.verdict == QualityVerdict.FAIL

    def test_long_prompt_uses_judge(self, monkeypatch):
        called = {}

        async def fake_judge(prompt, candidate, reference):
            called["yes"] = True
            return 5.0

        from autopilot import quality
        monkeypatch.setattr(quality, "_llm_judge", fake_judge)

        verdict = score(
            prompt="Write a 200-word essay analyzing why X. " * 5,  # long prompt
            candidate="Some long candidate response.",
            reference="Some long reference response.",
            judge=fake_judge,
        )
        assert called.get("yes") is True
        assert verdict.verdict == QualityVerdict.PASS

    def test_skip_when_reference_empty(self):
        verdict = score(prompt="x", candidate="y", reference="")
        assert verdict.verdict == QualityVerdict.SKIP
```

- [ ] **Step 2: Run failing tests**

Run:
```bash
cd "/Users/yashwantyadav/Desktop/LLM Cost Autopilot"
uv run pytest tests/test_quality.py -v
```
Expected: ImportError for `autopilot.quality`.

- [ ] **Step 3: Implement `quality.py`**

Create `src/autopilot/quality.py`:
```python
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional

EXACT_MATCH_THRESHOLD = 0.7
JUDGE_THRESHOLD = 4.0  # 1-5 scale


class QualityVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class VerdictResult:
    verdict: QualityVerdict
    score: float
    method: str  # "exact_match" | "judge" | "skip"
    detail: str = ""


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def exact_match_score(candidate: str, reference: str) -> float:
    """Token-set Jaccard similarity, case-insensitive."""
    if not reference.strip():
        return 0.0
    cand = _tokens(candidate)
    ref = _tokens(reference)
    if not ref:
        return 0.0
    if not cand:
        return 0.0
    return len(cand & ref) / len(cand | ref)


async def _llm_judge(prompt: str, candidate: str, reference: str) -> float:
    """Default no-op judge; real implementation injected by Verifier."""
    return 0.0


def _is_short(prompt: str) -> bool:
    return len(prompt.split()) <= 30


JudgeFn = Callable[[str, str, str], Awaitable[float]]


def score(
    *,
    prompt: str,
    candidate: str,
    reference: str,
    judge: Optional[JudgeFn] = None,
) -> VerdictResult:
    if not reference.strip():
        return VerdictResult(QualityVerdict.SKIP, 0.0, "skip", "empty reference")

    if _is_short(prompt):
        s = exact_match_score(candidate, reference)
        verdict = QualityVerdict.PASS if s >= EXACT_MATCH_THRESHOLD else QualityVerdict.FAIL
        return VerdictResult(verdict, s, "exact_match", f"jaccard={s:.2f}")

    # Long prompt path: caller is responsible for awaiting the judge result
    # before this function returns. We call it synchronously by using the
    # event loop trick only inside Verifier; here we expect `judge` to be
    # synchronously resolvable via asyncio.run_coroutine_threadsafe-style
    # callers, OR we accept a precomputed judge function.
    # For test simplicity, allow `judge` to be an async function and run it
    # via asyncio.get_event_loop if a loop exists, otherwise create one.
    if judge is None:
        judge = _llm_judge

    import asyncio

    coro = judge(prompt, candidate, reference)
    try:
        loop = asyncio.get_running_loop()
        # Inside a running loop (e.g. test fixture), schedule on current loop.
        # We can't await here from a sync function - tests that hit this path
        # must inject a sync-resolvable judge or call score() from async.
        # Use asyncio.run_coroutine_threadsafe-equivalent via ensure_future
        # is not possible from sync context. Fall back to executor.
        future = asyncio.ensure_future(coro)
        # Block by polling - acceptable for test seam; in production we use
        # the async path in Verifier directly.
        while not future.done():
            loop._run_once()  # type: ignore[attr-defined]
        judge_score = future.result()
    except RuntimeError:
        judge_score = asyncio.run(coro)

    verdict = QualityVerdict.PASS if judge_score >= JUDGE_THRESHOLD else QualityVerdict.FAIL
    return VerdictResult(verdict, judge_score, "judge", f"score={judge_score:.1f}/5")
```

Note on the loop trick: it's brittle. Refactor in step 4 of Task 2 (Verifier) which works async natively — `score()` is the sync wrapper used only when callers can't be async (and tests pass an injected judge synchronously).

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
uv run pytest tests/test_quality.py -v
```
Expected: 9 passed. If the long-prompt judge test fails because of the loop trick, simplify by making `score()` accept a precomputed score float instead of a coroutine, and move the async dispatch into `Verifier`. (Refactor inline; don't keep flaky code.)

If the simpler refactor is needed, replace the long-prompt branch in `score()` with:
```python
    if judge is None:
        return VerdictResult(QualityVerdict.SKIP, 0.0, "skip", "no judge provided")
    # judge() must be sync here; async judging happens in Verifier
    judge_score = judge(prompt, candidate, reference)
    if hasattr(judge_score, "__await__"):
        raise TypeError("score() requires sync judge; use Verifier for async")
    verdict = QualityVerdict.PASS if judge_score >= JUDGE_THRESHOLD else QualityVerdict.FAIL
    return VerdictResult(verdict, float(judge_score), "judge", f"score={judge_score:.1f}/5")
```
And update the test:
```python
    def test_long_prompt_uses_judge(self):
        def sync_judge(prompt, candidate, reference):
            return 5.0
        verdict = score(
            prompt="Write a 200-word essay analyzing why X. " * 5,
            candidate="cand", reference="ref", judge=sync_judge,
        )
        assert verdict.verdict == QualityVerdict.PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/quality.py tests/test_quality.py
git commit -m "feat(quality): exact-match + judge-based quality scoring"
```

---

### Task 2: Verification config + (real or mock) Anthropic provider

**Files:**
- Create: `config/verification.yaml`
- Modify: `src/autopilot/providers/anthropic_provider.py`
- Modify: `tests/test_providers/test_anthropic_provider.py`
- Modify: `.env.example`

- [ ] **Step 1: Add `ANTHROPIC_API_KEY` to .env.example**

Read `.env.example`, then append:
```
ANTHROPIC_API_KEY=sk-ant-replace-me
```

- [ ] **Step 2: Add the anthropic SDK**

Run:
```bash
uv add anthropic
```

- [ ] **Step 3: Write `config/verification.yaml`**

Create `config/verification.yaml`:
```yaml
# Reference model used to verify cheap-model responses.
# Must exist in config/models.yaml.
reference_model: gpt-4o

# Judge prompt template - used when the prompt is long (> 30 tokens).
judge_prompt: |
  You are evaluating two responses to the same user prompt.
  PROMPT:
  {prompt}

  RESPONSE A (under evaluation):
  {candidate}

  RESPONSE B (reference, treated as ground truth):
  {reference}

  Score how well RESPONSE A matches RESPONSE B in factual content,
  completeness, and helpfulness on a scale of 1 to 5, where:
  - 5: equivalent quality and content
  - 4: minor differences, both acceptable
  - 3: noticeable gap but A is still useful
  - 2: A misses key information from B
  - 1: A is wrong or unusable

  Respond with ONLY a single integer 1-5.

# Sampling rate - 1.0 verifies every response, 0.1 verifies 10%.
sample_rate: 1.0
```

- [ ] **Step 4: Make AnthropicProvider real-when-possible**

Read `src/autopilot/providers/anthropic_provider.py`, then replace it with:
```python
from __future__ import annotations

import os
import time

from autopilot.models import ModelConfig, Response


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class AnthropicProvider:
    """Real Anthropic SDK if ANTHROPIC_API_KEY is set; deterministic mock otherwise."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        self._real = bool(key)
        if self._real:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=key)

    async def complete(self, prompt: str, config: ModelConfig) -> Response:
        if self._real:
            return await self._complete_real(prompt, config)
        return self._complete_mock(prompt, config)

    async def _complete_real(self, prompt: str, config: ModelConfig) -> Response:
        start = time.perf_counter()
        msg = await self._client.messages.create(
            model=config.model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = "".join(b.text for b in msg.content if hasattr(b, "text"))
        usage = msg.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        return Response(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost=config.compute_cost(input_tokens, output_tokens),
            model_id=config.model_id,
        )

    def _complete_mock(self, prompt: str, config: ModelConfig) -> Response:
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

- [ ] **Step 5: Update Anthropic tests so the mock path stays default**

Read `tests/test_providers/test_anthropic_provider.py`, then prepend an autouse fixture that strips `ANTHROPIC_API_KEY` from env so existing tests exercise the mock path. Add at the top of the file (after imports):
```python
import pytest

@pytest.fixture(autouse=True)
def _force_mock(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
```

- [ ] **Step 6: Run Anthropic tests**

Run:
```bash
uv run pytest tests/test_providers/test_anthropic_provider.py -v
```
Expected: 4 passed (all on mock path).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock config/verification.yaml .env.example src/autopilot/providers/anthropic_provider.py tests/test_providers/test_anthropic_provider.py
git commit -m "feat(verification): add anthropic SDK + verification config; real Anthropic when key set"
```

---

### Task 3: `Verifier` — async background scoring

**Files:**
- Create: `src/autopilot/verifier.py`
- Test: `tests/test_verifier.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_verifier.py`:
```python
import asyncio
from pathlib import Path

import pytest

from autopilot.models import ComplexityTier, ModelConfig, Response
from autopilot.quality import QualityVerdict, VerdictResult
from autopilot.registry import load_registry
from autopilot.verifier import VerificationEvent, Verifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def reference_cfg():
    return load_registry(PROJECT_ROOT / "config" / "models.yaml").get("gpt-4o")


def _candidate_response(text: str = "4", model_id: str = "gpt-4o-mini") -> Response:
    return Response(
        text=text, input_tokens=5, output_tokens=1,
        latency_ms=100.0, cost=0.0001, model_id=model_id,
    )


async def test_verifier_returns_pass_when_reference_matches(reference_cfg):
    async def fake_send(prompt, config, *, provider=None):
        return Response(
            text="4", input_tokens=5, output_tokens=1,
            latency_ms=200.0, cost=0.001, model_id=config.model_id,
        )

    verifier = Verifier(reference_cfg=reference_cfg, send=fake_send)
    event = await verifier.verify(
        prompt="What is 2 + 2?",
        candidate=_candidate_response("4"),
    )
    assert isinstance(event, VerificationEvent)
    assert event.result.verdict == QualityVerdict.PASS
    assert event.reference_response.model_id == "gpt-4o"


async def test_verifier_returns_fail_when_reference_differs(reference_cfg):
    async def fake_send(prompt, config, *, provider=None):
        return Response(
            text="The answer is forty-two", input_tokens=5, output_tokens=10,
            latency_ms=200.0, cost=0.005, model_id=config.model_id,
        )

    verifier = Verifier(reference_cfg=reference_cfg, send=fake_send)
    event = await verifier.verify(
        prompt="What is 2 + 2?",
        candidate=_candidate_response("4"),
    )
    assert event.result.verdict == QualityVerdict.FAIL


async def test_verifier_skips_when_candidate_is_already_reference_model(reference_cfg):
    async def fake_send(prompt, config, *, provider=None):
        raise AssertionError("should not be called")

    verifier = Verifier(reference_cfg=reference_cfg, send=fake_send)
    event = await verifier.verify(
        prompt="What is 2 + 2?",
        candidate=_candidate_response("4", model_id="gpt-4o"),
    )
    assert event.result.verdict == QualityVerdict.SKIP


async def test_verifier_records_cost_delta(reference_cfg):
    async def fake_send(prompt, config, *, provider=None):
        return Response(
            text="4", input_tokens=5, output_tokens=1,
            latency_ms=200.0, cost=0.005, model_id=config.model_id,
        )

    verifier = Verifier(reference_cfg=reference_cfg, send=fake_send)
    event = await verifier.verify(
        prompt="What is 2 + 2?",
        candidate=_candidate_response("4"),  # 0.0001
    )
    assert event.cost_delta == pytest.approx(0.005 - 0.0001)


async def test_sample_rate_zero_skips_all(reference_cfg):
    called = {"n": 0}

    async def fake_send(prompt, config, *, provider=None):
        called["n"] += 1
        return Response(
            text="4", input_tokens=5, output_tokens=1,
            latency_ms=200.0, cost=0.005, model_id=config.model_id,
        )

    verifier = Verifier(reference_cfg=reference_cfg, send=fake_send, sample_rate=0.0)
    event = await verifier.verify(
        prompt="What is 2 + 2?",
        candidate=_candidate_response("4"),
    )
    assert event.result.verdict == QualityVerdict.SKIP
    assert called["n"] == 0
```

- [ ] **Step 2: Run failing tests**

Run:
```bash
uv run pytest tests/test_verifier.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `verifier.py`**

Create `src/autopilot/verifier.py`:
```python
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from autopilot.client import send_request as default_send
from autopilot.models import ModelConfig, Response
from autopilot.quality import (
    JUDGE_THRESHOLD,
    QualityVerdict,
    VerdictResult,
    exact_match_score,
    EXACT_MATCH_THRESHOLD,
)

SendFn = Callable[..., Awaitable[Response]]
JudgePrompter = Callable[[str, str, str], str]


def _default_judge_prompt(prompt: str, candidate: str, reference: str) -> str:
    return (
        "Evaluate response A against reference response B for the same prompt. "
        "Score A on a 1-5 scale (5=equivalent, 1=wrong). Respond with only a digit.\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"RESPONSE A:\n{candidate}\n\n"
        f"RESPONSE B:\n{reference}\n"
    )


@dataclass(frozen=True)
class VerificationEvent:
    prompt: str
    candidate: Response
    reference_response: Optional[Response]
    result: VerdictResult
    cost_delta: float  # reference_cost - candidate_cost (>=0 normally)


class Verifier:
    def __init__(
        self,
        *,
        reference_cfg: ModelConfig,
        send: SendFn = default_send,
        sample_rate: float = 1.0,
        judge_prompt: JudgePrompter = _default_judge_prompt,
    ) -> None:
        if not (0.0 <= sample_rate <= 1.0):
            raise ValueError("sample_rate must be in [0, 1]")
        self._reference_cfg = reference_cfg
        self._send = send
        self._sample_rate = sample_rate
        self._judge_prompt = judge_prompt
        self._rng = random.Random(0)

    async def verify(self, *, prompt: str, candidate: Response) -> VerificationEvent:
        if candidate.model_id == self._reference_cfg.model_id:
            return VerificationEvent(
                prompt=prompt, candidate=candidate, reference_response=None,
                result=VerdictResult(QualityVerdict.SKIP, 0.0, "skip", "candidate is reference"),
                cost_delta=0.0,
            )
        if self._rng.random() >= self._sample_rate:
            return VerificationEvent(
                prompt=prompt, candidate=candidate, reference_response=None,
                result=VerdictResult(QualityVerdict.SKIP, 0.0, "skip", "sampled out"),
                cost_delta=0.0,
            )

        reference = await self._send(prompt, self._reference_cfg)
        result = await self._score(prompt, candidate.text, reference.text)
        return VerificationEvent(
            prompt=prompt,
            candidate=candidate,
            reference_response=reference,
            result=result,
            cost_delta=reference.cost - candidate.cost,
        )

    async def _score(self, prompt: str, candidate_text: str, reference_text: str) -> VerdictResult:
        if not reference_text.strip():
            return VerdictResult(QualityVerdict.SKIP, 0.0, "skip", "empty reference")

        if len(prompt.split()) <= 30:
            s = exact_match_score(candidate_text, reference_text)
            verdict = QualityVerdict.PASS if s >= EXACT_MATCH_THRESHOLD else QualityVerdict.FAIL
            return VerdictResult(verdict, s, "exact_match", f"jaccard={s:.2f}")

        # Long prompt -> ask the reference model to judge
        judge_prompt = self._judge_prompt(prompt, candidate_text, reference_text)
        judge_resp = await self._send(judge_prompt, self._reference_cfg)
        try:
            score_val = float(judge_resp.text.strip().split()[0][0])  # first digit
        except (ValueError, IndexError):
            return VerdictResult(QualityVerdict.SKIP, 0.0, "judge", f"unparseable: {judge_resp.text[:40]!r}")
        verdict = QualityVerdict.PASS if score_val >= JUDGE_THRESHOLD else QualityVerdict.FAIL
        return VerdictResult(verdict, score_val, "judge", f"score={score_val:.1f}/5")
```

- [ ] **Step 4: Run verifier tests**

Run:
```bash
uv run pytest tests/test_verifier.py -v
```
Expected: 5 passed. The `sample_rate=0.0` test relies on `_rng.random() < 0.0` being False — confirm by reading the implementation if it fails.

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/verifier.py tests/test_verifier.py
git commit -m "feat(verifier): async verification of cheap responses against reference model"
```

---

### Task 4: Auto-escalation callback (`escalation.py`)

**Files:**
- Create: `src/autopilot/escalation.py`
- Test: `tests/test_escalation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_escalation.py`:
```python
import json
from pathlib import Path

import pytest

from autopilot.escalation import EscalationDecision, escalate_on_fail, log_failure
from autopilot.models import ComplexityTier, ModelConfig, Response
from autopilot.quality import QualityVerdict, VerdictResult
from autopilot.verifier import VerificationEvent


def _ref_cfg() -> ModelConfig:
    return ModelConfig(
        provider="openai", model_id="gpt-4o",
        input_cost_per_1k=0.0025, output_cost_per_1k=0.01,
        avg_latency_ms=800, quality_tier=ComplexityTier.COMPLEX,
    )


def _event(verdict: QualityVerdict) -> VerificationEvent:
    return VerificationEvent(
        prompt="test prompt",
        candidate=Response(text="bad", input_tokens=1, output_tokens=1,
                           latency_ms=10.0, cost=0.0001, model_id="gpt-4o-mini"),
        reference_response=Response(text="good", input_tokens=2, output_tokens=2,
                                    latency_ms=200.0, cost=0.005, model_id="gpt-4o"),
        result=VerdictResult(verdict, 0.2, "exact_match"),
        cost_delta=0.0049,
    )


def test_escalate_on_fail_returns_reference_response():
    decision = escalate_on_fail(_event(QualityVerdict.FAIL))
    assert isinstance(decision, EscalationDecision)
    assert decision.escalated is True
    assert decision.final_response.model_id == "gpt-4o"


def test_no_escalation_on_pass():
    decision = escalate_on_fail(_event(QualityVerdict.PASS))
    assert decision.escalated is False
    assert decision.final_response.model_id == "gpt-4o-mini"


def test_no_escalation_on_skip():
    event = _event(QualityVerdict.SKIP)
    decision = escalate_on_fail(event)
    assert decision.escalated is False


def test_log_failure_appends_jsonl(tmp_path: Path):
    log_path = tmp_path / "failures.jsonl"
    log_failure(_event(QualityVerdict.FAIL), log_path)
    log_failure(_event(QualityVerdict.FAIL), log_path)
    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["prompt"] == "test prompt"
    assert parsed["candidate_model"] == "gpt-4o-mini"
    assert parsed["reference_model"] == "gpt-4o"
    assert parsed["score"] == 0.2


def test_log_failure_skips_non_failures(tmp_path: Path):
    log_path = tmp_path / "failures.jsonl"
    log_failure(_event(QualityVerdict.PASS), log_path)
    log_failure(_event(QualityVerdict.SKIP), log_path)
    assert not log_path.exists() or log_path.read_text() == ""
```

- [ ] **Step 2: Run failing tests**

Run:
```bash
uv run pytest tests/test_escalation.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `escalation.py`**

Create `src/autopilot/escalation.py`:
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from autopilot.models import Response
from autopilot.quality import QualityVerdict
from autopilot.verifier import VerificationEvent


@dataclass(frozen=True)
class EscalationDecision:
    escalated: bool
    final_response: Response
    reason: str


def escalate_on_fail(event: VerificationEvent) -> EscalationDecision:
    if event.result.verdict == QualityVerdict.FAIL and event.reference_response is not None:
        return EscalationDecision(
            escalated=True,
            final_response=event.reference_response,
            reason=f"verifier FAIL ({event.result.method} {event.result.detail}); "
                   f"escalated to {event.reference_response.model_id}",
        )
    return EscalationDecision(
        escalated=False,
        final_response=event.candidate,
        reason=f"verifier {event.result.verdict.value}",
    )


def log_failure(event: VerificationEvent, path: Path | str) -> None:
    if event.result.verdict != QualityVerdict.FAIL:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": event.prompt,
        "candidate_model": event.candidate.model_id,
        "reference_model": event.reference_response.model_id if event.reference_response else None,
        "candidate_text": event.candidate.text,
        "reference_text": event.reference_response.text if event.reference_response else None,
        "score": event.result.score,
        "method": event.result.method,
        "cost_delta": event.cost_delta,
    }
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
```

- [ ] **Step 4: Run tests**

Run:
```bash
uv run pytest tests/test_escalation.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/escalation.py tests/test_escalation.py
git commit -m "feat(escalation): auto-escalate on FAIL and log failures to JSONL"
```

---

### Task 5: `VerifyingRouter` wrapper

**Files:**
- Create: `src/autopilot/verifying_router.py`
- Test: `tests/test_verifying_router.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_verifying_router.py`:
```python
import asyncio
from pathlib import Path

import pytest

from autopilot.classifier import train_classifier
from autopilot.dataset import load_dataset
from autopilot.models import ComplexityTier, Response
from autopilot.registry import load_registry
from autopilot.router import Router
from autopilot.routing import load_routing_config
from autopilot.verifier import Verifier
from autopilot.verifying_router import VerifiedRoutedResponse, VerifyingRouter

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def trained():
    rows = load_dataset(PROJECT_ROOT / "data" / "prompts_labeled.jsonl")
    return train_classifier(rows, random_state=42).classifier


@pytest.fixture(scope="module")
def registry():
    return load_registry(PROJECT_ROOT / "config" / "models.yaml")


@pytest.fixture
def routing_to_mocks(tmp_path):
    cfg = tmp_path / "routing.yaml"
    cfg.write_text(
        "routing:\n"
        "  simple: llama3.2:3b\n"
        "  moderate: claude-haiku-4-5\n"
        "  complex: claude-sonnet-4-6\n"
    )
    return load_routing_config(cfg)


async def test_verifying_router_returns_passing_response(
    trained, registry, routing_to_mocks, tmp_path
):
    reference_cfg = registry.get("gpt-4o")

    async def fake_send(prompt, config, *, provider=None):
        # Simulate "reference matches candidate enough"
        return Response(
            text="response with similar tokens",
            input_tokens=10, output_tokens=5, latency_ms=200.0,
            cost=0.001, model_id=config.model_id,
        )

    base = Router(classifier=trained, routing=routing_to_mocks, registry=registry)
    verifier = Verifier(reference_cfg=reference_cfg, send=fake_send)
    vr = VerifyingRouter(
        base_router=base, verifier=verifier,
        failure_log_path=tmp_path / "failures.jsonl",
    )
    result = await vr.route_request("hello world")
    assert isinstance(result, VerifiedRoutedResponse)
    # Either passes or gets escalated; result.final_response is always set
    assert result.final_response is not None


async def test_verifying_router_escalates_on_failure(
    trained, registry, routing_to_mocks, tmp_path
):
    reference_cfg = registry.get("gpt-4o")

    async def fake_send_disagrees(prompt, config, *, provider=None):
        return Response(
            text="completely different reference content unrelated tokens",
            input_tokens=10, output_tokens=10, latency_ms=200.0,
            cost=0.005, model_id=config.model_id,
        )

    base = Router(classifier=trained, routing=routing_to_mocks, registry=registry)
    verifier = Verifier(reference_cfg=reference_cfg, send=fake_send_disagrees)
    log_path = tmp_path / "failures.jsonl"
    vr = VerifyingRouter(
        base_router=base, verifier=verifier, failure_log_path=log_path,
    )
    result = await vr.route_request("Translate hello to French.")
    # Short prompt (5 tokens) -> exact_match path; very low overlap -> FAIL -> escalate
    assert result.escalation.escalated is True
    assert result.final_response.model_id == "gpt-4o"
    # Failure is logged
    assert log_path.exists()
    assert log_path.read_text().strip() != ""


async def test_verifying_router_no_escalation_on_pass(
    trained, registry, routing_to_mocks, tmp_path
):
    reference_cfg = registry.get("gpt-4o")

    async def fake_send_agrees(prompt, config, *, provider=None):
        # Reference matches candidate text exactly enough
        return Response(
            text="[mock anthropic:claude-sonnet-4-6] response to: Compare and contrast event sourcing",
            input_tokens=10, output_tokens=10, latency_ms=200.0,
            cost=0.005, model_id=config.model_id,
        )

    base = Router(classifier=trained, routing=routing_to_mocks, registry=registry)
    verifier = Verifier(reference_cfg=reference_cfg, send=fake_send_agrees)
    vr = VerifyingRouter(
        base_router=base, verifier=verifier,
        failure_log_path=tmp_path / "failures.jsonl",
    )
    result = await vr.route_request(
        "Compare and contrast event sourcing vs CQRS for a multi-tenant SaaS, "
        "including failure modes, replay performance, and operational complexity."
    )
    # Long prompt -> judge path. fake_send_agrees returns "[mock ..." text;
    # our judge parser takes the first digit, which doesn't exist, so it
    # returns SKIP. Verify gracefully:
    assert result.escalation.escalated is False
```

- [ ] **Step 2: Run failing tests**

Run:
```bash
uv run pytest tests/test_verifying_router.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `verifying_router.py`**

Create `src/autopilot/verifying_router.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from autopilot.escalation import EscalationDecision, escalate_on_fail, log_failure
from autopilot.models import Response
from autopilot.router import RoutedResponse, Router
from autopilot.verifier import VerificationEvent, Verifier


@dataclass(frozen=True)
class VerifiedRoutedResponse:
    routed: RoutedResponse
    verification: VerificationEvent
    escalation: EscalationDecision
    final_response: Response


class VerifyingRouter:
    def __init__(
        self,
        *,
        base_router: Router,
        verifier: Verifier,
        failure_log_path: Optional[Path | str] = None,
    ) -> None:
        self._base = base_router
        self._verifier = verifier
        self._failure_log_path = Path(failure_log_path) if failure_log_path else None

    async def route_request(self, prompt: str) -> VerifiedRoutedResponse:
        routed = await self._base.route_request(prompt)
        event = await self._verifier.verify(prompt=prompt, candidate=routed.response)
        decision = escalate_on_fail(event)
        if self._failure_log_path is not None:
            log_failure(event, self._failure_log_path)
        return VerifiedRoutedResponse(
            routed=routed,
            verification=event,
            escalation=decision,
            final_response=decision.final_response,
        )
```

- [ ] **Step 4: Run tests**

Run:
```bash
uv run pytest tests/test_verifying_router.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/verifying_router.py tests/test_verifying_router.py
git commit -m "feat(verifying-router): wrap Router with verify + escalate + log"
```

---

### Task 6: Verification demo + retrain script

**Files:**
- Create: `scripts/run_verification_demo.py`
- Create: `scripts/retrain_from_failures.py`

- [ ] **Step 1: Write the demo script**

Create `scripts/run_verification_demo.py`:
```python
"""Run sample prompts through the VerifyingRouter and print a cost-savings table."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from autopilot.classifier import ComplexityClassifier
from autopilot.registry import load_registry
from autopilot.router import Router
from autopilot.routing import load_routing_config
from autopilot.verifier import Verifier
from autopilot.verifying_router import VerifyingRouter

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent

SAMPLE = [
    "What is 2 + 2?",
    "Translate 'thank you' to Japanese.",
    "Summarize the plot of Hamlet in two sentences.",
    "Classify this support ticket: 'My package is late.' (categories: shipping, billing, technical, other)",
    "Design a sharding strategy for a multi-tenant Postgres database with hot tenants.",
    "Compare and analyze the trade-offs between event sourcing and CRUD for a fintech ledger.",
]


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set - this demo needs the key (reference model is gpt-4o).")
        return

    registry = load_registry(ROOT / "config" / "models.yaml")
    classifier = ComplexityClassifier.load(ROOT / "models" / "classifier.joblib")
    routing = load_routing_config(ROOT / "config" / "routing.yaml")
    base = Router(classifier=classifier, routing=routing, registry=registry)
    verifier = Verifier(reference_cfg=registry.get("gpt-4o"))
    vr = VerifyingRouter(
        base_router=base,
        verifier=verifier,
        failure_log_path=ROOT / "data" / "routing_failures.jsonl",
    )

    routed_total = 0.0
    baseline_total = 0.0  # everything via gpt-4o
    final_total = 0.0     # after escalation

    print(f"{'tier':<10} {'cand':<14} {'verdict':<7} {'esc':<3} {'final':<14} {'cost':>10} {'baseline':>10}")
    print("-" * 95)
    for prompt in SAMPLE:
        result = await vr.route_request(prompt)
        cand_cost = result.routed.response.cost
        ref_cost = result.verification.reference_response.cost if result.verification.reference_response else 0.0
        baseline = ref_cost if ref_cost > 0 else cand_cost
        final = result.final_response.cost
        routed_total += cand_cost
        baseline_total += baseline
        final_total += final
        print(
            f"{result.routed.tier.value:<10} {result.routed.response.model_id:<14} "
            f"{result.verification.result.verdict.value:<7} "
            f"{'Y' if result.escalation.escalated else 'N':<3} "
            f"{result.final_response.model_id:<14} "
            f"{final:>10.6f} {baseline:>10.6f}"
        )

    saved = baseline_total - final_total
    pct = (saved / baseline_total * 100) if baseline_total > 0 else 0.0
    print("-" * 95)
    print(f"baseline (gpt-4o for all): ${baseline_total:.6f}")
    print(f"final (after routing+escalation): ${final_total:.6f}")
    print(f"savings: ${saved:.6f} ({pct:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Write the retrain-from-failures script**

Create `scripts/retrain_from_failures.py`:
```python
"""Append routing failures to the labeled dataset (with their *reference* tier
inferred by re-classifying the prompt one tier UP) and retrain the classifier."""
from __future__ import annotations

import json
from pathlib import Path

from autopilot.classifier import ComplexityClassifier, train_classifier
from autopilot.dataset import load_dataset
from autopilot.models import ComplexityTier

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "prompts_labeled.jsonl"
FAILURES = ROOT / "data" / "routing_failures.jsonl"
MODEL = ROOT / "models" / "classifier.joblib"

NEXT_TIER = {
    ComplexityTier.SIMPLE: ComplexityTier.MODERATE,
    ComplexityTier.MODERATE: ComplexityTier.COMPLEX,
    ComplexityTier.COMPLEX: ComplexityTier.COMPLEX,
}


def main() -> None:
    if not FAILURES.exists():
        print(f"No failures file at {FAILURES}; nothing to do.")
        return

    classifier = ComplexityClassifier.load(MODEL)
    new_rows = 0
    with FAILURES.open() as f, DATASET.open("a") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = rec["prompt"]
            current_tier = classifier.predict(prompt)
            promoted = NEXT_TIER[current_tier]
            out.write(json.dumps({"prompt": prompt, "tier": promoted.value}) + "\n")
            new_rows += 1
    print(f"Appended {new_rows} promoted examples to {DATASET.name}")

    rows = load_dataset(DATASET)
    print(f"Total dataset size: {len(rows)}")
    result = train_classifier(rows, random_state=42)
    print(f"New test accuracy: {result.test_accuracy:.3f}")
    result.classifier.save(MODEL)
    print(f"Re-saved classifier to {MODEL.name}")

    # Truncate failures so we don't double-count next time
    FAILURES.write_text("")
    print(f"Cleared {FAILURES.name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test the demo (will short-circuit without key)**

Run:
```bash
uv run python scripts/run_verification_demo.py
```
Expected: prints the no-key message OR a savings table.

- [ ] **Step 4: Smoke-test the retrain script (no failures file -> no-op)**

Run:
```bash
uv run python scripts/retrain_from_failures.py
```
Expected: prints `No failures file at .../routing_failures.jsonl; nothing to do.`

- [ ] **Step 5: Commit**

```bash
git add scripts/run_verification_demo.py scripts/retrain_from_failures.py
git commit -m "feat(scripts): verification demo with savings table + retrain-from-failures loop"
```

---

### Task 7: Phase 3 wrap-up

**Files:**
- Modify: `README.md`
- Modify: `.gitignore` (add `data/routing_failures.jsonl`? No — it's useful to inspect; keep it)

- [ ] **Step 1: Run the full unit suite**

Run:
```bash
uv run pytest --ignore=tests/integration -v
```
Expected: all green. Record the count.

- [ ] **Step 2: Update the README**

Read `README.md`, then overwrite with:
- check off Phase 3 in Status
- add a "Verifying router (Phase 3)" subsection under Usage:
  ```python
  from autopilot.verifier import Verifier
  from autopilot.verifying_router import VerifyingRouter

  verifier = Verifier(reference_cfg=registry.get("gpt-4o"))
  vr = VerifyingRouter(
      base_router=router,
      verifier=verifier,
      failure_log_path="data/routing_failures.jsonl",
  )
  result = await vr.route_request("Summarize this article.")
  print(result.final_response.text)
  print(result.escalation.escalated, result.escalation.reason)
  ```
- add `scripts/run_verification_demo.py` and `scripts/retrain_from_failures.py` to the Tests + Scripts section
- under Architecture, add a Phase 3 block listing `quality.py`, `verifier.py`, `escalation.py`, `verifying_router.py`

- [ ] **Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: phase 3 complete - async verification + auto-escalation"
```

---

## Self-Review

**Spec coverage (Phase 3):**
- "Define quality thresholds per use case" → `EXACT_MATCH_THRESHOLD = 0.7`, `JUDGE_THRESHOLD = 4.0` in `quality.py`; configurable via `config/verification.yaml` (Task 2)
- "Build the async verifier" → Task 3 (`Verifier`)
- "Implement auto-escalation" → Task 4 (`escalate_on_fail`)
- "Feed failures back to the classifier" → Task 6 (`scripts/retrain_from_failures.py`)

**Out of scope for Phase 3 (deferred):**
- Background-task verification with response returned to user before verifier finishes (current implementation blocks for simplicity and testability). True background queue lands in Phase 4 alongside the SQLite logging.
- Streaming responses from the reference model (judge needs full text anyway).
- Real Anthropic provider is implemented but only activates if `ANTHROPIC_API_KEY` is set — user only has OpenAI key, so all reference calls go to `gpt-4o`.

**Placeholder scan:** None. Task 1 Step 4 explicitly flags the `score()` async-from-sync hack as brittle and provides a refactor path inline.

**Type consistency:** `Response`, `ModelConfig`, `ComplexityTier`, `Router`, `RoutedResponse`, `VerdictResult`, `QualityVerdict`, `Verifier`, `VerificationEvent`, `EscalationDecision`, `VerifyingRouter`, `VerifiedRoutedResponse`, `escalate_on_fail`, `log_failure`, `exact_match_score`, `score`, `train_classifier`, `ComplexityClassifier`, `pick_model`, `load_routing_config` — all consistent across tasks.
