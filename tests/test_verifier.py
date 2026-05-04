from pathlib import Path

import pytest

from autopilot.models import ComplexityTier, ModelConfig, Response
from autopilot.quality import QualityVerdict
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
    event = await verifier.verify(prompt="What is 2 + 2?", candidate=_candidate_response("4"))
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
    event = await verifier.verify(prompt="What is 2 + 2?", candidate=_candidate_response("4"))
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
    event = await verifier.verify(prompt="What is 2 + 2?", candidate=_candidate_response("4"))
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
    event = await verifier.verify(prompt="What is 2 + 2?", candidate=_candidate_response("4"))
    assert event.result.verdict == QualityVerdict.SKIP
    assert called["n"] == 0


async def test_long_prompt_uses_judge_pass(reference_cfg):
    """Long prompts go to the judge path. Reference model returns '5'."""
    long_prompt = "Compare and analyze the trade-offs between event sourcing and CRUD " * 5
    call_count = {"n": 0}

    async def fake_send(prompt, config, *, provider=None):
        call_count["n"] += 1
        # First call: reference response. Second call: judge.
        if call_count["n"] == 1:
            return Response(text="reference content here",
                            input_tokens=10, output_tokens=10, latency_ms=200.0,
                            cost=0.005, model_id=config.model_id)
        return Response(text="5", input_tokens=20, output_tokens=1,
                        latency_ms=200.0, cost=0.001, model_id=config.model_id)

    verifier = Verifier(reference_cfg=reference_cfg, send=fake_send)
    event = await verifier.verify(
        prompt=long_prompt,
        candidate=_candidate_response("candidate answer"),
    )
    assert event.result.verdict == QualityVerdict.PASS
    assert event.result.method == "judge"


async def test_long_prompt_uses_judge_fail(reference_cfg):
    long_prompt = "Compare and analyze the trade-offs between event sourcing and CRUD " * 5
    call_count = {"n": 0}

    async def fake_send(prompt, config, *, provider=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return Response(text="reference", input_tokens=10, output_tokens=10,
                            latency_ms=200.0, cost=0.005, model_id=config.model_id)
        return Response(text="2", input_tokens=20, output_tokens=1,
                        latency_ms=200.0, cost=0.001, model_id=config.model_id)

    verifier = Verifier(reference_cfg=reference_cfg, send=fake_send)
    event = await verifier.verify(prompt=long_prompt, candidate=_candidate_response("bad answer"))
    assert event.result.verdict == QualityVerdict.FAIL
    assert event.result.method == "judge"
