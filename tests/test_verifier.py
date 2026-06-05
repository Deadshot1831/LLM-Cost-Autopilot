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


# ---------------------------------------------------------------------
# Semantic scoring path (cosine similarity over MiniLM embeddings)
# The embedding call is monkey-patched so tests don't actually load the
# model — they verify the dispatcher routes correctly and the verdict is
# built from the right score.
# ---------------------------------------------------------------------

async def test_semantic_scoring_pass_on_paraphrase(reference_cfg, monkeypatch):
    """A short answer that's correct but worded differently from the
    reference should PASS under semantic scoring (the false-FAIL case
    that motivated this whole path)."""
    async def fake_send(prompt, config, *, provider=None):
        return Response(
            text="The capital of Japan is Tokyo. It is the country's largest city and the political and economic center.",
            input_tokens=10, output_tokens=20, latency_ms=200.0,
            cost=0.001, model_id=config.model_id,
        )

    async def fake_cosine(a, b):
        # Stand in for the real MiniLM call. 0.78 is well above the
        # SEMANTIC_THRESHOLD of 0.65.
        return 0.78

    import autopilot.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "cosine_similarity", fake_cosine)

    verifier = Verifier(
        reference_cfg=reference_cfg, send=fake_send, scoring_method="semantic",
    )
    event = await verifier.verify(
        prompt="What is the capital of Japan?",
        candidate=_candidate_response("Tokyo."),
    )
    assert event.result.verdict == QualityVerdict.PASS
    assert event.result.method == "semantic"
    assert abs(event.result.score - 0.78) < 1e-9


async def test_semantic_scoring_fails_when_unrelated(reference_cfg, monkeypatch):
    async def fake_send(prompt, config, *, provider=None):
        return Response(
            text="The capital of Japan is Tokyo.",
            input_tokens=8, output_tokens=8, latency_ms=200.0,
            cost=0.001, model_id=config.model_id,
        )

    async def fake_cosine(a, b):
        return 0.18  # well below threshold

    import autopilot.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "cosine_similarity", fake_cosine)

    verifier = Verifier(
        reference_cfg=reference_cfg, send=fake_send, scoring_method="semantic",
    )
    event = await verifier.verify(
        prompt="What is the capital of Japan?",
        candidate=_candidate_response("I have no idea what you're asking."),
    )
    assert event.result.verdict == QualityVerdict.FAIL
    assert event.result.method == "semantic"


async def test_semantic_falls_back_to_exact_match_on_import_failure(
    reference_cfg, monkeypatch,
):
    """If the embedding stack fails (e.g., torch not installed in CI), the
    verifier degrades to exact_match instead of crashing."""
    async def fake_send(prompt, config, *, provider=None):
        return Response(
            text="Tokyo",  # exact-match Jaccard with "Tokyo." -> 1.0 -> PASS
            input_tokens=2, output_tokens=1, latency_ms=200.0,
            cost=0.001, model_id=config.model_id,
        )

    async def broken_cosine(a, b):
        raise RuntimeError("simulated torch import failure")

    import autopilot.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "cosine_similarity", broken_cosine)

    verifier = Verifier(
        reference_cfg=reference_cfg, send=fake_send, scoring_method="semantic",
    )
    event = await verifier.verify(
        prompt="capital of Japan?",
        candidate=_candidate_response("Tokyo"),
    )
    # Verdict came from the fallback exact-match path
    assert event.result.method == "exact_match"
    assert "semantic unavailable" in event.result.detail
    assert event.result.verdict == QualityVerdict.PASS  # Jaccard("tokyo","tokyo")=1.0


def test_verifier_rejects_unknown_scoring_method(reference_cfg):
    with pytest.raises(ValueError):
        Verifier(reference_cfg=reference_cfg, scoring_method="bogus")  # type: ignore[arg-type]
