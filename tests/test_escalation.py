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
    decision = escalate_on_fail(_event(QualityVerdict.SKIP))
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
