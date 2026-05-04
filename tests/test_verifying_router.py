from pathlib import Path

import pytest

from autopilot.classifier import train_classifier
from autopilot.dataset import load_dataset
from autopilot.models import Response
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


async def test_verifying_router_returns_verified_response(
    trained, registry, routing_to_mocks, tmp_path
):
    reference_cfg = registry.get("gpt-4o")

    async def fake_send(prompt, config, *, provider=None):
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
    assert result.escalation.escalated is True
    assert result.final_response.model_id == "gpt-4o"
    assert log_path.exists()
    assert log_path.read_text().strip() != ""


async def test_verifying_router_no_log_when_pass(
    trained, registry, routing_to_mocks, tmp_path
):
    """When reference text matches candidate token-set, verdict is PASS,
    nothing escalates, and no failure is logged."""
    reference_cfg = registry.get("gpt-4o")

    async def fake_send_match_candidate(prompt, config, *, provider=None):
        # Ollama mock for "Translate hello to French." returns:
        # "[mock ollama:llama3.2:3b] response to: Translate hello to French."
        # Mirror that so Jaccard is 1.0.
        return Response(
            text="[mock ollama:llama3.2:3b] response to: Translate hello to French.",
            input_tokens=10, output_tokens=10, latency_ms=200.0,
            cost=0.005, model_id=config.model_id,
        )

    base = Router(classifier=trained, routing=routing_to_mocks, registry=registry)
    verifier = Verifier(reference_cfg=reference_cfg, send=fake_send_match_candidate)
    log_path = tmp_path / "failures.jsonl"
    vr = VerifyingRouter(
        base_router=base, verifier=verifier, failure_log_path=log_path,
    )
    result = await vr.route_request("Translate hello to French.")
    assert result.escalation.escalated is False
    assert not log_path.exists() or log_path.read_text() == ""


async def test_failure_log_path_optional(trained, registry, routing_to_mocks):
    reference_cfg = registry.get("gpt-4o")

    async def fake_send(prompt, config, *, provider=None):
        return Response(
            text="x", input_tokens=1, output_tokens=1, latency_ms=10.0,
            cost=0.0, model_id=config.model_id,
        )

    base = Router(classifier=trained, routing=routing_to_mocks, registry=registry)
    verifier = Verifier(reference_cfg=reference_cfg, send=fake_send)
    vr = VerifyingRouter(base_router=base, verifier=verifier, failure_log_path=None)
    result = await vr.route_request("anything")
    assert isinstance(result, VerifiedRoutedResponse)
