from pathlib import Path

import pytest

from autopilot.classifier import ComplexityClassifier, train_classifier
from autopilot.dataset import load_dataset
from autopilot.models import ComplexityTier
from autopilot.registry import load_registry
from autopilot.router import RoutedResponse, Router
from autopilot.routing import load_routing_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def trained() -> ComplexityClassifier:
    rows = load_dataset(PROJECT_ROOT / "data" / "prompts_labeled.jsonl")
    return train_classifier(rows, random_state=42).classifier


@pytest.fixture(scope="module")
def registry():
    return load_registry(PROJECT_ROOT / "config" / "models.yaml")


@pytest.fixture
def routing_with_mocks(tmp_path):
    """Route everything to mock providers so tests never touch OpenAI."""
    cfg = tmp_path / "routing.yaml"
    cfg.write_text(
        "routing:\n"
        "  simple: llama3.2:3b\n"
        "  moderate: claude-haiku-4-5\n"
        "  complex: claude-sonnet-4-6\n"
    )
    return load_routing_config(cfg)


async def test_router_returns_routed_response(trained, registry, routing_with_mocks):
    router = Router(classifier=trained, routing=routing_with_mocks, registry=registry)
    result = await router.route_request("What is 2 + 2?")
    assert isinstance(result, RoutedResponse)
    assert isinstance(result.tier, ComplexityTier)
    assert result.response.model_id in {"llama3.2:3b", "claude-haiku-4-5", "claude-sonnet-4-6"}


async def test_router_picks_complex_model_for_complex_prompt(
    trained, registry, routing_with_mocks
):
    router = Router(classifier=trained, routing=routing_with_mocks, registry=registry)
    result = await router.route_request(
        "Compare and contrast event sourcing vs CQRS for a multi-tenant SaaS, "
        "including failure modes, replay performance, and operational complexity."
    )
    assert result.response.model_id != "llama3.2:3b"


async def test_routing_reason_includes_confidence(
    trained, registry, routing_with_mocks
):
    router = Router(classifier=trained, routing=routing_with_mocks, registry=registry)
    result = await router.route_request("Translate 'goodbye' to Spanish.")
    assert "tier=" in result.routing_reason
    assert "confidence=" in result.routing_reason


async def test_router_dispatches_through_send_request(
    trained, registry, routing_with_mocks, monkeypatch
):
    """Confirm the router uses the unified send_request entry point."""
    seen: list[str] = []

    async def fake_send(prompt, config, *, provider=None):
        from autopilot.models import Response
        seen.append(config.model_id)
        return Response(
            text="stub", input_tokens=1, output_tokens=1,
            latency_ms=0.0, cost=0.0, model_id=config.model_id,
        )

    import autopilot.router as router_mod
    monkeypatch.setattr(router_mod, "send_request", fake_send)

    router = Router(classifier=trained, routing=routing_with_mocks, registry=registry)
    await router.route_request("hello")
    assert len(seen) == 1
