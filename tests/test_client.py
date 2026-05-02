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


async def test_injected_provider_overrides_default():
    from autopilot.providers.anthropic_provider import AnthropicProvider

    cfg = _cfg("anthropic", "claude-haiku-4-5")
    custom = AnthropicProvider()
    r = await send_request("hi", cfg, provider=custom)
    assert r.model_id == "claude-haiku-4-5"
