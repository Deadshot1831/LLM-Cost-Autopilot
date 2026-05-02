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
