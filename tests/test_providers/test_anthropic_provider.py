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
