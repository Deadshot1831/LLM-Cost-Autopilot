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
