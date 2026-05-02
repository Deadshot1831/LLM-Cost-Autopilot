from __future__ import annotations

from autopilot.models import ModelConfig, Response
from autopilot.providers.anthropic_provider import AnthropicProvider
from autopilot.providers.base import Provider
from autopilot.providers.ollama_provider import OllamaProvider
from autopilot.providers.openai_provider import OpenAIProvider


class UnsupportedProviderError(ValueError):
    pass


def _default_provider(name: str) -> Provider:
    if name == "openai":
        return OpenAIProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "ollama":
        return OllamaProvider()
    raise UnsupportedProviderError(f"Unknown provider: {name}")


async def send_request(
    prompt: str,
    config: ModelConfig,
    *,
    provider: Provider | None = None,
) -> Response:
    p = provider if provider is not None else _default_provider(config.provider)
    return await p.complete(prompt, config)
