from __future__ import annotations

import os
import time

from autopilot.models import ModelConfig, Response


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class AnthropicProvider:
    """Real Anthropic SDK if ANTHROPIC_API_KEY is set; deterministic mock otherwise."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        self._real = bool(key)
        if self._real:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=key)

    async def complete(self, prompt: str, config: ModelConfig) -> Response:
        if self._real:
            return await self._complete_real(prompt, config)
        return self._complete_mock(prompt, config)

    async def _complete_real(self, prompt: str, config: ModelConfig) -> Response:
        start = time.perf_counter()
        msg = await self._client.messages.create(
            model=config.model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = "".join(b.text for b in msg.content if hasattr(b, "text"))
        usage = msg.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        return Response(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost=config.compute_cost(input_tokens, output_tokens),
            model_id=config.model_id,
        )

    def _complete_mock(self, prompt: str, config: ModelConfig) -> Response:
        start = time.perf_counter()
        text = f"[mock anthropic:{config.model_id}] response to: {prompt[:60]}"
        input_tokens = _approx_tokens(prompt)
        output_tokens = _approx_tokens(text)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return Response(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost=config.compute_cost(input_tokens, output_tokens),
            model_id=config.model_id,
        )
