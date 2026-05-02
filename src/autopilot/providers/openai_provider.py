from __future__ import annotations

import os
import time

from openai import AsyncOpenAI

from autopilot.models import ModelConfig, Response


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key."
            )
        self._client = AsyncOpenAI(api_key=key)

    async def complete(self, prompt: str, config: ModelConfig) -> Response:
        start = time.perf_counter()
        completion = await self._client.chat.completions.create(
            model=config.model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = completion.choices[0].message.content or ""
        usage = completion.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return Response(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost=config.compute_cost(input_tokens, output_tokens),
            model_id=config.model_id,
        )
