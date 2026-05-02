from __future__ import annotations

import time

from autopilot.models import ModelConfig, Response


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class OllamaProvider:
    name = "ollama"

    async def complete(self, prompt: str, config: ModelConfig) -> Response:
        start = time.perf_counter()
        text = f"[mock ollama:{config.model_id}] response to: {prompt[:60]}"
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
