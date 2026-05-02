from autopilot.models import ComplexityTier, ModelConfig, Response
from autopilot.providers.base import Provider


class _FakeProvider:
    name = "fake"

    async def complete(self, prompt: str, config: ModelConfig) -> Response:
        return Response(
            text="ok",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1.0,
            cost=0.0,
            model_id=config.model_id,
        )


def test_fake_satisfies_protocol():
    fake: Provider = _FakeProvider()
    assert fake.name == "fake"


async def test_fake_returns_response():
    fake = _FakeProvider()
    cfg = ModelConfig(
        provider="fake",
        model_id="x",
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        avg_latency_ms=1,
        quality_tier=ComplexityTier.SIMPLE,
    )
    r = await fake.complete("hi", cfg)
    assert r.text == "ok"
