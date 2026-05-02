import pytest

from autopilot.models import ComplexityTier, ModelConfig, Response


class TestComplexityTier:
    def test_three_tiers_exist(self):
        assert ComplexityTier.SIMPLE.value == "simple"
        assert ComplexityTier.MODERATE.value == "moderate"
        assert ComplexityTier.COMPLEX.value == "complex"

    def test_ordering(self):
        assert ComplexityTier.SIMPLE < ComplexityTier.MODERATE
        assert ComplexityTier.MODERATE < ComplexityTier.COMPLEX


class TestModelConfig:
    def test_construction(self):
        cfg = ModelConfig(
            provider="openai",
            model_id="gpt-4o-mini",
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006,
            avg_latency_ms=400,
            quality_tier=ComplexityTier.MODERATE,
        )
        assert cfg.provider == "openai"
        assert cfg.model_id == "gpt-4o-mini"

    def test_compute_cost(self):
        cfg = ModelConfig(
            provider="openai",
            model_id="gpt-4o-mini",
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006,
            avg_latency_ms=400,
            quality_tier=ComplexityTier.MODERATE,
        )
        cost = cfg.compute_cost(input_tokens=1000, output_tokens=500)
        assert cost == pytest.approx(0.00015 + 0.0003)

    def test_negative_costs_rejected(self):
        with pytest.raises(ValueError):
            ModelConfig(
                provider="openai",
                model_id="x",
                input_cost_per_1k=-1,
                output_cost_per_1k=0.0,
                avg_latency_ms=100,
                quality_tier=ComplexityTier.SIMPLE,
            )


class TestResponse:
    def test_construction(self):
        r = Response(
            text="hello",
            input_tokens=10,
            output_tokens=5,
            latency_ms=123.4,
            cost=0.0001,
            model_id="gpt-4o-mini",
        )
        assert r.text == "hello"
        assert r.total_tokens == 15
