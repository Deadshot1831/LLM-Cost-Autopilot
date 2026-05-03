from pathlib import Path

import pytest

from autopilot.models import ComplexityTier
from autopilot.registry import load_registry
from autopilot.routing import (
    InvalidRoutingConfig,
    load_routing_config,
    pick_model,
)


@pytest.fixture
def routing_path(project_root: Path) -> Path:
    return project_root / "config" / "routing.yaml"


@pytest.fixture
def registry(project_root: Path):
    return load_registry(project_root / "config" / "models.yaml")


class TestLoadRoutingConfig:
    def test_loads_three_tiers(self, routing_path: Path):
        cfg = load_routing_config(routing_path)
        assert set(cfg.keys()) == {
            ComplexityTier.SIMPLE,
            ComplexityTier.MODERATE,
            ComplexityTier.COMPLEX,
        }

    def test_missing_tier_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("routing:\n  simple: gpt-4o-mini\n")
        with pytest.raises(InvalidRoutingConfig):
            load_routing_config(bad)


class TestPickModel:
    def test_picks_correct_model_for_each_tier(self, routing_path, registry):
        routing = load_routing_config(routing_path)
        for tier in (ComplexityTier.SIMPLE, ComplexityTier.MODERATE, ComplexityTier.COMPLEX):
            cfg = pick_model(tier, routing, registry)
            assert cfg.model_id == routing[tier]

    def test_pick_unknown_model_raises(self, registry, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "routing:\n  simple: nonexistent\n  moderate: gpt-4o-mini\n  complex: gpt-4o\n"
        )
        routing = load_routing_config(bad)
        with pytest.raises(KeyError):
            pick_model(ComplexityTier.SIMPLE, routing, registry)
