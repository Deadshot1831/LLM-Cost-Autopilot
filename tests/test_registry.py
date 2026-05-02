from pathlib import Path

import pytest

from autopilot.models import ComplexityTier
from autopilot.registry import ModelNotFoundError, load_registry


@pytest.fixture
def registry_path(project_root: Path) -> Path:
    return project_root / "config" / "models.yaml"


class TestLoadRegistry:
    def test_loads_all_five_models(self, registry_path: Path):
        registry = load_registry(registry_path)
        assert len(registry) == 5

    def test_lookup_by_model_id(self, registry_path: Path):
        registry = load_registry(registry_path)
        cfg = registry.get("gpt-4o-mini")
        assert cfg.provider == "openai"
        assert cfg.quality_tier == ComplexityTier.MODERATE

    def test_unknown_model_raises(self, registry_path: Path):
        registry = load_registry(registry_path)
        with pytest.raises(ModelNotFoundError):
            registry.get("does-not-exist")

    def test_list_models(self, registry_path: Path):
        registry = load_registry(registry_path)
        ids = registry.list_ids()
        assert "gpt-4o" in ids
        assert "llama3.2:3b" in ids

    def test_filter_by_tier(self, registry_path: Path):
        registry = load_registry(registry_path)
        complex_models = registry.by_tier(ComplexityTier.COMPLEX)
        ids = {m.model_id for m in complex_models}
        assert ids == {"gpt-4o", "claude-sonnet-4-6"}
