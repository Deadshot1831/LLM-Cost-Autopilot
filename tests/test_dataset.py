from pathlib import Path

import pytest

from autopilot.dataset import LabeledPrompt, load_dataset
from autopilot.models import ComplexityTier


@pytest.fixture
def dataset_path(project_root: Path) -> Path:
    return project_root / "data" / "prompts_labeled.jsonl"


class TestLoadDataset:
    def test_loads_at_least_30(self, dataset_path: Path):
        rows = load_dataset(dataset_path)
        assert len(rows) >= 30

    def test_returns_labeled_prompts(self, dataset_path: Path):
        rows = load_dataset(dataset_path)
        assert all(isinstance(r, LabeledPrompt) for r in rows)
        assert all(isinstance(r.tier, ComplexityTier) for r in rows)

    def test_each_tier_has_examples(self, dataset_path: Path):
        rows = load_dataset(dataset_path)
        tiers = {r.tier for r in rows}
        assert tiers == {ComplexityTier.SIMPLE, ComplexityTier.MODERATE, ComplexityTier.COMPLEX}

    def test_no_empty_prompts(self, dataset_path: Path):
        rows = load_dataset(dataset_path)
        assert all(r.prompt.strip() for r in rows)

    def test_unknown_tier_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"prompt": "x", "tier": "bogus"}\n')
        with pytest.raises(ValueError):
            load_dataset(bad)
