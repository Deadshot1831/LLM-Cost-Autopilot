from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from autopilot.models import ComplexityTier, ModelConfig


class ModelNotFoundError(KeyError):
    pass


@dataclass(frozen=True)
class ModelRegistry:
    models: dict[str, ModelConfig] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.models)

    def get(self, model_id: str) -> ModelConfig:
        try:
            return self.models[model_id]
        except KeyError as e:
            raise ModelNotFoundError(model_id) from e

    def list_ids(self) -> list[str]:
        return list(self.models.keys())

    def by_tier(self, tier: ComplexityTier) -> list[ModelConfig]:
        return [m for m in self.models.values() if m.quality_tier == tier]


def load_registry(path: Path | str) -> ModelRegistry:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not raw or "models" not in raw:
        raise ValueError(f"Registry {path} has no 'models' key")
    models: dict[str, ModelConfig] = {}
    for entry in raw["models"]:
        cfg = ModelConfig(
            provider=entry["provider"],
            model_id=entry["model_id"],
            input_cost_per_1k=float(entry["input_cost_per_1k"]),
            output_cost_per_1k=float(entry["output_cost_per_1k"]),
            avg_latency_ms=int(entry["avg_latency_ms"]),
            quality_tier=ComplexityTier(entry["quality_tier"]),
        )
        if cfg.model_id in models:
            raise ValueError(f"Duplicate model_id: {cfg.model_id}")
        models[cfg.model_id] = cfg
    return ModelRegistry(models=models)
