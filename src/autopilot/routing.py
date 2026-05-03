from __future__ import annotations

from pathlib import Path

import yaml

from autopilot.models import ComplexityTier, ModelConfig
from autopilot.registry import ModelRegistry


class InvalidRoutingConfig(ValueError):
    pass


def load_routing_config(path: Path | str) -> dict[ComplexityTier, str]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not raw or "routing" not in raw:
        raise InvalidRoutingConfig(f"{path}: missing 'routing' key")
    routing_raw = raw["routing"]
    expected = {t.value for t in ComplexityTier}
    actual = set(routing_raw.keys())
    if actual != expected:
        raise InvalidRoutingConfig(
            f"{path}: routing must define exactly {expected}, got {actual}"
        )
    return {ComplexityTier(k): str(v) for k, v in routing_raw.items()}


def pick_model(
    tier: ComplexityTier,
    routing: dict[ComplexityTier, str],
    registry: ModelRegistry,
) -> ModelConfig:
    model_id = routing[tier]
    return registry.get(model_id)
