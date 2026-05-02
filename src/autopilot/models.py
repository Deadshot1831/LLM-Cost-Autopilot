from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import total_ordering


@total_ordering
class ComplexityTier(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

    def _rank(self) -> int:
        return {"simple": 0, "moderate": 1, "complex": 2}[self.value]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ComplexityTier):
            return NotImplemented
        return self._rank() < other._rank()


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_id: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    avg_latency_ms: int
    quality_tier: ComplexityTier

    def __post_init__(self) -> None:
        if self.input_cost_per_1k < 0 or self.output_cost_per_1k < 0:
            raise ValueError("Costs must be non-negative")
        if self.avg_latency_ms < 0:
            raise ValueError("Latency must be non-negative")

    def compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1000.0) * self.input_cost_per_1k
            + (output_tokens / 1000.0) * self.output_cost_per_1k
        )


@dataclass(frozen=True)
class Response:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost: float
    model_id: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
