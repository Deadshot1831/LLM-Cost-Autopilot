from __future__ import annotations

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt")


class CompletionMeta(BaseModel):
    tier: str
    candidate_model: str
    final_model: str
    escalated: bool
    verdict: str
    verdict_score: float
    verdict_method: str
    routing_reason: str
    final_cost: float
    final_latency_ms: float


class CompletionResponse(BaseModel):
    text: str
    meta: CompletionMeta


class ModelInfo(BaseModel):
    model_id: str
    provider: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    avg_latency_ms: int
    quality_tier: str


class ModelsListResponse(BaseModel):
    models: list[ModelInfo]


class StatsResponse(BaseModel):
    total_requests: int
    final_cost_total: float
    baseline_cost_total: float
    savings_total: float
    savings_pct: float


class RoutingConfigRequest(BaseModel):
    simple: str = Field(..., description="model_id for SIMPLE tier")
    moderate: str = Field(..., description="model_id for MODERATE tier")
    complex: str = Field(..., description="model_id for COMPLEX tier")


class RoutingConfigResponse(BaseModel):
    simple: str
    moderate: str
    complex: str
