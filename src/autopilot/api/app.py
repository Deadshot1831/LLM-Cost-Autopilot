from __future__ import annotations

from fastapi import FastAPI, HTTPException

from autopilot.api.schemas import (
    CompletionMeta,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelsListResponse,
    RoutingConfigRequest,
    RoutingConfigResponse,
    StatsResponse,
)
from autopilot.api.state import AppState
from autopilot.db import query_aggregate_costs
from autopilot.models import ComplexityTier
from autopilot.registry import ModelNotFoundError


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="LLM Cost Autopilot", version="0.1.0")
    app.state.autopilot = state

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(req: CompletionRequest) -> CompletionResponse:
        s: AppState = app.state.autopilot
        result = await s.logging_router.route_request(req.prompt)
        return CompletionResponse(
            text=result.final_response.text,
            meta=CompletionMeta(
                tier=result.routed.tier.value,
                candidate_model=result.routed.response.model_id,
                final_model=result.final_response.model_id,
                escalated=result.escalation.escalated,
                verdict=result.verification.result.verdict.value,
                verdict_score=result.verification.result.score,
                verdict_method=result.verification.result.method,
                routing_reason=result.routed.routing_reason,
                final_cost=result.final_response.cost,
                final_latency_ms=result.final_response.latency_ms,
            ),
        )

    @app.get("/v1/models", response_model=ModelsListResponse)
    def list_models() -> ModelsListResponse:
        s: AppState = app.state.autopilot
        models = [
            ModelInfo(
                model_id=cfg.model_id,
                provider=cfg.provider,
                input_cost_per_1k=cfg.input_cost_per_1k,
                output_cost_per_1k=cfg.output_cost_per_1k,
                avg_latency_ms=cfg.avg_latency_ms,
                quality_tier=cfg.quality_tier.value,
            )
            for cfg in s.registry.models.values()
        ]
        return ModelsListResponse(models=models)

    @app.get("/v1/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        s: AppState = app.state.autopilot
        agg = query_aggregate_costs(s.db_conn)
        return StatsResponse(**agg)

    @app.get("/v1/routing-config", response_model=RoutingConfigResponse)
    def get_routing() -> RoutingConfigResponse:
        s: AppState = app.state.autopilot
        return RoutingConfigResponse(
            simple=s.routing[ComplexityTier.SIMPLE],
            moderate=s.routing[ComplexityTier.MODERATE],
            complex=s.routing[ComplexityTier.COMPLEX],
        )

    @app.put("/v1/routing-config", response_model=RoutingConfigResponse)
    def update_routing(req: RoutingConfigRequest) -> RoutingConfigResponse:
        s: AppState = app.state.autopilot
        new_map = {
            ComplexityTier.SIMPLE: req.simple,
            ComplexityTier.MODERATE: req.moderate,
            ComplexityTier.COMPLEX: req.complex,
        }
        try:
            s.update_routing(new_map)
        except ModelNotFoundError as e:
            raise HTTPException(status_code=400, detail=f"Unknown model_id: {e}")
        return RoutingConfigResponse(
            simple=s.routing[ComplexityTier.SIMPLE],
            moderate=s.routing[ComplexityTier.MODERATE],
            complex=s.routing[ComplexityTier.COMPLEX],
        )

    return app
