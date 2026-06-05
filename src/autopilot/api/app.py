from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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

STATIC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "static"


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="LLM Cost Autopilot", version="0.1.0")
    app.state.autopilot = state

    # Mount the chat UI: GET / serves the SPA; /static/* serves any additional assets.
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def chat_ui() -> FileResponse:
            return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(req: CompletionRequest) -> CompletionResponse:
        s: AppState = app.state.autopilot
        try:
            result = await s.logging_router.route_request(req.prompt)
        except ValueError as e:
            # OpenAIProvider raises ValueError when OPENAI_API_KEY is missing.
            if "api key" in str(e).lower():
                raise HTTPException(
                    status_code=503,
                    detail="OPENAI_API_KEY not configured on the server. Add it to .env and restart.",
                )
            raise
        except Exception as e:
            # Surface OpenAI/Anthropic SDK errors with their actual messages
            # instead of letting them bubble up as a generic 500.
            name = type(e).__name__
            if name in {"RateLimitError", "APIStatusError", "AuthenticationError",
                        "PermissionDeniedError", "BadRequestError", "APIError"}:
                # Map quota errors to 402 (Payment Required) for clarity in the UI.
                status = 402 if "quota" in str(e).lower() or "billing" in str(e).lower() else 502
                # Pull just the human-friendly message if the SDK exposes it.
                msg = getattr(e, "message", None) or str(e)
                raise HTTPException(status_code=status, detail=f"{name}: {msg}")
            raise
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
