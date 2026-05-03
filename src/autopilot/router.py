from __future__ import annotations

from dataclasses import dataclass

from autopilot.classifier import ComplexityClassifier
from autopilot.client import send_request
from autopilot.models import ComplexityTier, Response
from autopilot.registry import ModelRegistry
from autopilot.routing import pick_model


@dataclass(frozen=True)
class RoutedResponse:
    response: Response
    tier: ComplexityTier
    routing_reason: str


class Router:
    def __init__(
        self,
        *,
        classifier: ComplexityClassifier,
        routing: dict[ComplexityTier, str],
        registry: ModelRegistry,
    ) -> None:
        self._classifier = classifier
        self._routing = routing
        self._registry = registry

    async def route_request(self, prompt: str) -> RoutedResponse:
        tier, confidence = self._classifier.predict_with_confidence(prompt)
        cfg = pick_model(tier, self._routing, self._registry)
        reason = (
            f"tier={tier.value} confidence={confidence:.2f} "
            f"-> model={cfg.model_id} (provider={cfg.provider})"
        )
        response = await send_request(prompt, cfg)
        return RoutedResponse(response=response, tier=tier, routing_reason=reason)
