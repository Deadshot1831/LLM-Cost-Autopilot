from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from autopilot.escalation import EscalationDecision, escalate_on_fail, log_failure
from autopilot.models import Response
from autopilot.router import RoutedResponse, Router
from autopilot.verifier import VerificationEvent, Verifier


@dataclass(frozen=True)
class VerifiedRoutedResponse:
    routed: RoutedResponse
    verification: VerificationEvent
    escalation: EscalationDecision
    final_response: Response


class VerifyingRouter:
    def __init__(
        self,
        *,
        base_router: Router,
        verifier: Verifier,
        failure_log_path: Optional[Union[Path, str]] = None,
    ) -> None:
        self._base = base_router
        self._verifier = verifier
        self._failure_log_path = Path(failure_log_path) if failure_log_path else None

    async def route_request(self, prompt: str) -> VerifiedRoutedResponse:
        routed = await self._base.route_request(prompt)
        event = await self._verifier.verify(prompt=prompt, candidate=routed.response)
        decision = escalate_on_fail(event)
        if self._failure_log_path is not None:
            log_failure(event, self._failure_log_path)
        return VerifiedRoutedResponse(
            routed=routed,
            verification=event,
            escalation=decision,
            final_response=decision.final_response,
        )
