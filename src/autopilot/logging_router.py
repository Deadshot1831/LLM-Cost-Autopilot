from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone

from autopilot.db import RequestRecord, insert_request
from autopilot.verifying_router import VerifiedRoutedResponse, VerifyingRouter

PROMPT_PREVIEW_MAX = 200


class LoggingRouter:
    def __init__(
        self,
        *,
        verifying_router: VerifyingRouter,
        conn: sqlite3.Connection,
    ) -> None:
        self._vr = verifying_router
        self._conn = conn

    async def route_request(self, prompt: str) -> VerifiedRoutedResponse:
        result = await self._vr.route_request(prompt)
        self._record(prompt, result)
        return result

    def _record(self, prompt: str, result: VerifiedRoutedResponse) -> None:
        cand = result.routed.response
        ref = result.verification.reference_response
        baseline_model = ref.model_id if ref is not None else cand.model_id
        baseline_cost = ref.cost if ref is not None else cand.cost
        record = RequestRecord(
            timestamp=datetime.now(timezone.utc),
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            prompt_preview=prompt[:PROMPT_PREVIEW_MAX],
            tier=result.routed.tier.value,
            candidate_model=cand.model_id,
            candidate_cost=cand.cost,
            candidate_latency_ms=cand.latency_ms,
            baseline_model=baseline_model,
            baseline_cost=baseline_cost,
            verdict=result.verification.result.verdict.value,
            verdict_score=result.verification.result.score,
            verdict_method=result.verification.result.method,
            escalated=result.escalation.escalated,
            final_model=result.final_response.model_id,
            final_cost=result.final_response.cost,
        )
        insert_request(self._conn, record)
