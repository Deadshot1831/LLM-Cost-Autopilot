from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from autopilot.models import Response
from autopilot.quality import QualityVerdict
from autopilot.verifier import VerificationEvent


@dataclass(frozen=True)
class EscalationDecision:
    escalated: bool
    final_response: Response
    reason: str


def escalate_on_fail(event: VerificationEvent) -> EscalationDecision:
    if event.result.verdict == QualityVerdict.FAIL and event.reference_response is not None:
        return EscalationDecision(
            escalated=True,
            final_response=event.reference_response,
            reason=(
                f"verifier FAIL ({event.result.method} {event.result.detail}); "
                f"escalated to {event.reference_response.model_id}"
            ),
        )
    return EscalationDecision(
        escalated=False,
        final_response=event.candidate,
        reason=f"verifier {event.result.verdict.value}",
    )


def log_failure(event: VerificationEvent, path: Path | str) -> None:
    if event.result.verdict != QualityVerdict.FAIL:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": event.prompt,
        "candidate_model": event.candidate.model_id,
        "reference_model": event.reference_response.model_id if event.reference_response else None,
        "candidate_text": event.candidate.text,
        "reference_text": event.reference_response.text if event.reference_response else None,
        "score": event.result.score,
        "method": event.result.method,
        "cost_delta": event.cost_delta,
    }
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
