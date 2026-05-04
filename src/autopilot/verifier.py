from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from autopilot.client import send_request as default_send
from autopilot.models import ModelConfig, Response
from autopilot.quality import (
    EXACT_MATCH_THRESHOLD,
    JUDGE_THRESHOLD,
    QualityVerdict,
    VerdictResult,
    exact_match_score,
    is_short_prompt,
)

SendFn = Callable[..., Awaitable[Response]]


def _default_judge_prompt(prompt: str, candidate: str, reference: str) -> str:
    return (
        "Evaluate response A against reference response B for the same prompt. "
        "Score A on a 1-5 scale (5=equivalent, 1=wrong). "
        "Respond with ONLY a single integer 1-5.\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"RESPONSE A:\n{candidate}\n\n"
        f"RESPONSE B:\n{reference}\n"
    )


@dataclass(frozen=True)
class VerificationEvent:
    prompt: str
    candidate: Response
    reference_response: Optional[Response]
    result: VerdictResult
    cost_delta: float


class Verifier:
    def __init__(
        self,
        *,
        reference_cfg: ModelConfig,
        send: SendFn = default_send,
        sample_rate: float = 1.0,
        judge_prompt: Callable[[str, str, str], str] = _default_judge_prompt,
        rng_seed: int = 0,
    ) -> None:
        if not (0.0 <= sample_rate <= 1.0):
            raise ValueError("sample_rate must be in [0, 1]")
        self._reference_cfg = reference_cfg
        self._send = send
        self._sample_rate = sample_rate
        self._judge_prompt = judge_prompt
        self._rng = random.Random(rng_seed)

    async def verify(self, *, prompt: str, candidate: Response) -> VerificationEvent:
        if candidate.model_id == self._reference_cfg.model_id:
            return VerificationEvent(
                prompt=prompt, candidate=candidate, reference_response=None,
                result=VerdictResult(QualityVerdict.SKIP, 0.0, "skip", "candidate is reference"),
                cost_delta=0.0,
            )
        if self._rng.random() >= self._sample_rate:
            return VerificationEvent(
                prompt=prompt, candidate=candidate, reference_response=None,
                result=VerdictResult(QualityVerdict.SKIP, 0.0, "skip", "sampled out"),
                cost_delta=0.0,
            )

        reference = await self._send(prompt, self._reference_cfg)
        result = await self._score(prompt, candidate.text, reference.text)
        return VerificationEvent(
            prompt=prompt,
            candidate=candidate,
            reference_response=reference,
            result=result,
            cost_delta=reference.cost - candidate.cost,
        )

    async def _score(self, prompt: str, candidate_text: str, reference_text: str) -> VerdictResult:
        if not reference_text.strip():
            return VerdictResult(QualityVerdict.SKIP, 0.0, "skip", "empty reference")

        if is_short_prompt(prompt):
            s = exact_match_score(candidate_text, reference_text)
            verdict = QualityVerdict.PASS if s >= EXACT_MATCH_THRESHOLD else QualityVerdict.FAIL
            return VerdictResult(verdict, s, "exact_match", f"jaccard={s:.2f}")

        judge_prompt = self._judge_prompt(prompt, candidate_text, reference_text)
        judge_resp = await self._send(judge_prompt, self._reference_cfg)
        match = re.search(r"[1-5]", judge_resp.text)
        if not match:
            return VerdictResult(
                QualityVerdict.SKIP, 0.0, "judge",
                f"unparseable: {judge_resp.text[:40]!r}",
            )
        score_val = float(match.group(0))
        verdict = QualityVerdict.PASS if score_val >= JUDGE_THRESHOLD else QualityVerdict.FAIL
        return VerdictResult(verdict, score_val, "judge", f"score={score_val:.1f}/5")
