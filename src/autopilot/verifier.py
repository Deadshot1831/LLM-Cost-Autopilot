from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Optional

from autopilot.client import send_request as default_send
from autopilot.models import ModelConfig, Response
from autopilot.quality import (
    EXACT_MATCH_THRESHOLD,
    JUDGE_THRESHOLD,
    QualityVerdict,
    VerdictResult,
    exact_match_score,
    is_short_prompt,
    score_semantic,
)

SendFn = Callable[..., Awaitable[Response]]
ScoringMethod = Literal["semantic", "exact_match"]


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
        scoring_method: ScoringMethod = "exact_match",
    ) -> None:
        if not (0.0 <= sample_rate <= 1.0):
            raise ValueError("sample_rate must be in [0, 1]")
        if scoring_method not in ("semantic", "exact_match"):
            raise ValueError(f"unknown scoring_method: {scoring_method!r}")
        self._reference_cfg = reference_cfg
        self._send = send
        self._sample_rate = sample_rate
        self._judge_prompt = judge_prompt
        self._rng = random.Random(rng_seed)
        self._scoring_method = scoring_method

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
            if self._scoring_method == "semantic":
                # Cosine similarity over MiniLM embeddings. Catches semantic
                # equivalence regardless of verbosity (e.g. "Tokyo." vs the
                # reference's 25-word answer about Tokyo). Falls back to
                # exact-match if the embedding stack fails to load at runtime.
                try:
                    from autopilot.embeddings import cosine_similarity
                    s = await cosine_similarity(candidate_text, reference_text)
                    return score_semantic(s)
                except Exception as e:
                    # Don't break the verifier on encoder load failure;
                    # degrade to exact_match and record the reason.
                    s = exact_match_score(candidate_text, reference_text)
                    verdict = (
                        QualityVerdict.PASS if s >= EXACT_MATCH_THRESHOLD
                        else QualityVerdict.FAIL
                    )
                    return VerdictResult(
                        verdict, s, "exact_match",
                        f"jaccard={s:.2f} (semantic unavailable: {type(e).__name__})",
                    )
            # Explicit exact_match path
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
