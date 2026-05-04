from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

EXACT_MATCH_THRESHOLD = 0.7
JUDGE_THRESHOLD = 4.0
SHORT_PROMPT_TOKEN_LIMIT = 30


class QualityVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class VerdictResult:
    verdict: QualityVerdict
    score: float
    method: str
    detail: str = ""


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def exact_match_score(candidate: str, reference: str) -> float:
    """Token-set Jaccard similarity, case-insensitive."""
    if not reference.strip() or not candidate.strip():
        return 0.0
    cand = _tokens(candidate)
    ref = _tokens(reference)
    if not cand or not ref:
        return 0.0
    return len(cand & ref) / len(cand | ref)


def score_exact_match(candidate: str, reference: str) -> VerdictResult:
    if not reference.strip():
        return VerdictResult(QualityVerdict.SKIP, 0.0, "exact_match", "empty reference")
    s = exact_match_score(candidate, reference)
    verdict = QualityVerdict.PASS if s >= EXACT_MATCH_THRESHOLD else QualityVerdict.FAIL
    return VerdictResult(verdict, s, "exact_match", f"jaccard={s:.2f}")


def is_short_prompt(prompt: str) -> bool:
    return len(prompt.split()) <= SHORT_PROMPT_TOKEN_LIMIT
