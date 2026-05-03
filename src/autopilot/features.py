from __future__ import annotations

import re
from dataclasses import dataclass

INSTRUCTION_VERBS = {
    "analyze", "compare", "evaluate", "assess", "explain", "justify",
    "synthesize", "argue", "debate", "predict", "design", "optimize",
    "critique", "reason", "derive", "prove",
}

CONSTRAINT_PATTERNS = [
    r"\bexactly\s+\d+\b",
    r"\bno more than\b",
    r"\bat least\b",
    r"\bmust\b",
    r"\bmust not\b",
    r"\bonly\b",
    r"\bin (json|yaml|xml|csv|markdown)\b",
    r"\bwithin \d+\b",
]

STRUCTURED_FORMATS = ["json", "yaml", "xml", "csv", "markdown table", "table", "list"]


@dataclass(frozen=True)
class PromptFeatures:
    token_count: int
    instruction_verb_count: int
    constraint_count: int
    has_context: bool
    output_format_complexity: int

    def to_vector(self) -> list[float]:
        return [
            float(self.token_count),
            float(self.instruction_verb_count),
            float(self.constraint_count),
            1.0 if self.has_context else 0.0,
            float(self.output_format_complexity),
        ]


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _instruction_verbs(text: str) -> int:
    lowered = text.lower()
    words = set(re.findall(r"[a-z]+", lowered))
    return sum(1 for v in INSTRUCTION_VERBS if v in words)


def _constraints(text: str) -> int:
    lowered = text.lower()
    return sum(1 for pattern in CONSTRAINT_PATTERNS if re.search(pattern, lowered))


def _has_context(text: str) -> bool:
    if '"""' in text or "'''" in text or "```" in text:
        return True
    return _word_count(text) > 200


def _output_format_complexity(text: str) -> int:
    lowered = text.lower()
    return sum(1 for fmt in STRUCTURED_FORMATS if fmt in lowered)


def extract_features(prompt: str) -> PromptFeatures:
    return PromptFeatures(
        token_count=_word_count(prompt),
        instruction_verb_count=_instruction_verbs(prompt),
        constraint_count=_constraints(prompt),
        has_context=_has_context(prompt),
        output_format_complexity=_output_format_complexity(prompt),
    )
