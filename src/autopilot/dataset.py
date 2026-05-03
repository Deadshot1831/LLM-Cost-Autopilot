from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from autopilot.models import ComplexityTier


@dataclass(frozen=True)
class LabeledPrompt:
    prompt: str
    tier: ComplexityTier


def load_dataset(path: Path | str) -> list[LabeledPrompt]:
    path = Path(path)
    rows: list[LabeledPrompt] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno} bad JSON: {e}") from e
        try:
            tier = ComplexityTier(obj["tier"])
        except (KeyError, ValueError) as e:
            raise ValueError(f"{path}:{lineno} bad tier: {e}") from e
        prompt = (obj.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"{path}:{lineno} empty prompt")
        rows.append(LabeledPrompt(prompt=prompt, tier=tier))
    return rows
