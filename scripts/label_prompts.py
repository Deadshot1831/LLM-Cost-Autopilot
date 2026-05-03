"""Append labeled prompts to data/prompts_labeled.jsonl."""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "prompts_labeled.jsonl"


def append(prompt: str, tier: str) -> None:
    with DATA.open("a") as f:
        f.write(json.dumps({"prompt": prompt, "tier": tier}) + "\n")


def add_batch(entries: list[tuple[str, str]]) -> None:
    for prompt, tier in entries:
        if tier not in {"simple", "moderate", "complex"}:
            raise ValueError(f"bad tier: {tier}")
        append(prompt, tier)


if __name__ == "__main__":
    BATCH: list[tuple[str, str]] = [
        # Edit and re-run: uv run python scripts/label_prompts.py
    ]
    add_batch(BATCH)
    print(f"Appended {len(BATCH)} entries.")
