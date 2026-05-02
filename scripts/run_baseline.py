"""Send the 10 baseline prompts to all registered models and print a comparison table."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from autopilot.client import send_request
from autopilot.registry import load_registry

load_dotenv()

PROMPTS = [
    "What is 2 + 2?",
    "Translate 'hello' to French.",
    "Summarize: The quick brown fox jumps over the lazy dog.",
    "Extract the email from: Contact me at jane@example.com.",
    "Classify sentiment (pos/neg): I love this product.",
    "Write a haiku about autumn.",
    "List three benefits of exercise.",
    "Convert 100 km to miles.",
    "What language is 'Bonjour le monde'?",
    "Reverse the string 'autopilot'.",
]


async def _run_one(prompt: str, cfg, idx: int) -> dict:
    r = await send_request(prompt, cfg)
    return {
        "idx": idx,
        "model": cfg.model_id,
        "cost": r.cost,
        "latency_ms": round(r.latency_ms, 1),
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "preview": r.text[:60].replace("\n", " "),
    }


async def main() -> None:
    registry = load_registry(Path(__file__).resolve().parent.parent / "config" / "models.yaml")
    rows: list[dict] = []
    for cfg in registry.models.values():
        if cfg.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            print(f"[skip] {cfg.model_id} - OPENAI_API_KEY not set")
            continue
        for i, prompt in enumerate(PROMPTS, start=1):
            rows.append(await _run_one(prompt, cfg, i))

    print(f"{'idx':>3} {'model':<24} {'cost':>10} {'lat_ms':>8} {'in':>5} {'out':>5}  preview")
    print("-" * 100)
    for row in rows:
        print(
            f"{row['idx']:>3} {row['model']:<24} {row['cost']:>10.6f} "
            f"{row['latency_ms']:>8} {row['input_tokens']:>5} {row['output_tokens']:>5}  {row['preview']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
