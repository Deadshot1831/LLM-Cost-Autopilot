"""Run a sample batch through the full router and print per-prompt routing decisions."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from autopilot.classifier import ComplexityClassifier
from autopilot.registry import load_registry
from autopilot.router import Router
from autopilot.routing import load_routing_config

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent

SAMPLE = [
    "What is 2 + 2?",
    "Translate 'thank you' to Japanese.",
    "Summarize the plot of Hamlet in two sentences.",
    "Classify this support ticket: 'My package is late.' (categories: shipping, billing, technical, other)",
    "Design a sharding strategy for a multi-tenant Postgres database with hot tenants.",
    "Compare and analyze the trade-offs between event sourcing and CRUD for a fintech ledger.",
]


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set - this demo routes to gpt-4o/4o-mini and needs the key.")
        return

    classifier = ComplexityClassifier.load(ROOT / "models" / "classifier.joblib")
    routing = load_routing_config(ROOT / "config" / "routing.yaml")
    registry = load_registry(ROOT / "config" / "models.yaml")
    router = Router(classifier=classifier, routing=routing, registry=registry)

    print(f"{'tier':<10} {'model':<14} {'cost':>10} {'lat_ms':>8}  prompt")
    print("-" * 100)
    for prompt in SAMPLE:
        result = await router.route_request(prompt)
        r = result.response
        print(
            f"{result.tier.value:<10} {r.model_id:<14} {r.cost:>10.6f} "
            f"{r.latency_ms:>8.0f}  {prompt[:60]}"
        )


if __name__ == "__main__":
    asyncio.run(main())
