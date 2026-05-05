"""Send N diverse prompts through the LoggingRouter so the dashboard has data."""
from __future__ import annotations

import argparse
import asyncio
import os
import random
from pathlib import Path

from dotenv import load_dotenv

from autopilot.classifier import ComplexityClassifier
from autopilot.db import open_db
from autopilot.logging_router import LoggingRouter
from autopilot.registry import load_registry
from autopilot.router import Router
from autopilot.routing import load_routing_config
from autopilot.verifier import Verifier
from autopilot.verifying_router import VerifyingRouter

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent

PROMPTS = [
    # SIMPLE
    "What is 17 + 25?",
    "Translate 'good morning' to Spanish.",
    "Convert 50 km to miles.",
    "Reverse the string 'autopilot'.",
    "What is the capital of Brazil?",
    "What is the chemical symbol for iron?",
    # MODERATE
    "Summarize: The quick brown fox jumps over the lazy dog.",
    "Classify the sentiment of: I'm disappointed with the service.",
    "Write a short professional email rescheduling a one-on-one.",
    "Generate a JSON object with 'name' and 'role' fields for a backend engineer.",
    "Explain what a database index is in two sentences.",
    "Extract action items from: 'Bob will draft the spec by Friday. Alice owns design.'",
    # COMPLEX
    "Compare the trade-offs between SQL and NoSQL for a fintech ledger, considering consistency and operational complexity.",
    "Design a sharding strategy for a multi-tenant Postgres database with hot tenants.",
    "Critique a microservices architecture where 30 services share a single Postgres instance.",
    "Synthesize a 6-month plan to migrate a Rails monolith to services without freezing features.",
    "Predict three failure modes of a distributed Kafka pipeline under network partition.",
    "Argue both sides of building vs buying an internal feature flag system for a 200-engineer org.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--count", type=int, default=30, help="Number of prompts to send")
    p.add_argument("--db", type=str, default=str(ROOT / "data" / "autopilot.db"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set - load test needs the key (reference model is gpt-4o).")
        return

    rng = random.Random(args.seed)
    classifier = ComplexityClassifier.load(ROOT / "models" / "classifier.joblib")
    registry = load_registry(ROOT / "config" / "models.yaml")
    routing = load_routing_config(ROOT / "config" / "routing.yaml")
    base = Router(classifier=classifier, routing=routing, registry=registry)
    verifier = Verifier(reference_cfg=registry.get("gpt-4o"))
    vr = VerifyingRouter(
        base_router=base, verifier=verifier,
        failure_log_path=ROOT / "data" / "routing_failures.jsonl",
    )
    conn = open_db(args.db)
    lr = LoggingRouter(verifying_router=vr, conn=conn)

    selected = [rng.choice(PROMPTS) for _ in range(args.count)]
    print(f"Sending {len(selected)} prompts...")
    for i, prompt in enumerate(selected, start=1):
        result = await lr.route_request(prompt)
        print(
            f"[{i:>3}/{len(selected)}] tier={result.routed.tier.value:<8} "
            f"model={result.final_response.model_id:<14} "
            f"verdict={result.verification.result.verdict.value:<5} "
            f"esc={'Y' if result.escalation.escalated else 'N'}"
        )
    print("Done. Open the dashboard: ./scripts/run_dashboard.sh")


if __name__ == "__main__":
    asyncio.run(main())
