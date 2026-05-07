"""Run N prompts through the LoggingRouter with a mocked send_request.

Costs are computed from registry pricing as if the real APIs were called,
so the resulting savings % in the dashboard is a faithful estimate. No
real API credits burned. Useful for portfolio screenshots and the case
study report.

Usage:
    uv run python scripts/simulate_load.py -n 500
"""
from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path

from autopilot.classifier import ComplexityClassifier
from autopilot.db import open_db
from autopilot.logging_router import LoggingRouter
from autopilot.models import Response
from autopilot.registry import load_registry
from autopilot.router import Router
from autopilot.routing import load_routing_config
from autopilot.verifier import Verifier
from autopilot.verifying_router import VerifyingRouter

ROOT = Path(__file__).resolve().parent.parent

PROMPTS = [
    # SIMPLE
    "What is 17 + 25?",
    "Translate 'good morning' to Spanish.",
    "Convert 50 km to miles.",
    "Reverse the string 'autopilot'.",
    "What is the capital of Brazil?",
    "What is the chemical symbol for iron?",
    "How many sides does an octagon have?",
    "Convert 32 degrees Fahrenheit to Celsius.",
    "What year did World War 2 end?",
    "Lowercase the string 'HELLO WORLD'.",
    # MODERATE
    "Summarize: The quick brown fox jumps over the lazy dog.",
    "Classify the sentiment of: I'm disappointed with the service.",
    "Write a short professional email rescheduling a one-on-one.",
    "Generate a JSON object with 'name' and 'role' fields for a backend engineer.",
    "Explain what a database index is in two sentences.",
    "Extract action items from: 'Bob will draft the spec by Friday. Alice owns design.'",
    "Generate a markdown table comparing Python, Go, and Rust on syntax simplicity.",
    "Write a SQL query that returns the top 5 customers by total order amount.",
    "Categorize this support ticket as billing, shipping, or technical: 'My package is late.'",
    "Outline the steps for setting up a Postgres database in Docker.",
    # COMPLEX
    "Compare the trade-offs between SQL and NoSQL for a fintech ledger, considering consistency and operational complexity.",
    "Design a sharding strategy for a multi-tenant Postgres database with hot tenants.",
    "Critique a microservices architecture where 30 services share a single Postgres instance.",
    "Synthesize a 6-month plan to migrate a Rails monolith to services without freezing features.",
    "Predict three failure modes of a distributed Kafka pipeline under network partition.",
    "Argue both sides of building vs buying an internal feature flag system for a 200-engineer org.",
    "Derive a queue-based backpressure strategy for an async image processing pipeline that occasionally sees 10x traffic spikes.",
    "Design a multi-region disaster recovery strategy for a Postgres-backed SaaS, including RPO/RTO targets.",
    "Justify a choice between blue-green and canary deployments for a payment processing service with strict rollback SLAs.",
    "Reason about whether to acquire a small competitor or build the equivalent feature in-house.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--count", type=int, default=500)
    p.add_argument("--db", type=str, default=str(ROOT / "data" / "autopilot.db"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--escalation-rate", type=float, default=0.1,
        help="Fraction of prompts where the reference disagrees (triggers FAIL/escalation).",
    )
    return p.parse_args()


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _make_fake_send(rng: random.Random, escalation_rate: float):
    """Return responses that mostly agree with the candidate (so PASS),
    but `escalation_rate` fraction disagree token-wise (so FAIL ->
    escalate). Costs are computed exactly as the real provider would."""

    async def fake_send(prompt, config, *, provider=None):
        if rng.random() < escalation_rate:
            text = (
                "After analysis, this requires careful consideration of multiple "
                "factors that the cheaper model may have overlooked entirely."
            )
        else:
            words = prompt.split()[:20]
            text = "Response: " + " ".join(words) + " - acknowledged."
        input_tokens = _approx_tokens(prompt)
        output_tokens = _approx_tokens(text)
        return Response(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=rng.uniform(150, 450),
            cost=config.compute_cost(input_tokens, output_tokens),
            model_id=config.model_id,
        )

    return fake_send


async def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    classifier = ComplexityClassifier.load(ROOT / "models" / "classifier.joblib")
    registry = load_registry(ROOT / "config" / "models.yaml")
    routing = load_routing_config(ROOT / "config" / "routing.yaml")

    fake_send = _make_fake_send(rng, args.escalation_rate)

    base = Router(classifier=classifier, routing=routing, registry=registry)
    verifier = Verifier(reference_cfg=registry.get("gpt-4o"), send=fake_send)
    vr = VerifyingRouter(
        base_router=base, verifier=verifier,
        failure_log_path=ROOT / "data" / "routing_failures.jsonl",
    )
    conn = open_db(args.db)
    lr = LoggingRouter(verifying_router=vr, conn=conn)

    # Patch the Router's downstream send_request so candidate calls are mocked too.
    import autopilot.router as router_mod
    router_mod.send_request = fake_send  # type: ignore[assignment]

    selected = [rng.choice(PROMPTS) for _ in range(args.count)]
    print(f"Simulating {len(selected)} requests (escalation rate target: {args.escalation_rate:.0%})...")
    n_escalated = 0
    for i, prompt in enumerate(selected, start=1):
        result = await lr.route_request(prompt)
        if result.escalation.escalated:
            n_escalated += 1
        if i % 50 == 0:
            print(f"  [{i}/{len(selected)}] {n_escalated} escalations so far")
    print(f"Done. {n_escalated}/{len(selected)} escalated ({n_escalated / len(selected) * 100:.1f}%).")
    print(f"Database: {args.db}")
    print("Next: uv run python scripts/generate_report.py")


if __name__ == "__main__":
    asyncio.run(main())
