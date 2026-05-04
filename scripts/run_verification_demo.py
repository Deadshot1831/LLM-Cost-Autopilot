"""Run sample prompts through the VerifyingRouter and print a cost-savings table."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from autopilot.classifier import ComplexityClassifier
from autopilot.registry import load_registry
from autopilot.router import Router
from autopilot.routing import load_routing_config
from autopilot.verifier import Verifier
from autopilot.verifying_router import VerifyingRouter

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
        print("OPENAI_API_KEY not set - reference model is gpt-4o, this needs the key.")
        return

    registry = load_registry(ROOT / "config" / "models.yaml")
    classifier = ComplexityClassifier.load(ROOT / "models" / "classifier.joblib")
    routing = load_routing_config(ROOT / "config" / "routing.yaml")
    base = Router(classifier=classifier, routing=routing, registry=registry)
    verifier = Verifier(reference_cfg=registry.get("gpt-4o"))
    vr = VerifyingRouter(
        base_router=base,
        verifier=verifier,
        failure_log_path=ROOT / "data" / "routing_failures.jsonl",
    )

    baseline_total = 0.0
    final_total = 0.0

    print(f"{'tier':<10} {'cand':<14} {'verdict':<7} {'esc':<3} {'final':<14} {'final$':>10} {'baseline$':>10}")
    print("-" * 95)
    for prompt in SAMPLE:
        result = await vr.route_request(prompt)
        cand_cost = result.routed.response.cost
        ref_cost = (
            result.verification.reference_response.cost
            if result.verification.reference_response else 0.0
        )
        baseline = ref_cost if ref_cost > 0 else cand_cost
        final = result.final_response.cost
        baseline_total += baseline
        final_total += final
        print(
            f"{result.routed.tier.value:<10} {result.routed.response.model_id:<14} "
            f"{result.verification.result.verdict.value:<7} "
            f"{'Y' if result.escalation.escalated else 'N':<3} "
            f"{result.final_response.model_id:<14} "
            f"{final:>10.6f} {baseline:>10.6f}"
        )

    saved = baseline_total - final_total
    pct = (saved / baseline_total * 100) if baseline_total > 0 else 0.0
    print("-" * 95)
    print(f"baseline (gpt-4o for all): ${baseline_total:.6f}")
    print(f"final (after routing+escalation): ${final_total:.6f}")
    print(f"savings: ${saved:.6f} ({pct:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
