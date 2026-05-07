"""Generate docs/case-study/REPORT.md from data/autopilot.db.

Auto-generated portion of the case study - headline numbers, breakdown
tables, distributions. The hand-written narrative lives in CASE_STUDY.md.

Usage:
    uv run python scripts/generate_report.py
"""
from __future__ import annotations

from pathlib import Path

from autopilot.db import (
    open_db,
    query_aggregate_costs,
    query_routing_distribution,
    query_verdict_distribution,
)

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "autopilot.db"
OUT = ROOT / "docs" / "case-study" / "REPORT.md"


def main() -> None:
    if not DB.exists():
        raise SystemExit(f"No database at {DB}. Run scripts/simulate_load.py first.")

    conn = open_db(DB)
    agg = query_aggregate_costs(conn)
    routing = query_routing_distribution(conn)
    verdicts = query_verdict_distribution(conn)

    per_tier = conn.execute(
        """
        SELECT tier,
               COUNT(*) AS n,
               SUM(final_cost) AS final_cost,
               SUM(baseline_cost) AS baseline_cost,
               AVG(escalated) AS escalation_rate
        FROM requests
        GROUP BY tier
        ORDER BY tier
        """
    ).fetchall()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# LLM Cost Autopilot - Cost Savings Report")
    lines.append("")
    lines.append("_Auto-generated from `data/autopilot.db`. Do not edit by hand._")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **Total requests:** {agg['total_requests']:,}")
    lines.append(f"- **Final cost (routed + escalated):** ${agg['final_cost_total']:.4f}")
    lines.append(f"- **Baseline cost (every request via gpt-4o):** ${agg['baseline_cost_total']:.4f}")
    lines.append(f"- **Savings:** ${agg['savings_total']:.4f}")
    lines.append(f"- **Cost reduction:** **{agg['savings_pct']:.1f}%**")
    lines.append("")
    lines.append("## Per-tier breakdown")
    lines.append("")
    lines.append("| Tier | Requests | Final cost | Baseline cost | Savings | Escalation rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in per_tier:
        savings = float(row["baseline_cost"]) - float(row["final_cost"])
        lines.append(
            f"| {row['tier']} | {row['n']} | ${float(row['final_cost']):.4f} | "
            f"${float(row['baseline_cost']):.4f} | ${savings:.4f} | "
            f"{float(row['escalation_rate']) * 100:.1f}% |"
        )
    lines.append("")
    lines.append("## Routing distribution")
    lines.append("")
    lines.append("| Model | Requests |")
    lines.append("|---|---:|")
    for row in routing:
        lines.append(f"| `{row['model']}` | {row['count']} |")
    lines.append("")
    lines.append("## Quality verdicts")
    lines.append("")
    lines.append("| Verdict | Count |")
    lines.append("|---|---:|")
    for row in verdicts:
        lines.append(f"| {row['verdict']} | {row['count']} |")
    lines.append("")
    lines.append("## How costs are computed")
    lines.append("")
    lines.append(
        "Cost per request comes from `ModelConfig.compute_cost(input_tokens, "
        "output_tokens)` using prices from `config/models.yaml` (May 2026 "
        "pricing). The baseline cost is what the same request would have "
        "cost if it had been sent to `gpt-4o` (the highest-tier model in the "
        "registry). The savings figure is `baseline_cost - final_cost`, where "
        "`final_cost` is the candidate model's cost on PASS/SKIP and the "
        "reference model's cost on FAIL (auto-escalation)."
    )
    lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
