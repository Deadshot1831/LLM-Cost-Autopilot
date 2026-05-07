# LLM Cost Autopilot - Cost Savings Report

_Auto-generated from `data/autopilot.db`. Do not edit by hand._

## Headline

- **Total requests:** 500
- **Final cost (routed + escalated):** $0.0795
- **Baseline cost (every request via gpt-4o):** $0.1358
- **Savings:** $0.0564
- **Cost reduction:** **41.5%**

## Per-tier breakdown

| Tier | Requests | Final cost | Baseline cost | Savings | Escalation rate |
|---|---:|---:|---:|---:|---:|
| complex | 161 | $0.0592 | $0.0592 | $0.0000 | 0.0% |
| moderate | 127 | $0.0090 | $0.0373 | $0.0284 | 17.3% |
| simple | 212 | $0.0113 | $0.0393 | $0.0280 | 17.5% |

## Routing distribution

| Model | Requests |
|---|---:|
| `gpt-4o` | 161 |
| `gpt-4o-mini` | 339 |

## Quality verdicts

| Verdict | Count |
|---|---:|
| fail | 59 |
| pass | 280 |
| skip | 161 |

## How costs are computed

Cost per request comes from `ModelConfig.compute_cost(input_tokens, output_tokens)` using prices from `config/models.yaml` (May 2026 pricing). The baseline cost is what the same request would have cost if it had been sent to `gpt-4o` (the highest-tier model in the registry). The savings figure is `baseline_cost - final_cost`, where `final_cost` is the candidate model's cost on PASS/SKIP and the reference model's cost on FAIL (auto-escalation).
