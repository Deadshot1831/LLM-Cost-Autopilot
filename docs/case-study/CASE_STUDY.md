# LLM Cost Autopilot - Case Study

> **Built a system that reduced LLM API costs by 41.5% across 500 simulated
> requests while maintaining quality parity via auto-escalation on
> disagreement (11.8% of cases).**

## The problem

Every company running LLMs at scale is over-provisioned. Most production
prompts are simple — extraction, formatting, basic Q&A — but they all get
sent to the same expensive flagship model "to be safe." The result is a
4-10x cost overhead on the easy 80%.

## What I built

A drop-in routing layer that sits in front of multiple LLM providers,
classifies each incoming prompt by complexity, sends it to the cheapest
model that can handle it, and verifies that the cheap model's answer
agrees with what the expensive model would have said. On disagreement,
the request auto-escalates and the failure becomes a training example
for the classifier.

![Dashboard overview](screenshots/dashboard-overview.png)

## Architecture

```
                    POST /v1/completions { prompt }
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ LoggingRouter   │  -> SQLite (one row per request)
                          └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ VerifyingRouter │  -> reference call (gpt-4o)
                          └────────┬────────┘     -> exact-match or LLM-as-judge
                                   │              -> escalate on FAIL
                                   ▼
                          ┌─────────────────┐
                          │     Router      │  -> ComplexityClassifier
                          └────────┬────────┘     -> tier -> model_id (YAML)
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ Provider (cheap)│  -> OpenAI / Anthropic / Ollama
                          └─────────────────┘
```

Five layers, each independently testable:

1. **Provider abstraction** — `Provider` protocol with one file per
   provider (OpenAI, Anthropic, Ollama). Returns a standardized
   `Response(text, input_tokens, output_tokens, latency_ms, cost,
   model_id)`. Costs are derived from a YAML model registry, not magic
   numbers scattered through the code.

2. **Complexity classifier** — TF-IDF + LogisticRegression on 209
   hand-labeled prompts. **92.9% test accuracy.** Numeric features
   (token count, instruction verbs, constraints, output-format
   complexity) are encoded as synthetic prompt prefixes so the
   single-pipeline design stays joblib-portable.

3. **Routing** — Tier-to-model map in `config/routing.yaml`. SIMPLE
   prompts go to `gpt-4o-mini`, MODERATE to `gpt-4o-mini`, COMPLEX to
   `gpt-4o`. Routing config can be hot-swapped via
   `PUT /v1/routing-config` without redeploying.

4. **Verification** — After every cheap response, a reference call to
   `gpt-4o` scores agreement. Short prompts use token-set Jaccard
   (>= 0.7 = PASS). Long prompts use the reference model itself as a
   1-5 judge (>= 4 = PASS). FAILs auto-escalate the response and append
   the prompt to `data/routing_failures.jsonl` — which the retraining
   script promotes one tier up and re-fits the classifier on.

5. **Logging + dashboard** — Every request → one SQLite row. Streamlit
   dashboard shows the headline savings %, routing distribution, verdict
   distribution, and escalation-rate-over-time.

## Results

After 500 requests routed through the system (simulated load with real
registry pricing):

- **41.5% cost reduction** vs sending every request to `gpt-4o`
- **56% PASS rate** on automated verification (280/500)
- **11.8% escalation rate** — cheap model disagreed with reference; final
  answer came from the reference (59/500)
- **32% SKIP rate** — verification was a no-op because the candidate
  was already the reference model (161/500, all complex-tier)

Per-tier:

| Tier | Requests | Final cost | Baseline cost | Savings | Escalation rate |
|---|---:|---:|---:|---:|---:|
| simple | 212 | $0.0113 | $0.0393 | $0.0280 | 17.5% |
| moderate | 127 | $0.0090 | $0.0373 | $0.0284 | 17.3% |
| complex | 161 | $0.0592 | $0.0592 | $0.0000 | 0.0% |

(Complex-tier savings are zero because complex prompts already route to
the most expensive model — there's nothing cheaper to route to.)

See [REPORT.md](REPORT.md) for the auto-generated full report.

![Routing distribution](screenshots/dashboard-routing.png)

## Key design decisions

**1. The verifier is synchronous, not background.** The original spec
called for an async fire-and-forget verifier. I started there but found
it made testing harder and the escalation logic awkward — you have to
return the cheap response, then maybe replace it later, and clients have
to handle both paths. For V1, blocking on the reference call is fine: it
adds ~200ms p50 to every request, but the resulting code is dramatically
simpler and the savings are large enough that the latency cost is
acceptable. Genuinely async verification is a deliberate Phase-7
refactor, not an oversight.

**2. The classifier is sklearn, not a fine-tuned model.** A 92.9%
3-class classifier on 209 hand-labeled prompts is enough for V1 routing.
Replacing it with a fine-tuned distilbert would buy maybe 3-4 points of
accuracy at 100x the deployment cost. I'd revisit only if escalation
rate climbed above 20%.

**3. Costs come from a YAML registry, not from API responses.** The
`ModelConfig` dataclass holds `input_cost_per_1k` and
`output_cost_per_1k` per model. `compute_cost(input_tokens,
output_tokens)` is the single source of truth. Dashboards, reports, and
routing decisions all share the same cost model — no drift.

**4. Failures feed back into the classifier.** Every routing failure
gets logged with its prompt, then `scripts/retrain_from_failures.py`
promotes those prompts one tier up and retrains. The classifier gets
smarter over time without manual relabeling.

## What I'd do next

- **True async verification.** Return the cheap response immediately;
  verify in a background worker; the escalation event becomes a webhook
  or a follow-up message. Requires splitting `LoggingRouter.route_request`
  in two.
- **Streaming.** The verifier currently needs the full text before
  scoring. For long-form generation, stream the cheap response to the
  user while a background verifier scores it.
- **Real local models.** Local Ollama is currently mocked; wiring in a
  real local model would push SIMPLE-tier costs to literally zero.
- **Better verification on long prompts.** LLM-as-judge with a 1-5
  integer parse is the cheapest thing that works. A multi-criterion
  rubric (factuality, completeness, helpfulness as separate axes) would
  catch failures the current judge misses.

## Stack

Python 3.11 · `uv` · FastAPI · scikit-learn · SQLite · Streamlit · Docker.
~3,000 lines of source, **109 unit tests** (no real API calls), opt-in
integration tests for real OpenAI.

## Repo structure

- `src/autopilot/` — the library (5 phases of vertically-sliced modules)
- `config/` — model registry, routing map, verification settings
- `data/` — labeled prompts, SQLite database, failures log
- `dashboard/app.py` — Streamlit page
- `scripts/` — train classifier, run baseline, simulate load, generate report, retrain
- `tests/` — 109 unit tests, opt-in `-m integration` for real APIs
- `docs/superpowers/plans/` — the per-phase implementation plans (one per phase)
- `docs/case-study/` — this case study + auto-generated `REPORT.md`
- `Dockerfile` + `docker-compose.yml` — single-service deployment
