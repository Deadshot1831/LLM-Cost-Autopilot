# Phase 4: Logging + Cost Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every routed-and-verified request gets a structured row in SQLite (timestamp, prompt hash, tier, model, cost, latency, verdict, escalation, full texts). A Streamlit dashboard reads that database and shows the headline cost-reduction percentage, plus routing distribution, quality, and escalation trend charts.

**Architecture:** A new `db.py` owns the schema (single `requests` table) plus a tiny ORM-free helper (`open_db(path) -> sqlite3.Connection`, `insert_request(conn, RequestRecord)`, `query_*` aggregators). A new `logging_router.py` wraps `VerifyingRouter` and writes one row per call — keeping logging out of `VerifyingRouter` itself so it stays pure. A new `dashboard/app.py` Streamlit page uses `query_*` to render the cost-savings headline + four charts. A new `scripts/load_test.py` runs N diverse prompts through the logging router so the dashboard has data to show.

**Tech Stack:** stdlib `sqlite3` (no ORM — schema is one table), `streamlit` + `pandas` + `altair` for the dashboard, reusing everything from Phase 1-3.

---

## File Structure

```
LLM Cost Autopilot/
├── data/
│   └── autopilot.db                       # NEW: SQLite, gitignored
├── src/autopilot/
│   ├── db.py                              # NEW: schema, RequestRecord, insert/query helpers
│   └── logging_router.py                  # NEW: LoggingRouter wraps VerifyingRouter
├── dashboard/
│   └── app.py                             # NEW: Streamlit page
├── scripts/
│   ├── load_test.py                       # NEW: run N prompts through LoggingRouter
│   └── run_dashboard.sh                   # NEW: thin shell helper to launch streamlit
└── tests/
    ├── test_db.py                         # NEW
    └── test_logging_router.py             # NEW
```

---

### Task 1: Add streamlit + pandas + altair deps

**Files:**
- Modify: `pyproject.toml` (auto-edited by `uv add`)

- [ ] **Step 1: Install deps**

Run:
```bash
cd "/Users/yashwantyadav/Desktop/LLM Cost Autopilot"
uv add streamlit pandas altair
```

- [ ] **Step 2: Verify imports**

Run:
```bash
uv run python -c "import streamlit, pandas, altair; print(streamlit.__version__, pandas.__version__, altair.__version__)"
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add streamlit + pandas + altair for Phase 4 dashboard"
```

---

### Task 2: SQLite schema + helpers (`db.py`)

**Files:**
- Create: `src/autopilot/db.py`
- Test: `tests/test_db.py`
- Modify: `.gitignore` (already excludes `*.db` from Phase 1)

- [ ] **Step 1: Write failing tests**

Create `tests/test_db.py`:
```python
from datetime import datetime, timezone
from pathlib import Path

import pytest

from autopilot.db import (
    RequestRecord,
    insert_request,
    open_db,
    query_aggregate_costs,
    query_recent,
    query_routing_distribution,
    query_verdict_distribution,
)


def _record(
    *,
    model_id: str = "gpt-4o-mini",
    tier: str = "simple",
    cost: float = 0.0001,
    baseline_cost: float = 0.005,
    verdict: str = "pass",
    escalated: bool = False,
) -> RequestRecord:
    return RequestRecord(
        timestamp=datetime.now(timezone.utc),
        prompt_hash="abc123",
        prompt_preview="What is 2 + 2?",
        tier=tier,
        candidate_model=model_id,
        candidate_cost=cost,
        candidate_latency_ms=120.0,
        baseline_model="gpt-4o",
        baseline_cost=baseline_cost,
        verdict=verdict,
        verdict_score=0.95,
        verdict_method="exact_match",
        escalated=escalated,
        final_model=model_id if not escalated else "gpt-4o",
        final_cost=cost if not escalated else baseline_cost,
    )


def test_open_db_creates_schema(tmp_path: Path):
    db = tmp_path / "test.db"
    conn = open_db(db)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='requests'"
    )
    assert cursor.fetchone() is not None


def test_insert_and_query_recent(tmp_path: Path):
    conn = open_db(tmp_path / "test.db")
    insert_request(conn, _record())
    insert_request(conn, _record(model_id="gpt-4o", cost=0.005))
    rows = query_recent(conn, limit=10)
    assert len(rows) == 2


def test_query_aggregate_costs(tmp_path: Path):
    conn = open_db(tmp_path / "test.db")
    # Two cheap requests, one escalated
    insert_request(conn, _record(cost=0.0001, baseline_cost=0.005))
    insert_request(conn, _record(cost=0.0001, baseline_cost=0.005))
    insert_request(conn, _record(cost=0.0001, baseline_cost=0.005, escalated=True))
    agg = query_aggregate_costs(conn)
    assert agg["total_requests"] == 3
    # final cost: 0.0001 + 0.0001 + 0.005 = 0.0052
    assert agg["final_cost_total"] == pytest.approx(0.0052)
    # baseline cost: 0.005 * 3 = 0.015
    assert agg["baseline_cost_total"] == pytest.approx(0.015)
    # savings: 0.015 - 0.0052 = 0.0098
    assert agg["savings_total"] == pytest.approx(0.0098)
    # savings_pct: 0.0098 / 0.015 ~ 65.3%
    assert 60.0 < agg["savings_pct"] < 70.0


def test_query_routing_distribution(tmp_path: Path):
    conn = open_db(tmp_path / "test.db")
    insert_request(conn, _record(model_id="gpt-4o-mini"))
    insert_request(conn, _record(model_id="gpt-4o-mini"))
    insert_request(conn, _record(model_id="gpt-4o"))
    dist = query_routing_distribution(conn)
    by_model = {row["model"]: row["count"] for row in dist}
    assert by_model == {"gpt-4o-mini": 2, "gpt-4o": 1}


def test_query_verdict_distribution(tmp_path: Path):
    conn = open_db(tmp_path / "test.db")
    insert_request(conn, _record(verdict="pass"))
    insert_request(conn, _record(verdict="pass"))
    insert_request(conn, _record(verdict="fail"))
    insert_request(conn, _record(verdict="skip"))
    dist = query_verdict_distribution(conn)
    by_verdict = {row["verdict"]: row["count"] for row in dist}
    assert by_verdict == {"pass": 2, "fail": 1, "skip": 1}


def test_recent_orders_newest_first(tmp_path: Path):
    conn = open_db(tmp_path / "test.db")
    r1 = _record(model_id="m1")
    r2 = _record(model_id="m2")
    insert_request(conn, r1)
    insert_request(conn, r2)
    rows = query_recent(conn, limit=10)
    # newest first - last inserted is m2
    assert rows[0]["candidate_model"] == "m2"
```

- [ ] **Step 2: Run failing tests**

Run:
```bash
uv run pytest tests/test_db.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `db.py`**

Create `src/autopilot/db.py`:
```python
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt_preview TEXT NOT NULL,
    tier TEXT NOT NULL,
    candidate_model TEXT NOT NULL,
    candidate_cost REAL NOT NULL,
    candidate_latency_ms REAL NOT NULL,
    baseline_model TEXT NOT NULL,
    baseline_cost REAL NOT NULL,
    verdict TEXT NOT NULL,
    verdict_score REAL NOT NULL,
    verdict_method TEXT NOT NULL,
    escalated INTEGER NOT NULL,
    final_model TEXT NOT NULL,
    final_cost REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(candidate_model);
"""


@dataclass(frozen=True)
class RequestRecord:
    timestamp: datetime
    prompt_hash: str
    prompt_preview: str
    tier: str
    candidate_model: str
    candidate_cost: float
    candidate_latency_ms: float
    baseline_model: str
    baseline_cost: float
    verdict: str
    verdict_score: float
    verdict_method: str
    escalated: bool
    final_model: str
    final_cost: float


def open_db(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def insert_request(conn: sqlite3.Connection, record: RequestRecord) -> None:
    data = asdict(record)
    data["timestamp"] = record.timestamp.isoformat()
    data["escalated"] = 1 if record.escalated else 0
    conn.execute(
        """
        INSERT INTO requests (
            timestamp, prompt_hash, prompt_preview, tier,
            candidate_model, candidate_cost, candidate_latency_ms,
            baseline_model, baseline_cost,
            verdict, verdict_score, verdict_method,
            escalated, final_model, final_cost
        ) VALUES (
            :timestamp, :prompt_hash, :prompt_preview, :tier,
            :candidate_model, :candidate_cost, :candidate_latency_ms,
            :baseline_model, :baseline_cost,
            :verdict, :verdict_score, :verdict_method,
            :escalated, :final_model, :final_cost
        )
        """,
        data,
    )
    conn.commit()


def query_recent(conn: sqlite3.Connection, *, limit: int = 50) -> list[dict[str, Any]]:
    cursor = conn.execute(
        "SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,)
    )
    return [dict(row) for row in cursor.fetchall()]


def query_aggregate_costs(conn: sqlite3.Connection) -> dict[str, float]:
    cursor = conn.execute(
        """
        SELECT
            COUNT(*) AS total_requests,
            COALESCE(SUM(final_cost), 0.0) AS final_cost_total,
            COALESCE(SUM(baseline_cost), 0.0) AS baseline_cost_total
        FROM requests
        """
    )
    row = cursor.fetchone()
    final_cost = float(row["final_cost_total"])
    baseline = float(row["baseline_cost_total"])
    savings = baseline - final_cost
    pct = (savings / baseline * 100) if baseline > 0 else 0.0
    return {
        "total_requests": int(row["total_requests"]),
        "final_cost_total": final_cost,
        "baseline_cost_total": baseline,
        "savings_total": savings,
        "savings_pct": pct,
    }


def query_routing_distribution(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cursor = conn.execute(
        "SELECT candidate_model AS model, COUNT(*) AS count FROM requests GROUP BY candidate_model"
    )
    return [dict(row) for row in cursor.fetchall()]


def query_verdict_distribution(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cursor = conn.execute(
        "SELECT verdict, COUNT(*) AS count FROM requests GROUP BY verdict"
    )
    return [dict(row) for row in cursor.fetchall()]


def query_escalation_rate_over_time(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Group by date (YYYY-MM-DD) and compute escalation rate."""
    cursor = conn.execute(
        """
        SELECT
            substr(timestamp, 1, 10) AS day,
            AVG(escalated) AS escalation_rate,
            COUNT(*) AS n
        FROM requests
        GROUP BY day
        ORDER BY day
        """
    )
    return [dict(row) for row in cursor.fetchall()]
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
uv run pytest tests/test_db.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/db.py tests/test_db.py
git commit -m "feat(db): SQLite schema + insert/query helpers for request log"
```

---

### Task 3: `LoggingRouter` wraps `VerifyingRouter`

**Files:**
- Create: `src/autopilot/logging_router.py`
- Test: `tests/test_logging_router.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_logging_router.py`:
```python
from pathlib import Path

import pytest

from autopilot.classifier import train_classifier
from autopilot.dataset import load_dataset
from autopilot.db import open_db, query_aggregate_costs, query_recent
from autopilot.logging_router import LoggingRouter
from autopilot.models import Response
from autopilot.registry import load_registry
from autopilot.router import Router
from autopilot.routing import load_routing_config
from autopilot.verifier import Verifier
from autopilot.verifying_router import VerifyingRouter

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def trained():
    rows = load_dataset(PROJECT_ROOT / "data" / "prompts_labeled.jsonl")
    return train_classifier(rows, random_state=42).classifier


@pytest.fixture(scope="module")
def registry():
    return load_registry(PROJECT_ROOT / "config" / "models.yaml")


@pytest.fixture
def routing_to_mocks(tmp_path):
    cfg = tmp_path / "routing.yaml"
    cfg.write_text(
        "routing:\n"
        "  simple: llama3.2:3b\n"
        "  moderate: claude-haiku-4-5\n"
        "  complex: claude-sonnet-4-6\n"
    )
    return load_routing_config(cfg)


def _make_logging_router(trained, registry, routing_to_mocks, db_path, log_path):
    reference_cfg = registry.get("gpt-4o")

    async def fake_send(prompt, config, *, provider=None):
        # Make the reference cost notably higher than candidate so savings show
        return Response(
            text="reference content here",
            input_tokens=10, output_tokens=10, latency_ms=200.0,
            cost=0.005, model_id=config.model_id,
        )

    base = Router(classifier=trained, routing=routing_to_mocks, registry=registry)
    verifier = Verifier(reference_cfg=reference_cfg, send=fake_send)
    vr = VerifyingRouter(
        base_router=base, verifier=verifier, failure_log_path=log_path,
    )
    conn = open_db(db_path)
    return LoggingRouter(verifying_router=vr, conn=conn), conn


async def test_logging_router_inserts_row_per_request(
    trained, registry, routing_to_mocks, tmp_path
):
    lr, conn = _make_logging_router(
        trained, registry, routing_to_mocks,
        db_path=tmp_path / "test.db", log_path=tmp_path / "fail.jsonl",
    )
    await lr.route_request("hello world")
    await lr.route_request("Translate hi to French.")
    rows = query_recent(conn, limit=10)
    assert len(rows) == 2
    assert all(r["timestamp"] for r in rows)
    assert all(r["prompt_hash"] for r in rows)


async def test_logging_router_returns_verified_response(
    trained, registry, routing_to_mocks, tmp_path
):
    lr, _ = _make_logging_router(
        trained, registry, routing_to_mocks,
        db_path=tmp_path / "test.db", log_path=tmp_path / "fail.jsonl",
    )
    result = await lr.route_request("hello")
    assert result.final_response is not None
    assert result.final_response.text


async def test_logging_router_records_baseline_for_savings(
    trained, registry, routing_to_mocks, tmp_path
):
    lr, conn = _make_logging_router(
        trained, registry, routing_to_mocks,
        db_path=tmp_path / "test.db", log_path=tmp_path / "fail.jsonl",
    )
    await lr.route_request("hello world")
    agg = query_aggregate_costs(conn)
    assert agg["total_requests"] == 1
    assert agg["baseline_cost_total"] > 0  # reference cost was recorded


async def test_prompt_preview_is_truncated(
    trained, registry, routing_to_mocks, tmp_path
):
    lr, conn = _make_logging_router(
        trained, registry, routing_to_mocks,
        db_path=tmp_path / "test.db", log_path=tmp_path / "fail.jsonl",
    )
    long_prompt = "a " * 200
    await lr.route_request(long_prompt)
    rows = query_recent(conn, limit=1)
    assert len(rows[0]["prompt_preview"]) <= 200
```

- [ ] **Step 2: Run failing tests**

Run:
```bash
uv run pytest tests/test_logging_router.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `logging_router.py`**

Create `src/autopilot/logging_router.py`:
```python
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone

from autopilot.db import RequestRecord, insert_request
from autopilot.verifying_router import VerifiedRoutedResponse, VerifyingRouter

PROMPT_PREVIEW_MAX = 200


class LoggingRouter:
    def __init__(
        self,
        *,
        verifying_router: VerifyingRouter,
        conn: sqlite3.Connection,
    ) -> None:
        self._vr = verifying_router
        self._conn = conn

    async def route_request(self, prompt: str) -> VerifiedRoutedResponse:
        result = await self._vr.route_request(prompt)
        self._record(prompt, result)
        return result

    def _record(self, prompt: str, result: VerifiedRoutedResponse) -> None:
        cand = result.routed.response
        ref = result.verification.reference_response
        baseline_model = ref.model_id if ref is not None else cand.model_id
        baseline_cost = ref.cost if ref is not None else cand.cost
        record = RequestRecord(
            timestamp=datetime.now(timezone.utc),
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            prompt_preview=prompt[:PROMPT_PREVIEW_MAX],
            tier=result.routed.tier.value,
            candidate_model=cand.model_id,
            candidate_cost=cand.cost,
            candidate_latency_ms=cand.latency_ms,
            baseline_model=baseline_model,
            baseline_cost=baseline_cost,
            verdict=result.verification.result.verdict.value,
            verdict_score=result.verification.result.score,
            verdict_method=result.verification.result.method,
            escalated=result.escalation.escalated,
            final_model=result.final_response.model_id,
            final_cost=result.final_response.cost,
        )
        insert_request(self._conn, record)
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
uv run pytest tests/test_logging_router.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/logging_router.py tests/test_logging_router.py
git commit -m "feat(logging-router): persist every routed+verified request to SQLite"
```

---

### Task 4: Streamlit dashboard (`dashboard/app.py`)

**Files:**
- Create: `dashboard/app.py`
- Create: `scripts/run_dashboard.sh`

- [ ] **Step 1: Write `dashboard/app.py`**

Create `dashboard/app.py`:
```python
"""Streamlit dashboard for LLM Cost Autopilot.

Run:
    uv run streamlit run dashboard/app.py
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from autopilot.db import (
    open_db,
    query_aggregate_costs,
    query_escalation_rate_over_time,
    query_recent,
    query_routing_distribution,
    query_verdict_distribution,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "autopilot.db"

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide")
st.title("LLM Cost Autopilot")

db_path = st.sidebar.text_input("Database path", value=str(DEFAULT_DB))
db_file = Path(db_path)

if not db_file.exists():
    st.warning(
        f"No database found at `{db_path}`.\n\n"
        "Run `uv run python scripts/load_test.py` to populate it, "
        "then refresh this page."
    )
    st.stop()

conn = open_db(db_file)
agg = query_aggregate_costs(conn)

# ---- Headline metric ----
st.header("Cost Reduction")
left, mid, right = st.columns(3)
left.metric("Total requests", f"{agg['total_requests']:,}")
mid.metric(
    "Final cost",
    f"${agg['final_cost_total']:.4f}",
    delta=f"-${agg['savings_total']:.4f} vs baseline",
    delta_color="inverse",
)
right.metric(
    "Savings",
    f"{agg['savings_pct']:.1f}%",
    help="Compared to running every request through gpt-4o.",
)

st.caption(
    f"Baseline (every request via gpt-4o): ${agg['baseline_cost_total']:.4f}"
)

# ---- Routing distribution ----
st.header("Routing distribution")
routing = pd.DataFrame(query_routing_distribution(conn))
if not routing.empty:
    chart = (
        alt.Chart(routing)
        .mark_arc(innerRadius=60)
        .encode(theta="count:Q", color="model:N", tooltip=["model", "count"])
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No routing data yet.")

# ---- Verdict distribution ----
st.header("Quality verdicts")
verdicts = pd.DataFrame(query_verdict_distribution(conn))
if not verdicts.empty:
    chart = (
        alt.Chart(verdicts)
        .mark_bar()
        .encode(x="verdict:N", y="count:Q", color="verdict:N",
                tooltip=["verdict", "count"])
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No verdicts yet.")

# ---- Escalation rate over time ----
st.header("Escalation rate over time")
esc = pd.DataFrame(query_escalation_rate_over_time(conn))
if not esc.empty:
    chart = (
        alt.Chart(esc)
        .mark_line(point=True)
        .encode(
            x="day:T",
            y=alt.Y("escalation_rate:Q", title="escalation rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=["day", "escalation_rate", "n"],
        )
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No escalation data yet.")

# ---- Recent requests ----
st.header("Recent requests")
recent = pd.DataFrame(query_recent(conn, limit=50))
if not recent.empty:
    display = recent[
        [
            "timestamp", "tier", "candidate_model", "verdict",
            "escalated", "final_model", "final_cost", "baseline_cost",
            "prompt_preview",
        ]
    ]
    st.dataframe(display, use_container_width=True)
else:
    st.info("No requests yet.")
```

- [ ] **Step 2: Write the launcher**

Create `scripts/run_dashboard.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run streamlit run dashboard/app.py "$@"
```

Then make it executable:
```bash
chmod +x scripts/run_dashboard.sh
```

- [ ] **Step 3: Smoke-test (just import the module - don't actually run streamlit)**

Run:
```bash
uv run python -c "import dashboard.app" 2>&1 | head -5
```
Expected: no traceback. (The Streamlit set_page_config call will warn about being outside a Streamlit runtime, but should not error.)

If `dashboard.app` is not importable as a module because there's no `__init__.py`, that's fine — Streamlit imports it via path. To smoke-test syntax:
```bash
uv run python -m py_compile dashboard/app.py
```

- [ ] **Step 4: Commit**

```bash
git add dashboard/app.py scripts/run_dashboard.sh
git commit -m "feat(dashboard): Streamlit page with cost-savings headline + 4 charts"
```

---

### Task 5: Load-test script

**Files:**
- Create: `scripts/load_test.py`

- [ ] **Step 1: Write `scripts/load_test.py`**

Create `scripts/load_test.py`:
```python
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
    print(f"Done. Open the dashboard: ./scripts/run_dashboard.sh")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Smoke-test (will short-circuit without key)**

Run:
```bash
uv run python scripts/load_test.py -n 1
```
Expected: prints the no-key message OR runs one request and prints the row.

- [ ] **Step 3: Commit**

```bash
git add scripts/load_test.py
git commit -m "feat(scripts): load test that populates the dashboard database"
```

---

### Task 6: Phase 4 wrap-up

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full unit suite**

Run:
```bash
uv run pytest --ignore=tests/integration
```
Expected: all green.

- [ ] **Step 2: Update README**

Read `README.md`, then:
- check off Phase 4 in Status
- add a "Logging + dashboard (Phase 4)" subsection under Usage:
  ```python
  from autopilot.db import open_db
  from autopilot.logging_router import LoggingRouter

  conn = open_db("data/autopilot.db")
  lr = LoggingRouter(verifying_router=vr, conn=conn)
  await lr.route_request("Summarize this article.")
  # then: ./scripts/run_dashboard.sh
  ```
- add `scripts/load_test.py` and `scripts/run_dashboard.sh` to the Tests + Scripts section
- under Architecture, add a Phase 4 block listing `db.py`, `logging_router.py`, `dashboard/app.py`

- [ ] **Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: phase 4 complete - SQLite logging + Streamlit dashboard"
```

---

## Self-Review

**Spec coverage (Phase 4):**
- "Log everything: timestamp, prompt hash, complexity tier, routed model, cost, latency, quality score, escalation flag" → Tasks 2, 3 (`RequestRecord`, `insert_request`)
- "Build the cost dashboard: total cost / baseline / routing distribution / quality / escalation rate" → Task 4
- "Money shot metric: cost reduction percentage prominently displayed" → Task 4 (headline `st.metric` with savings_pct)
- "Audit trail per request" → recent-requests table at the bottom of the dashboard

**Out of scope for Phase 4 (deferred):**
- True async fire-and-forget logging via background queue. Current `LoggingRouter` writes synchronously after `route_request` resolves; latency impact is negligible (one local SQLite insert) and tests stay deterministic.
- Migrations / multi-version schema. SQLite `IF NOT EXISTS` is enough for V1; Phase 5 may revisit if we add columns.
- Daily/weekly cost aggregation rolled up — current dashboard does it via SQL on demand.

**Placeholder scan:** None.

**Type consistency:** `RequestRecord`, `open_db`, `insert_request`, `query_recent`, `query_aggregate_costs`, `query_routing_distribution`, `query_verdict_distribution`, `query_escalation_rate_over_time`, `LoggingRouter`, `VerifyingRouter`, `VerifiedRoutedResponse` — all match across tasks 2-5.
