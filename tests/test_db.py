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
    insert_request(conn, _record(cost=0.0001, baseline_cost=0.005))
    insert_request(conn, _record(cost=0.0001, baseline_cost=0.005))
    insert_request(conn, _record(cost=0.0001, baseline_cost=0.005, escalated=True))
    agg = query_aggregate_costs(conn)
    assert agg["total_requests"] == 3
    # final cost: 0.0001 + 0.0001 + 0.005 = 0.0052
    assert agg["final_cost_total"] == pytest.approx(0.0052)
    # baseline cost: 0.005 * 3 = 0.015
    assert agg["baseline_cost_total"] == pytest.approx(0.015)
    assert agg["savings_total"] == pytest.approx(0.0098)
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
    assert rows[0]["candidate_model"] == "m2"
