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
        "SELECT candidate_model AS model, COUNT(*) AS count "
        "FROM requests GROUP BY candidate_model"
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
