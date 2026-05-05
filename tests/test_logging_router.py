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
    assert agg["baseline_cost_total"] > 0


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
