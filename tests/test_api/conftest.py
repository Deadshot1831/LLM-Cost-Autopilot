from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autopilot.api.app import create_app
from autopilot.api.state import AppState
from autopilot.classifier import train_classifier
from autopilot.db import open_db
from autopilot.dataset import load_dataset
from autopilot.logging_router import LoggingRouter
from autopilot.models import Response
from autopilot.registry import load_registry
from autopilot.router import Router
from autopilot.routing import load_routing_config
from autopilot.verifier import Verifier
from autopilot.verifying_router import VerifyingRouter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="session")
def trained():
    rows = load_dataset(PROJECT_ROOT / "data" / "prompts_labeled.jsonl")
    return train_classifier(rows, random_state=42).classifier


@pytest.fixture
def app_state(trained, tmp_path):
    routing_yaml = tmp_path / "routing.yaml"
    routing_yaml.write_text(
        "routing:\n"
        "  simple: llama3.2:3b\n"
        "  moderate: claude-haiku-4-5\n"
        "  complex: claude-sonnet-4-6\n"
    )
    registry = load_registry(PROJECT_ROOT / "config" / "models.yaml")
    routing = load_routing_config(routing_yaml)
    base = Router(classifier=trained, routing=routing, registry=registry)

    async def fake_send(prompt, config, *, provider=None):
        return Response(
            text="reference content here", input_tokens=10, output_tokens=10,
            latency_ms=200.0, cost=0.005, model_id=config.model_id,
        )

    verifier = Verifier(reference_cfg=registry.get("gpt-4o"), send=fake_send)
    vr = VerifyingRouter(
        base_router=base, verifier=verifier,
        failure_log_path=tmp_path / "fail.jsonl",
    )
    conn = open_db(tmp_path / "test.db")
    lr = LoggingRouter(verifying_router=vr, conn=conn)
    state = AppState(
        registry=registry, routing=routing, routing_path=routing_yaml,
        classifier=trained, base_router=base, verifier=verifier,
        verifying_router=vr, db_conn=conn, logging_router=lr,
    )
    return state


@pytest.fixture
def client(app_state):
    app = create_app(app_state)
    return TestClient(app)
