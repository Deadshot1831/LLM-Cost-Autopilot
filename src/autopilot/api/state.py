from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import yaml

from autopilot.classifier import ComplexityClassifier
from autopilot.db import open_db
from autopilot.logging_router import LoggingRouter
from autopilot.models import ComplexityTier
from autopilot.registry import ModelRegistry, load_registry
from autopilot.router import Router
from autopilot.routing import load_routing_config
from autopilot.verifier import Verifier
from autopilot.verifying_router import VerifyingRouter

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_REFERENCE_MODEL = "gpt-4o"


def _read_reference_model(verification_yaml: Path | str) -> str:
    """Read reference_model from config/verification.yaml.

    Falls back to DEFAULT_REFERENCE_MODEL if the file or key is absent.
    """
    p = Path(verification_yaml)
    if not p.is_file():
        return DEFAULT_REFERENCE_MODEL
    raw = yaml.safe_load(p.read_text()) or {}
    return str(raw.get("reference_model") or DEFAULT_REFERENCE_MODEL)


@dataclass
class AppState:
    registry: ModelRegistry
    routing: dict[ComplexityTier, str]
    routing_path: Path
    classifier: ComplexityClassifier
    base_router: Router
    verifier: Verifier
    verifying_router: VerifyingRouter
    db_conn: sqlite3.Connection
    logging_router: LoggingRouter

    @classmethod
    def from_paths(
        cls,
        *,
        models_yaml: Path | str = ROOT / "config" / "models.yaml",
        routing_yaml: Path | str = ROOT / "config" / "routing.yaml",
        verification_yaml: Path | str = ROOT / "config" / "verification.yaml",
        classifier_path: Path | str = ROOT / "models" / "classifier.joblib",
        db_path: Path | str = ROOT / "data" / "autopilot.db",
        failure_log_path: Path | str = ROOT / "data" / "routing_failures.jsonl",
        reference_model_id: str | None = None,
    ) -> "AppState":
        registry = load_registry(models_yaml)
        routing = load_routing_config(routing_yaml)
        classifier = ComplexityClassifier.load(classifier_path)
        # Reference model resolution order:
        #   1. explicit kwarg
        #   2. config/verification.yaml `reference_model`
        #   3. DEFAULT_REFERENCE_MODEL
        ref_id = reference_model_id or _read_reference_model(verification_yaml)
        base_router = Router(
            classifier=classifier, routing=routing, registry=registry,
        )
        verifier = Verifier(reference_cfg=registry.get(ref_id))
        verifying_router = VerifyingRouter(
            base_router=base_router, verifier=verifier,
            failure_log_path=failure_log_path,
        )
        conn = open_db(db_path)
        logging_router = LoggingRouter(
            verifying_router=verifying_router, conn=conn,
        )
        return cls(
            registry=registry,
            routing=routing,
            routing_path=Path(routing_yaml),
            classifier=classifier,
            base_router=base_router,
            verifier=verifier,
            verifying_router=verifying_router,
            db_conn=conn,
            logging_router=logging_router,
        )

    def update_routing(self, new_routing: dict[ComplexityTier, str]) -> None:
        """Mutate in-memory routing AND persist back to YAML."""
        for model_id in new_routing.values():
            self.registry.get(model_id)  # raises ModelNotFoundError if unknown
        self.routing.clear()
        self.routing.update(new_routing)
        self.base_router = Router(
            classifier=self.classifier,
            routing=self.routing,
            registry=self.registry,
        )
        self.verifying_router = VerifyingRouter(
            base_router=self.base_router,
            verifier=self.verifier,
            failure_log_path=getattr(self.verifying_router, "_failure_log_path", None),
        )
        self.logging_router = LoggingRouter(
            verifying_router=self.verifying_router, conn=self.db_conn,
        )
        self.routing_path.write_text(
            "routing:\n"
            f"  simple: {self.routing[ComplexityTier.SIMPLE]}\n"
            f"  moderate: {self.routing[ComplexityTier.MODERATE]}\n"
            f"  complex: {self.routing[ComplexityTier.COMPLEX]}\n"
        )
