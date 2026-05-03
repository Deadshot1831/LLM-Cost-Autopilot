"""Train the complexity classifier and save it to models/classifier.joblib."""
from __future__ import annotations

from pathlib import Path

from autopilot.classifier import train_classifier
from autopilot.dataset import load_dataset

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "prompts_labeled.jsonl"
MODEL_PATH = ROOT / "models" / "classifier.joblib"


def main() -> None:
    rows = load_dataset(DATASET_PATH)
    print(f"Loaded {len(rows)} labeled prompts")
    result = train_classifier(rows, random_state=42)
    print(f"Trained on {result.n_train}, tested on {result.n_test}")
    print(f"Test accuracy: {result.test_accuracy:.3f}")
    result.classifier.save(MODEL_PATH)
    print(f"Saved classifier to {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
