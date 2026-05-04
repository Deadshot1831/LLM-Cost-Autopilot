"""Append routing failures to the labeled dataset (promoted one tier UP) and retrain."""
from __future__ import annotations

import json
from pathlib import Path

from autopilot.classifier import ComplexityClassifier, train_classifier
from autopilot.dataset import load_dataset
from autopilot.models import ComplexityTier

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "prompts_labeled.jsonl"
FAILURES = ROOT / "data" / "routing_failures.jsonl"
MODEL = ROOT / "models" / "classifier.joblib"

NEXT_TIER = {
    ComplexityTier.SIMPLE: ComplexityTier.MODERATE,
    ComplexityTier.MODERATE: ComplexityTier.COMPLEX,
    ComplexityTier.COMPLEX: ComplexityTier.COMPLEX,
}


def main() -> None:
    if not FAILURES.exists() or FAILURES.stat().st_size == 0:
        print(f"No failures file at {FAILURES}; nothing to do.")
        return

    classifier = ComplexityClassifier.load(MODEL)
    new_rows = 0
    with FAILURES.open() as f, DATASET.open("a") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = rec["prompt"]
            current_tier = classifier.predict(prompt)
            promoted = NEXT_TIER[current_tier]
            out.write(json.dumps({"prompt": prompt, "tier": promoted.value}) + "\n")
            new_rows += 1
    print(f"Appended {new_rows} promoted examples to {DATASET.name}")

    rows = load_dataset(DATASET)
    print(f"Total dataset size: {len(rows)}")
    result = train_classifier(rows, random_state=42)
    print(f"New test accuracy: {result.test_accuracy:.3f}")
    result.classifier.save(MODEL)
    print(f"Re-saved classifier to {MODEL.name}")

    FAILURES.write_text("")
    print(f"Cleared {FAILURES.name}")


if __name__ == "__main__":
    main()
