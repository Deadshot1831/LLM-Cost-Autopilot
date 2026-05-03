from pathlib import Path

import pytest

from autopilot.classifier import (
    ComplexityClassifier,
    TrainResult,
    train_classifier,
)
from autopilot.dataset import load_dataset
from autopilot.models import ComplexityTier

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(PROJECT_ROOT / "data" / "prompts_labeled.jsonl")


@pytest.fixture(scope="module")
def train_result(dataset) -> TrainResult:
    return train_classifier(dataset, random_state=42)


@pytest.fixture(scope="module")
def trained_classifier(train_result) -> ComplexityClassifier:
    return train_result.classifier


class TestTrainClassifier:
    def test_returns_train_result(self, train_result: TrainResult):
        assert train_result.classifier is not None
        assert 0.0 <= train_result.test_accuracy <= 1.0

    def test_test_accuracy_above_threshold(self, train_result: TrainResult):
        assert train_result.test_accuracy >= 0.70


class TestClassifierPredict:
    def test_predict_returns_tier(self, trained_classifier: ComplexityClassifier):
        tier = trained_classifier.predict("What is 2 + 2?")
        assert isinstance(tier, ComplexityTier)

    def test_predict_simple_prompt(self, trained_classifier: ComplexityClassifier):
        tier = trained_classifier.predict("Translate 'hello' to French.")
        assert isinstance(tier, ComplexityTier)

    def test_predict_returns_confidence(self, trained_classifier: ComplexityClassifier):
        tier, conf = trained_classifier.predict_with_confidence(
            "Compare the trade-offs between microservices and monoliths for a 10-engineer startup."
        )
        assert isinstance(tier, ComplexityTier)
        assert 0.0 <= conf <= 1.0


class TestPersistence:
    def test_save_and_load_roundtrip(
        self, trained_classifier: ComplexityClassifier, tmp_path: Path
    ):
        path = tmp_path / "clf.joblib"
        trained_classifier.save(path)
        loaded = ComplexityClassifier.load(path)
        original = trained_classifier.predict("Summarize this article.")
        roundtripped = loaded.predict("Summarize this article.")
        assert original == roundtripped
