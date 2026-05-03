from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from autopilot.dataset import LabeledPrompt
from autopilot.features import extract_features
from autopilot.models import ComplexityTier


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


@dataclass
class TrainResult:
    classifier: "ComplexityClassifier"
    test_accuracy: float
    n_train: int
    n_test: int


class ComplexityClassifier:
    """TF-IDF + LogisticRegression. Numeric features are concatenated by
    appending a synthetic prefix to the text so TF-IDF picks them up too -
    keeps the pipeline single-stage for joblib compatibility."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    @staticmethod
    def _augment(prompt: str) -> str:
        f = extract_features(prompt)
        markers = [
            f"__tok{min(f.token_count // 10, 50)}",
            f"__verbs{min(f.instruction_verb_count, 5)}",
            f"__cons{min(f.constraint_count, 5)}",
            f"__ctx{1 if f.has_context else 0}",
            f"__fmt{min(f.output_format_complexity, 3)}",
        ]
        return " ".join(markers) + " " + prompt

    def predict(self, prompt: str) -> ComplexityTier:
        label = self._pipeline.predict([self._augment(prompt)])[0]
        return ComplexityTier(label)

    def predict_with_confidence(self, prompt: str) -> tuple[ComplexityTier, float]:
        proba = self._pipeline.predict_proba([self._augment(prompt)])[0]
        idx = int(np.argmax(proba))
        label = self._pipeline.classes_[idx]
        return ComplexityTier(label), float(proba[idx])

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)

    @classmethod
    def load(cls, path: Path | str) -> "ComplexityClassifier":
        pipeline = joblib.load(Path(path))
        return cls(pipeline)


def train_classifier(
    rows: Iterable[LabeledPrompt],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    rows = list(rows)
    if len(rows) < 30:
        raise ValueError(f"Need at least 30 rows, got {len(rows)}")

    texts = [ComplexityClassifier._augment(r.prompt) for r in rows]
    labels = [r.tier.value for r in rows]

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    pipeline = _build_pipeline()
    pipeline.fit(x_train, y_train)

    accuracy = float(pipeline.score(x_test, y_test))
    return TrainResult(
        classifier=ComplexityClassifier(pipeline),
        test_accuracy=accuracy,
        n_train=len(x_train),
        n_test=len(x_test),
    )
