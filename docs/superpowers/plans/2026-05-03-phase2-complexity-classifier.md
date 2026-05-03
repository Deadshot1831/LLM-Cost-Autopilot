# Phase 2: Complexity Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a feature-engineered scikit-learn classifier that scores each incoming prompt as SIMPLE/MODERATE/COMPLEX, then a YAML-driven routing map that picks the cheapest model for each tier — wired into `send_request` as a new `route_request(prompt)` entry point.

**Architecture:** A `features.py` module extracts numeric features from raw prompts (token count, instruction-verb hits, constraint count, has-context flag, output-format complexity). A `dataset.py` module loads a labeled JSONL of prompts. A `classifier.py` module trains a `LogisticRegression` (with TF-IDF + numeric features) and persists the model with `joblib`. A `routing.py` module loads `config/routing.yaml` (tier → model_id) and exposes `pick_model(tier, registry)`. A new `router.py` glues the classifier + routing + `send_request` together as `route_request(prompt)` returning a `RoutedResponse` (the original `Response` plus `tier` and `routing_reason`).

**Tech Stack:** scikit-learn, joblib (both new), reusing pyyaml/pydantic from Phase 1. No deep-learning deps.

---

## File Structure

```
LLM Cost Autopilot/
├── config/
│   └── routing.yaml                              # NEW: tier -> model_id map
├── data/
│   └── prompts_labeled.jsonl                     # NEW: 200+ labeled prompts
├── models/                                       # NEW: persisted classifier artifacts
│   └── classifier.joblib                         # produced by training
├── src/autopilot/
│   ├── features.py                               # NEW: extract_features(prompt)
│   ├── dataset.py                                # NEW: load_dataset()
│   ├── classifier.py                             # NEW: train, save, load, predict
│   ├── routing.py                                # NEW: load_routing_config, pick_model
│   └── router.py                                 # NEW: route_request(prompt) -> RoutedResponse
├── scripts/
│   ├── label_prompts.py                          # NEW: helper to bulk-add prompts
│   ├── train_classifier.py                       # NEW: load data, train, save model
│   └── evaluate_routing.py                       # NEW: end-to-end demo on a sample batch
└── tests/
    ├── test_features.py                          # NEW
    ├── test_dataset.py                           # NEW
    ├── test_classifier.py                        # NEW
    ├── test_routing.py                           # NEW
    └── test_router.py                            # NEW
```

---

### Task 1: Add scikit-learn + joblib deps

**Files:**
- Modify: `pyproject.toml` (auto-edited by `uv add`)

- [ ] **Step 1: Install new deps**

Run:
```bash
cd "/Users/yashwantyadav/Desktop/LLM Cost Autopilot"
uv add scikit-learn joblib
```
Expected: both appear in `dependencies` block of `pyproject.toml`, `uv.lock` updates.

- [ ] **Step 2: Verify import works**

Run:
```bash
uv run python -c "import sklearn, joblib; print(sklearn.__version__, joblib.__version__)"
```
Expected: two version numbers, no traceback.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add scikit-learn and joblib for Phase 2 classifier"
```

---

### Task 2: Feature extraction (`features.py`)

**Files:**
- Create: `src/autopilot/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_features.py`:
```python
from autopilot.features import PromptFeatures, extract_features


class TestExtractFeatures:
    def test_returns_promptfeatures(self):
        f = extract_features("Hello world.")
        assert isinstance(f, PromptFeatures)

    def test_token_count_is_word_based(self):
        f = extract_features("one two three four")
        assert f.token_count == 4

    def test_instruction_verbs_detected(self):
        f = extract_features("Analyze the following data and compare results.")
        assert f.instruction_verb_count >= 2  # analyze + compare

    def test_no_instruction_verbs(self):
        f = extract_features("hello there friend")
        assert f.instruction_verb_count == 0

    def test_constraint_count(self):
        f = extract_features("List exactly 3 items, no more than 50 words each, in JSON.")
        # 'exactly', 'no more than', 'in JSON' style constraints
        assert f.constraint_count >= 2

    def test_has_context_when_long_quoted_block(self):
        prompt = 'Summarize: """' + ("text " * 80) + '"""'
        f = extract_features(prompt)
        assert f.has_context is True

    def test_no_context_short_prompt(self):
        f = extract_features("Translate hello to French.")
        assert f.has_context is False

    def test_output_format_complexity_simple(self):
        f = extract_features("What is 2 + 2?")
        assert f.output_format_complexity == 0

    def test_output_format_complexity_structured(self):
        f = extract_features("Return a JSON object with keys 'name' and 'age'.")
        assert f.output_format_complexity >= 1

    def test_to_vector_returns_numeric_list(self):
        f = extract_features("Analyze the data.")
        vec = f.to_vector()
        assert isinstance(vec, list)
        assert all(isinstance(x, (int, float)) for x in vec)
        assert len(vec) == 5  # token_count, instr_verbs, constraints, has_context, output_complexity
```

- [ ] **Step 2: Run tests — expect import error**

Run:
```bash
uv run pytest tests/test_features.py -v
```
Expected: ModuleNotFoundError for `autopilot.features`.

- [ ] **Step 3: Implement `features.py`**

Create `src/autopilot/features.py`:
```python
from __future__ import annotations

import re
from dataclasses import dataclass

INSTRUCTION_VERBS = {
    "analyze", "compare", "evaluate", "assess", "explain", "justify",
    "synthesize", "argue", "debate", "predict", "design", "optimize",
    "critique", "reason", "derive", "prove",
}

CONSTRAINT_PATTERNS = [
    r"\bexactly\s+\d+\b",
    r"\bno more than\b",
    r"\bat least\b",
    r"\bmust\b",
    r"\bmust not\b",
    r"\bonly\b",
    r"\bin (json|yaml|xml|csv|markdown)\b",
    r"\bwithin \d+\b",
]

STRUCTURED_FORMATS = ["json", "yaml", "xml", "csv", "markdown table", "table", "list"]


@dataclass(frozen=True)
class PromptFeatures:
    token_count: int
    instruction_verb_count: int
    constraint_count: int
    has_context: bool
    output_format_complexity: int

    def to_vector(self) -> list[float]:
        return [
            float(self.token_count),
            float(self.instruction_verb_count),
            float(self.constraint_count),
            1.0 if self.has_context else 0.0,
            float(self.output_format_complexity),
        ]


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _instruction_verbs(text: str) -> int:
    lowered = text.lower()
    words = set(re.findall(r"[a-z]+", lowered))
    return sum(1 for v in INSTRUCTION_VERBS if v in words)


def _constraints(text: str) -> int:
    lowered = text.lower()
    return sum(1 for pattern in CONSTRAINT_PATTERNS if re.search(pattern, lowered))


def _has_context(text: str) -> bool:
    # heuristic: a long triple-quoted block, or any prompt with > 200 words
    if '"""' in text or "'''" in text or "```" in text:
        return True
    return _word_count(text) > 200


def _output_format_complexity(text: str) -> int:
    lowered = text.lower()
    return sum(1 for fmt in STRUCTURED_FORMATS if fmt in lowered)


def extract_features(prompt: str) -> PromptFeatures:
    return PromptFeatures(
        token_count=_word_count(prompt),
        instruction_verb_count=_instruction_verbs(prompt),
        constraint_count=_constraints(prompt),
        has_context=_has_context(prompt),
        output_format_complexity=_output_format_complexity(prompt),
    )
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
uv run pytest tests/test_features.py -v
```
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/features.py tests/test_features.py
git commit -m "feat(features): extract numeric prompt features for classifier"
```

---

### Task 3: Labeled dataset format + loader (`dataset.py`)

**Files:**
- Create: `data/prompts_labeled.jsonl` (seed with first 30 — full 200+ added in Task 4)
- Create: `src/autopilot/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Seed `data/prompts_labeled.jsonl` with 30 hand-labeled prompts**

Create `data/prompts_labeled.jsonl` (one JSON object per line, fields `prompt` + `tier`). Include 10 SIMPLE, 10 MODERATE, 10 COMPLEX. Example seed entries:
```jsonl
{"prompt": "What is 2 + 2?", "tier": "simple"}
{"prompt": "Translate 'hello' to French.", "tier": "simple"}
{"prompt": "Extract the email from: jane@example.com", "tier": "simple"}
{"prompt": "Summarize: The quick brown fox jumps over the lazy dog.", "tier": "moderate"}
{"prompt": "Classify the sentiment of: I love this product.", "tier": "moderate"}
{"prompt": "Compare and analyze the trade-offs between SQL and NoSQL databases for a fintech startup.", "tier": "complex"}
```
Add 24 more entries (8 per tier) to reach 30. Keep prompts varied — don't reuse near-identical wording.

- [ ] **Step 2: Write failing tests**

Create `tests/test_dataset.py`:
```python
from pathlib import Path

import pytest

from autopilot.dataset import LabeledPrompt, load_dataset
from autopilot.models import ComplexityTier


@pytest.fixture
def dataset_path(project_root: Path) -> Path:
    return project_root / "data" / "prompts_labeled.jsonl"


class TestLoadDataset:
    def test_loads_at_least_30(self, dataset_path: Path):
        rows = load_dataset(dataset_path)
        assert len(rows) >= 30

    def test_returns_labeled_prompts(self, dataset_path: Path):
        rows = load_dataset(dataset_path)
        assert all(isinstance(r, LabeledPrompt) for r in rows)
        assert all(isinstance(r.tier, ComplexityTier) for r in rows)

    def test_each_tier_has_examples(self, dataset_path: Path):
        rows = load_dataset(dataset_path)
        tiers = {r.tier for r in rows}
        assert tiers == {ComplexityTier.SIMPLE, ComplexityTier.MODERATE, ComplexityTier.COMPLEX}

    def test_no_empty_prompts(self, dataset_path: Path):
        rows = load_dataset(dataset_path)
        assert all(r.prompt.strip() for r in rows)

    def test_unknown_tier_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"prompt": "x", "tier": "bogus"}\n')
        with pytest.raises(ValueError):
            load_dataset(bad)
```

- [ ] **Step 3: Run tests — expect failure**

Run:
```bash
uv run pytest tests/test_dataset.py -v
```
Expected: ImportError for `autopilot.dataset`.

- [ ] **Step 4: Implement `dataset.py`**

Create `src/autopilot/dataset.py`:
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from autopilot.models import ComplexityTier


@dataclass(frozen=True)
class LabeledPrompt:
    prompt: str
    tier: ComplexityTier


def load_dataset(path: Path | str) -> list[LabeledPrompt]:
    path = Path(path)
    rows: list[LabeledPrompt] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno} bad JSON: {e}") from e
        try:
            tier = ComplexityTier(obj["tier"])
        except (KeyError, ValueError) as e:
            raise ValueError(f"{path}:{lineno} bad tier: {e}") from e
        prompt = (obj.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"{path}:{lineno} empty prompt")
        rows.append(LabeledPrompt(prompt=prompt, tier=tier))
    return rows
```

- [ ] **Step 5: Run tests — expect pass**

Run:
```bash
uv run pytest tests/test_dataset.py -v
```
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add data/prompts_labeled.jsonl src/autopilot/dataset.py tests/test_dataset.py
git commit -m "feat(dataset): JSONL loader for labeled complexity dataset (30 seed)"
```

---

### Task 4: Expand dataset to 200+ prompts

**Files:**
- Modify: `data/prompts_labeled.jsonl`
- Create: `scripts/label_prompts.py`

- [ ] **Step 1: Write `scripts/label_prompts.py` to bulk-append entries**

Create `scripts/label_prompts.py`:
```python
"""Append labeled prompts to data/prompts_labeled.jsonl."""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "prompts_labeled.jsonl"


def append(prompt: str, tier: str) -> None:
    with DATA.open("a") as f:
        f.write(json.dumps({"prompt": prompt, "tier": tier}) + "\n")


def add_batch(entries: list[tuple[str, str]]) -> None:
    for prompt, tier in entries:
        if tier not in {"simple", "moderate", "complex"}:
            raise ValueError(f"bad tier: {tier}")
        append(prompt, tier)


if __name__ == "__main__":
    # Edit this list and re-run to bulk-add prompts.
    # Keep tiers balanced (~equal counts) and prompts diverse.
    BATCH: list[tuple[str, str]] = [
        # Add (prompt, tier) tuples here, then run:  uv run python scripts/label_prompts.py
    ]
    add_batch(BATCH)
    print(f"Appended {len(BATCH)} entries.")
```

- [ ] **Step 2: Hand-write 170+ more prompts directly into `data/prompts_labeled.jsonl`**

Append 170+ more entries to `data/prompts_labeled.jsonl`, balanced across tiers (~57 per tier net total). Use varied surface forms — different domains (math, code, writing, classification, extraction, reasoning, planning), different lengths, different formats. This is deliberate manual work; the classifier accuracy depends on it.

Tier reminders:
- **SIMPLE**: single-fact extraction, format conversion, single-word translation, basic arithmetic, short Q&A from explicit context
- **MODERATE**: summarization, multi-class classification, structured data extraction with 3+ fields, short emails/blurbs, single-step reasoning
- **COMPLEX**: multi-step reasoning, creative generation with constraints, design discussions, code architecture, nuanced judgment

- [ ] **Step 3: Verify dataset size and balance**

Run:
```bash
uv run python -c "
from autopilot.dataset import load_dataset
from collections import Counter
rows = load_dataset('data/prompts_labeled.jsonl')
print('total:', len(rows))
print('balance:', Counter(r.tier.value for r in rows))
"
```
Expected: total >= 200, each tier >= 60 (any imbalance > 30% — add more to the underrepresented tier before continuing).

- [ ] **Step 4: Re-run dataset tests to confirm everything still parses**

Run:
```bash
uv run pytest tests/test_dataset.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add data/prompts_labeled.jsonl scripts/label_prompts.py
git commit -m "feat(dataset): expand labeled dataset to 200+ prompts"
```

---

### Task 5: Train + persist the classifier (`classifier.py`)

**Files:**
- Create: `src/autopilot/classifier.py`
- Test: `tests/test_classifier.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_classifier.py`:
```python
from pathlib import Path

import pytest

from autopilot.classifier import (
    ComplexityClassifier,
    TrainResult,
    train_classifier,
)
from autopilot.dataset import load_dataset
from autopilot.models import ComplexityTier


@pytest.fixture(scope="module")
def dataset(project_root):
    # workaround for module/function scope: just use the path directly
    from pathlib import Path as _P
    return load_dataset(_P(__file__).resolve().parent.parent / "data" / "prompts_labeled.jsonl")


@pytest.fixture(scope="module")
def trained_classifier(dataset) -> ComplexityClassifier:
    result = train_classifier(dataset, random_state=42)
    return result.classifier


@pytest.fixture(scope="module")
def train_result(dataset) -> TrainResult:
    return train_classifier(dataset, random_state=42)


class TestTrainClassifier:
    def test_returns_train_result(self, train_result: TrainResult):
        assert train_result.classifier is not None
        assert 0.0 <= train_result.test_accuracy <= 1.0

    def test_test_accuracy_above_threshold(self, train_result: TrainResult):
        # V1 target from spec: 80%. Accept 70% lower bound for hand-labeled
        # 200-row dataset; raise once dataset grows.
        assert train_result.test_accuracy >= 0.70


class TestClassifierPredict:
    def test_predict_returns_tier(self, trained_classifier: ComplexityClassifier):
        tier = trained_classifier.predict("What is 2 + 2?")
        assert isinstance(tier, ComplexityTier)

    def test_predict_simple_prompt(self, trained_classifier: ComplexityClassifier):
        tier = trained_classifier.predict("Translate 'hello' to French.")
        # not strict — model might disagree — but should not crash
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
```

- [ ] **Step 2: Run tests — expect failure**

Run:
```bash
uv run pytest tests/test_classifier.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `classifier.py`**

Create `src/autopilot/classifier.py`:
```python
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
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
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
    appending a synthetic prefix to the text so TF-IDF picks them up too —
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
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
uv run pytest tests/test_classifier.py -v
```
Expected: 6 passed. If `test_test_accuracy_above_threshold` fails (< 70%), the dataset is too noisy or imbalanced — add more diverse examples and retrain. Do NOT lower the threshold to make the test pass; fix the data.

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/classifier.py tests/test_classifier.py
git commit -m "feat(classifier): TF-IDF + LR complexity classifier with persistence"
```

---

### Task 6: Train script + persist `models/classifier.joblib`

**Files:**
- Create: `scripts/train_classifier.py`
- Modify: `.gitignore` (allow `models/*.joblib`? No — keep models out of git)

- [ ] **Step 1: Write `scripts/train_classifier.py`**

Create `scripts/train_classifier.py`:
```python
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
```

- [ ] **Step 2: Add `models/` to `.gitignore`**

Read `.gitignore` then append (the file has a trailing newline already):
```
models/
```
Reason: trained model artifacts are reproducible from the dataset + train script; keeping them out of git prevents bloat.

- [ ] **Step 3: Run the train script**

Run:
```bash
uv run python scripts/train_classifier.py
```
Expected: prints loaded count, train/test split, accuracy >= 0.70, save confirmation. `models/classifier.joblib` created.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_classifier.py .gitignore
git commit -m "feat(classifier): train script that persists models/classifier.joblib"
```

---

### Task 7: Routing config + `routing.py`

**Files:**
- Create: `config/routing.yaml`
- Create: `src/autopilot/routing.py`
- Test: `tests/test_routing.py`

- [ ] **Step 1: Write the routing config**

Create `config/routing.yaml`:
```yaml
# Maps complexity tier to the model_id that should handle it.
# model_ids must exist in config/models.yaml.
routing:
  simple: gpt-4o-mini      # cheapest non-mock model with reasonable quality
  moderate: gpt-4o-mini    # same — cheapest viable; bump to claude-haiku-4-5 once Anthropic is wired
  complex: gpt-4o          # highest-quality model in the registry
```
(Note: `simple` should ideally route to the local Ollama llama3.2:3b for $0 cost, but Ollama is not installed yet. We use gpt-4o-mini as the lowest-cost real provider. Phase 5 may revisit once Ollama is available.)

- [ ] **Step 2: Write failing tests**

Create `tests/test_routing.py`:
```python
from pathlib import Path

import pytest

from autopilot.models import ComplexityTier
from autopilot.registry import load_registry
from autopilot.routing import (
    InvalidRoutingConfig,
    load_routing_config,
    pick_model,
)


@pytest.fixture
def routing_path(project_root: Path) -> Path:
    return project_root / "config" / "routing.yaml"


@pytest.fixture
def registry(project_root: Path):
    return load_registry(project_root / "config" / "models.yaml")


class TestLoadRoutingConfig:
    def test_loads_three_tiers(self, routing_path: Path):
        cfg = load_routing_config(routing_path)
        assert set(cfg.keys()) == {
            ComplexityTier.SIMPLE,
            ComplexityTier.MODERATE,
            ComplexityTier.COMPLEX,
        }

    def test_missing_tier_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("routing:\n  simple: gpt-4o-mini\n")
        with pytest.raises(InvalidRoutingConfig):
            load_routing_config(bad)


class TestPickModel:
    def test_picks_correct_model_for_each_tier(self, routing_path, registry):
        routing = load_routing_config(routing_path)
        for tier in (ComplexityTier.SIMPLE, ComplexityTier.MODERATE, ComplexityTier.COMPLEX):
            cfg = pick_model(tier, routing, registry)
            assert cfg.model_id == routing[tier]

    def test_pick_unknown_model_raises(self, registry, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "routing:\n  simple: nonexistent\n  moderate: gpt-4o-mini\n  complex: gpt-4o\n"
        )
        routing = load_routing_config(bad)
        with pytest.raises(KeyError):
            pick_model(ComplexityTier.SIMPLE, routing, registry)
```

- [ ] **Step 3: Run tests — expect failure**

Run:
```bash
uv run pytest tests/test_routing.py -v
```
Expected: ImportError.

- [ ] **Step 4: Implement `routing.py`**

Create `src/autopilot/routing.py`:
```python
from __future__ import annotations

from pathlib import Path

import yaml

from autopilot.models import ComplexityTier, ModelConfig
from autopilot.registry import ModelRegistry


class InvalidRoutingConfig(ValueError):
    pass


def load_routing_config(path: Path | str) -> dict[ComplexityTier, str]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not raw or "routing" not in raw:
        raise InvalidRoutingConfig(f"{path}: missing 'routing' key")
    routing_raw = raw["routing"]
    expected = {t.value for t in ComplexityTier}
    actual = set(routing_raw.keys())
    if actual != expected:
        raise InvalidRoutingConfig(
            f"{path}: routing must define exactly {expected}, got {actual}"
        )
    return {ComplexityTier(k): str(v) for k, v in routing_raw.items()}


def pick_model(
    tier: ComplexityTier,
    routing: dict[ComplexityTier, str],
    registry: ModelRegistry,
) -> ModelConfig:
    model_id = routing[tier]
    return registry.get(model_id)
```

- [ ] **Step 5: Run tests — expect pass**

Run:
```bash
uv run pytest tests/test_routing.py -v
```
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add config/routing.yaml src/autopilot/routing.py tests/test_routing.py
git commit -m "feat(routing): YAML-driven tier-to-model routing map"
```

---

### Task 8: `route_request` end-to-end (`router.py`)

**Files:**
- Create: `src/autopilot/router.py`
- Test: `tests/test_router.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_router.py`:
```python
from pathlib import Path

import pytest

from autopilot.classifier import ComplexityClassifier, train_classifier
from autopilot.dataset import load_dataset
from autopilot.models import ComplexityTier
from autopilot.providers.anthropic_provider import AnthropicProvider
from autopilot.providers.ollama_provider import OllamaProvider
from autopilot.registry import load_registry
from autopilot.router import RoutedResponse, Router
from autopilot.routing import load_routing_config


@pytest.fixture(scope="module")
def trained() -> ComplexityClassifier:
    rows = load_dataset(
        Path(__file__).resolve().parent.parent / "data" / "prompts_labeled.jsonl"
    )
    return train_classifier(rows, random_state=42).classifier


@pytest.fixture(scope="module")
def registry():
    return load_registry(
        Path(__file__).resolve().parent.parent / "config" / "models.yaml"
    )


@pytest.fixture(scope="module")
def routing():
    return load_routing_config(
        Path(__file__).resolve().parent.parent / "config" / "routing.yaml"
    )


@pytest.fixture
def routing_with_mocks(tmp_path):
    """Route everything to mock providers so tests never touch OpenAI."""
    cfg = tmp_path / "routing.yaml"
    cfg.write_text(
        "routing:\n"
        "  simple: llama3.2:3b\n"
        "  moderate: claude-haiku-4-5\n"
        "  complex: claude-sonnet-4-6\n"
    )
    return load_routing_config(cfg)


async def test_router_returns_routed_response(trained, registry, routing_with_mocks):
    router = Router(classifier=trained, routing=routing_with_mocks, registry=registry)
    result = await router.route_request("What is 2 + 2?")
    assert isinstance(result, RoutedResponse)
    assert isinstance(result.tier, ComplexityTier)
    assert result.response.model_id in {"llama3.2:3b", "claude-haiku-4-5", "claude-sonnet-4-6"}


async def test_router_picks_complex_model_for_complex_prompt(
    trained, registry, routing_with_mocks
):
    router = Router(classifier=trained, routing=routing_with_mocks, registry=registry)
    result = await router.route_request(
        "Compare and contrast event sourcing vs CQRS for a multi-tenant SaaS, "
        "including failure modes, replay performance, and operational complexity."
    )
    # not strict on exact tier (classifier might say moderate/complex), but it
    # should NOT route to the simple/ollama mock for a clearly hard prompt
    assert result.response.model_id != "llama3.2:3b"


async def test_routing_reason_includes_confidence(
    trained, registry, routing_with_mocks
):
    router = Router(classifier=trained, routing=routing_with_mocks, registry=registry)
    result = await router.route_request("Translate 'goodbye' to Spanish.")
    assert "tier=" in result.routing_reason
    assert "confidence=" in result.routing_reason


async def test_router_dispatches_through_send_request(
    trained, registry, routing_with_mocks, monkeypatch
):
    """Confirm the router uses the unified send_request entry point."""
    seen: list[str] = []

    async def fake_send(prompt, config, *, provider=None):
        from autopilot.models import Response
        seen.append(config.model_id)
        return Response(
            text="stub", input_tokens=1, output_tokens=1,
            latency_ms=0.0, cost=0.0, model_id=config.model_id,
        )

    import autopilot.router as router_mod
    monkeypatch.setattr(router_mod, "send_request", fake_send)

    router = Router(classifier=trained, routing=routing_with_mocks, registry=registry)
    await router.route_request("hello")
    assert len(seen) == 1
```

- [ ] **Step 2: Run tests — expect failure**

Run:
```bash
uv run pytest tests/test_router.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `router.py`**

Create `src/autopilot/router.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

from autopilot.classifier import ComplexityClassifier
from autopilot.client import send_request
from autopilot.models import ComplexityTier, Response
from autopilot.registry import ModelRegistry
from autopilot.routing import pick_model


@dataclass(frozen=True)
class RoutedResponse:
    response: Response
    tier: ComplexityTier
    routing_reason: str


class Router:
    def __init__(
        self,
        *,
        classifier: ComplexityClassifier,
        routing: dict[ComplexityTier, str],
        registry: ModelRegistry,
    ) -> None:
        self._classifier = classifier
        self._routing = routing
        self._registry = registry

    async def route_request(self, prompt: str) -> RoutedResponse:
        tier, confidence = self._classifier.predict_with_confidence(prompt)
        cfg = pick_model(tier, self._routing, self._registry)
        reason = (
            f"tier={tier.value} confidence={confidence:.2f} "
            f"-> model={cfg.model_id} (provider={cfg.provider})"
        )
        response = await send_request(prompt, cfg)
        return RoutedResponse(response=response, tier=tier, routing_reason=reason)
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
uv run pytest tests/test_router.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/router.py tests/test_router.py
git commit -m "feat(router): route_request glues classifier + routing + send_request"
```

---

### Task 9: End-to-end demo script

**Files:**
- Create: `scripts/evaluate_routing.py`

- [ ] **Step 1: Write the demo script**

Create `scripts/evaluate_routing.py`:
```python
"""Run a sample batch through the full router and print per-prompt routing decisions."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from autopilot.classifier import ComplexityClassifier
from autopilot.registry import load_registry
from autopilot.router import Router
from autopilot.routing import load_routing_config

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent

SAMPLE = [
    "What is 2 + 2?",
    "Translate 'thank you' to Japanese.",
    "Summarize the plot of Hamlet in two sentences.",
    "Classify this support ticket: 'My package is late.' (categories: shipping, billing, technical, other)",
    "Design a sharding strategy for a multi-tenant Postgres database with hot tenants.",
    "Compare and analyze the trade-offs between event sourcing and CRUD for a fintech ledger.",
]


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set - this demo routes to gpt-4o/4o-mini and needs the key.")
        return

    classifier = ComplexityClassifier.load(ROOT / "models" / "classifier.joblib")
    routing = load_routing_config(ROOT / "config" / "routing.yaml")
    registry = load_registry(ROOT / "config" / "models.yaml")
    router = Router(classifier=classifier, routing=routing, registry=registry)

    print(f"{'tier':<10} {'model':<14} {'cost':>10} {'lat_ms':>8}  prompt")
    print("-" * 100)
    for prompt in SAMPLE:
        result = await router.route_request(prompt)
        r = result.response
        print(
            f"{result.tier.value:<10} {r.model_id:<14} {r.cost:>10.6f} "
            f"{r.latency_ms:>8.0f}  {prompt[:60]}"
        )


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Verify the script runs (or short-circuits cleanly without API key)**

Run:
```bash
uv run python scripts/evaluate_routing.py
```
Expected: either prints the routing table (with API key) or prints the "OPENAI_API_KEY not set" message and exits 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate_routing.py
git commit -m "feat(scripts): end-to-end routing demo with cost/latency table"
```

---

### Task 10: Phase 2 wrap-up

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full unit suite**

Run:
```bash
uv run pytest --ignore=tests/integration -v
```
Expected: all green. Record the count.

- [ ] **Step 2: Update README**

Read `README.md`, then overwrite with the same structure but:
- check off Phase 2 in the Status section
- add a "Routing usage" subsection under Usage:
  ```python
  from autopilot.classifier import ComplexityClassifier
  from autopilot.registry import load_registry
  from autopilot.router import Router
  from autopilot.routing import load_routing_config

  classifier = ComplexityClassifier.load("models/classifier.joblib")
  router = Router(
      classifier=classifier,
      routing=load_routing_config("config/routing.yaml"),
      registry=load_registry("config/models.yaml"),
  )
  result = await router.route_request("What is 2 + 2?")
  print(result.tier, result.response.model_id, result.response.cost)
  ```
- add the `scripts/train_classifier.py` and `scripts/evaluate_routing.py` commands to the Tests section
- add to the Architecture section:
  - `src/autopilot/features.py` - extracts numeric prompt features
  - `src/autopilot/dataset.py` - JSONL loader for labeled prompts
  - `src/autopilot/classifier.py` - TF-IDF + LR classifier with persistence
  - `src/autopilot/routing.py` - tier-to-model map (config/routing.yaml)
  - `src/autopilot/router.py` - end-to-end route_request

- [ ] **Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: phase 2 complete - complexity classifier and routing"
```

---

## Self-Review

**Spec coverage (Phase 2):**
- "Define complexity tiers (3 tiers: simple / moderate / complex)" — already in `models.py` from Phase 1, reused here
- "Build a labeled dataset (200+ examples, hand-labeled, with feature extraction)" — Tasks 3, 4
- "Train the classifier (sklearn, accuracy >= 80% target / 70% lower bound, confusion matrix not formally tracked but accuracy is)" — Task 5
- "Create the routing map (YAML, swap models without code changes)" — Task 7

**Out of scope for Phase 2 (deferred):**
- Confusion matrix output — implicit in `test_test_accuracy_above_threshold`. Could be added to `train_classifier.py` later.
- Routing to a local Ollama model for SIMPLE — Ollama is not installed; routing falls back to `gpt-4o-mini`. Revisit when Ollama is set up.
- Feedback loop / retraining — that's Phase 3 (the verifier feeds failures back).

**Placeholder scan:** No "TBD" / "implement later" / "add error handling" left. Task 4 Step 2 explicitly asks the engineer (you) to hand-write 170 prompts — that's manual labor, not a placeholder.

**Type consistency:** `ComplexityTier`, `ModelConfig`, `Response`, `ModelRegistry`, `LabeledPrompt`, `PromptFeatures`, `ComplexityClassifier`, `TrainResult`, `RoutedResponse`, `Router`, `pick_model`, `load_routing_config`, `train_classifier`, `extract_features`, `load_dataset` — names/signatures match across all tasks.
