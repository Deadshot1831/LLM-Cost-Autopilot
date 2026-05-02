import os
from pathlib import Path

import pytest

from autopilot.client import send_request
from autopilot.registry import load_registry

pytestmark = pytest.mark.integration


BASELINE_PROMPTS = [
    "What is 2 + 2?",
    "Translate 'hello' to French.",
    "Summarize: The quick brown fox jumps over the lazy dog.",
    "Extract the email from: Contact me at jane@example.com.",
    "Classify sentiment (pos/neg): I love this product.",
    "Write a haiku about autumn.",
    "List three benefits of exercise.",
    "Convert 100 km to miles.",
    "What language is 'Bonjour le monde'?",
    "Reverse the string 'autopilot'.",
]


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def registry():
    return load_registry(PROJECT_ROOT / "config" / "models.yaml")


@pytest.fixture(autouse=True)
def _require_openai_key():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


async def test_gpt4o_mini_handles_all_baseline_prompts(registry):
    cfg = registry.get("gpt-4o-mini")
    for prompt in BASELINE_PROMPTS:
        r = await send_request(prompt, cfg)
        assert r.text.strip(), f"Empty response for prompt: {prompt!r}"
        assert r.input_tokens > 0
        assert r.output_tokens > 0
        assert r.cost > 0
