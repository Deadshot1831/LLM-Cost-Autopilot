"""Lazy singleton wrapper around a small sentence-transformer.

Used by the verifier to compute cosine similarity between the cheap
model's answer and the reference model's answer. Loaded once per
process and reused for every verification call.

Default model: sentence-transformers/all-MiniLM-L6-v2 (~80MB, free, runs
on CPU in ~50ms per pair after warmup).
"""
from __future__ import annotations

import asyncio
from typing import Optional

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model: Optional[object] = None  # SentenceTransformer instance


def _load_model(name: str = DEFAULT_MODEL):
    """Load and cache the encoder. First call may take seconds (downloads
    the weights to ~/.cache/huggingface on the very first run)."""
    global _model
    if _model is None:
        # Lazy import so the (heavy) torch/transformers chain only loads
        # when verification is actually used.
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(name)
    return _model


def warmup(name: str = DEFAULT_MODEL) -> None:
    """Pre-load the model + run one encode pass so the FIRST real
    verification request doesn't pay the warmup cost."""
    m = _load_model(name)
    m.encode(["warmup"], show_progress_bar=False)


def _cosine_similarity_sync(a: str, b: str) -> float:
    """Encode two strings and return cosine similarity in [-1, 1].
    Empty strings short-circuit to 0.0."""
    if not a.strip() or not b.strip():
        return 0.0
    m = _load_model()
    import numpy as np
    embs = m.encode([a, b], show_progress_bar=False, normalize_embeddings=True)
    # Cosine on L2-normalized vectors == dot product
    return float(np.dot(embs[0], embs[1]))


async def cosine_similarity(a: str, b: str) -> float:
    """Async wrapper: runs the encode call in a thread so it doesn't
    block the asyncio event loop (torch ops are CPU-bound)."""
    return await asyncio.to_thread(_cosine_similarity_sync, a, b)
