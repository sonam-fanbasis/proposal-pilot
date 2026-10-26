"""Local embedding and reranking models for the Report Lab.

Both run on CPU through ONNX (fastembed), load lazily on first use and are
shared across threads. Local models are the right call for this workload:

* a 200-page report produces ~1,000 chunks, which embeds locally in seconds
  but runs into the free-tier embedding quota in minutes;
* a cross-encoder reranker reads the question and the passage *together*,
  which is what separates "South Asia growth" from "Sub-Saharan Africa
  growth" -- two passages a bi-encoder embeds almost identically.

If the models cannot be loaded (no network on first run), embeddings fall back
to the deterministic hashing embedder and reranking to the fused rank, so the
pipeline degrades instead of failing.
"""

from __future__ import annotations

import threading

import numpy as np

from .config import get_settings

# bge and arctic-embed models are trained with an instruction prefix on the query side only.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_lock = threading.Lock()
_embedder = None
_reranker = None
_embedder_failed = False
_reranker_failed = False


def embedding_dim() -> int:
    return get_settings().report_embed_dim


def _get_embedder():
    global _embedder, _embedder_failed
    if _embedder is None and not _embedder_failed:
        with _lock:
            if _embedder is None and not _embedder_failed:
                try:
                    from fastembed import TextEmbedding

                    settings = get_settings()
                    _embedder = TextEmbedding(settings.report_embed_model, cache_dir=str(settings.model_cache_dir))
                except Exception as exc:  # pragma: no cover - network/platform
                    print(f"[report_models] embedder unavailable ({exc}); using hashing embedder")
                    _embedder_failed = True
    return _embedder


def _get_reranker():
    global _reranker, _reranker_failed
    if _reranker is None and not _reranker_failed:
        with _lock:
            if _reranker is None and not _reranker_failed:
                try:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    settings = get_settings()
                    _reranker = TextCrossEncoder(settings.report_rerank_model, cache_dir=str(settings.model_cache_dir))
                except Exception as exc:  # pragma: no cover
                    print(f"[report_models] reranker unavailable ({exc}); using fused rank")
                    _reranker_failed = True
    return _reranker


def embedder_name() -> str:
    return get_settings().report_embed_model if _get_embedder() else "hashing (fallback)"


def reranker_name() -> str:
    return get_settings().report_rerank_model if _get_reranker() else "none (fused rank)"


def embed_passages(texts: list[str]) -> np.ndarray:
    model = _get_embedder()
    if model is None:
        from .embeddings import _hash_embed

        return _hash_embed(texts, embedding_dim())
    vectors = np.asarray(list(model.embed(texts, batch_size=32)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def embed_query(text: str) -> np.ndarray:
    model = _get_embedder()
    if model is None:
        from .embeddings import _hash_embed

        return _hash_embed([text], embedding_dim())[0]
    vector = np.asarray(next(iter(model.embed([_QUERY_PREFIX + text]))), dtype=np.float32)
    return vector / (np.linalg.norm(vector) or 1.0)


def rerank(query: str, passages: list[str]) -> list[float] | None:
    """Cross-encoder relevance logits, or None when no reranker is available."""
    model = _get_reranker()
    if model is None or not passages:
        return None
    return [float(s) for s in model.rerank(query, passages, batch_size=16)]
