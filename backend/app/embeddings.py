"""Embeddings, with a provider that degrades instead of failing.

Primary: Gemini `gemini-embedding-001`, requested at 768-d. It is used
*asymmetrically* --
documents are embedded with `task_type=RETRIEVAL_DOCUMENT` and queries with
`RETRIEVAL_QUERY`. Google trains those two projections as a pair, and using
them correctly is a free accuracy win that a surprising number of RAG
implementations miss by embedding both sides identically.

Fallback: a deterministic hashing embedder (character n-grams + hashed word
unigrams, sublinear term frequency, L2-normalised) at the same 768 dims. It
is not semantic -- it cannot match "revenue" to "turnover" -- but it is
genuinely useful for the proper-noun-heavy queries this product sees
("Zenith", "HIPAA", "React Native"), it needs no key or network, and it keeps
the app demoable offline. It shares the embedding width with Gemini so a
corpus does not need re-indexing when a key is added or removed.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter

import numpy as np

from .config import get_settings

_WORD = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")
_client = None
_warned = False

DOCUMENT = "RETRIEVAL_DOCUMENT"
QUERY = "RETRIEVAL_QUERY"


# --- offline hashing embedder ----------------------------------------------

def _hash_bucket(token: str, dim: int) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big") % dim


def _hash_embed_one(text: str, dim: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    words = _WORD.findall(text.lower())
    if not words:
        return vector

    counts: Counter[str] = Counter(words)
    # Word bigrams keep a little word order, which matters for
    # "fixed bid" vs "bid fixed".
    counts.update(f"{a}_{b}" for a, b in zip(words, words[1:]))
    # Character 4-grams give partial-match robustness against typos and
    # morphology ("scalability" ~ "scalable").
    joined = " ".join(words)
    counts.update(joined[i : i + 4] for i in range(0, max(0, len(joined) - 3), 2))

    for token, count in counts.items():
        bucket = _hash_bucket(token, dim)
        # Sublinear tf: a word repeated 50 times is not 50x more meaningful.
        vector[bucket] += 1.0 + math.log(count)

    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _hash_embed(texts: list[str], dim: int) -> np.ndarray:
    return np.vstack([_hash_embed_one(t, dim) for t in texts]) if texts else np.zeros((0, dim), dtype=np.float32)


# --- Gemini ----------------------------------------------------------------

def _get_client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=get_settings().gemini_api_key)
    return _client


def _gemini_embed(texts: list[str], task_type: str, dim: int) -> np.ndarray:
    from google.genai import types

    client = _get_client()
    settings = get_settings()
    vectors: list[list[float]] = []

    # The API caps a batch at 100 inputs; larger ingests are chunked.
    for start in range(0, len(texts), 100):
        batch = texts[start : start + 100]
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = client.models.embed_content(
                    model=settings.gemini_embed_model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        # Ask for the index width explicitly. Truncated vectors
                        # are not unit-length, so they are re-normalised below.
                        output_dimensionality=dim,
                    ),
                )
                vectors.extend(list(e.values) for e in response.embeddings)
                last_error = None
                break
            except Exception as exc:
                message = str(exc)
                # A retired model is permanent -- retrying wastes 4.5 seconds
                # and buries the one line that says how to fix it.
                if "NOT_FOUND" in message or "not found" in message:
                    raise RuntimeError(
                        f"Embedding model '{settings.gemini_embed_model}' is unavailable. "
                        f"List current models with: curl -s "
                        f"'https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY' "
                        f"then set GEMINI_EMBED_MODEL in backend/.env. Original error: {message[:200]}"
                    ) from exc
                last_error = exc  # transient 429/503 on the free tier
                time.sleep(1.5 * (attempt + 1))
        if last_error is not None:
            raise last_error

    array = np.asarray(vectors, dtype=np.float32)
    if array.shape[1] != dim:  # pragma: no cover - guards a model swap
        raise ValueError(
            f"Embedding model returned {array.shape[1]} dims but the index expects {dim}. "
            f"Set EMBEDDING_DIM={array.shape[1]} and re-ingest."
        )
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.where(norms == 0, 1, norms)


# --- public API -------------------------------------------------------------

def active_backend() -> str:
    settings = get_settings()
    return "gemini" if settings.resolved_provider == "gemini" else "hashing"


def embed(texts: list[str], task_type: str = DOCUMENT) -> np.ndarray:
    """Return L2-normalised embeddings; cosine similarity is then a dot product."""
    global _warned
    settings = get_settings()
    dim = settings.embedding_dim
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)

    if settings.resolved_provider == "gemini":
        try:
            return _gemini_embed(texts, task_type, dim)
        except Exception as exc:
            if not _warned:
                print(f"[embeddings] Gemini unavailable ({exc}); using offline hashing embedder.")
                _warned = True
    return _hash_embed(texts, dim)


def embed_one(text: str, task_type: str = QUERY) -> np.ndarray:
    return embed([text], task_type)[0]
