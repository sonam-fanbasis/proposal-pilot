"""Vector index over chunk embeddings.

Backed by `sqlite-vec`, which stores the vectors in the same SQLite file as
the relational data and answers KNN with a real index. The column is declared
`distance_metric=cosine`, so `similarity = 1 - distance` for our
L2-normalised vectors.

The capability that matters for this product is **metadata-filtered KNN**:

    SELECT chunk_id, distance FROM vec_chunks
    WHERE embedding MATCH ? AND k = ? AND doc_id IN (...)

"Have we done anything in fintech with React Native?" is a metadata question
wearing a semantic question's clothes. Answering it by running an unfiltered
KNN and discarding non-matching hits afterwards is what most implementations
do, and it silently fails whenever the matching documents are a small part of
the corpus -- the top-k fills up with irrelevant industries and the right
answer never makes it into the candidate set. Pushing the filter into the
index means k is spent entirely on documents that already qualify.

A NumPy brute-force scan stands in if the extension will not load.
"""

from __future__ import annotations

import math
import struct

import numpy as np

from . import db
from .config import get_settings


def _pack(vector: np.ndarray) -> bytes:
    vector = np.asarray(vector, dtype=np.float32).ravel()
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def upsert(chunk_ids: list[int], doc_ids: list[int], vectors: np.ndarray) -> None:
    conn = db.get_conn()
    table = "vec_chunks" if db.VEC_AVAILABLE else "vec_chunks_fallback"
    rows = [
        (int(cid), int(did), _pack(vec))
        for cid, did, vec in zip(chunk_ids, doc_ids, vectors)
    ]
    conn.executemany(f"DELETE FROM {table} WHERE chunk_id = ?", [(r[0],) for r in rows])
    conn.executemany(
        f"INSERT INTO {table}(chunk_id, doc_id, embedding) VALUES (?, ?, ?)", rows
    )
    conn.commit()


def delete_document(doc_id: int) -> None:
    table = "vec_chunks" if db.VEC_AVAILABLE else "vec_chunks_fallback"
    conn = db.get_conn()
    conn.execute(f"DELETE FROM {table} WHERE doc_id = ?", (int(doc_id),))
    conn.commit()


def count() -> int:
    table = "vec_chunks" if db.VEC_AVAILABLE else "vec_chunks_fallback"
    row = db.query_one(f"SELECT COUNT(*) AS n FROM {table}")
    return int(row["n"]) if row else 0


def search(
    query_vector: np.ndarray, k: int, doc_ids: list[int] | None = None
) -> list[tuple[int, float]]:
    """Return `(chunk_id, cosine_similarity)` best-first.

    `doc_ids=None` searches everything; an empty list means "the filter
    matched no documents", which is a real answer, not an error.
    """
    if doc_ids is not None and len(doc_ids) == 0:
        return []
    if count() == 0:
        return []

    if db.VEC_AVAILABLE:
        return _search_vec0(query_vector, k, doc_ids)
    return _search_numpy(query_vector, k, doc_ids)


def _search_vec0(
    query_vector: np.ndarray, k: int, doc_ids: list[int] | None
) -> list[tuple[int, float]]:
    conn = db.get_conn()
    blob = _pack(query_vector)

    if doc_ids is None:
        sql = "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ?"
        params: list = [blob, int(k)]
    else:
        # vec0 runs the KNN once per value in an IN list, so ask for fewer
        # per document and re-sort. Without the cap a 200-document filter
        # would request 200*k rows.
        per_doc = max(1, min(k, math.ceil(k / max(1, len(doc_ids) ** 0.5))))
        placeholders = ",".join("?" for _ in doc_ids)
        sql = (
            f"SELECT chunk_id, distance FROM vec_chunks "
            f"WHERE embedding MATCH ? AND k = ? AND doc_id IN ({placeholders})"
        )
        params = [blob, int(per_doc), *[int(d) for d in doc_ids]]

    rows = conn.execute(sql, params).fetchall()
    results = [(int(r["chunk_id"]), 1.0 - float(r["distance"])) for r in rows]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]


def _search_numpy(
    query_vector: np.ndarray, k: int, doc_ids: list[int] | None
) -> list[tuple[int, float]]:
    if doc_ids is None:
        rows = db.query("SELECT chunk_id, embedding FROM vec_chunks_fallback")
    else:
        placeholders = ",".join("?" for _ in doc_ids)
        rows = db.query(
            f"SELECT chunk_id, embedding FROM vec_chunks_fallback WHERE doc_id IN ({placeholders})",
            [int(d) for d in doc_ids],
        )
    if not rows:
        return []
    matrix = np.vstack([_unpack(r["embedding"]) for r in rows])
    sims = matrix @ np.asarray(query_vector, dtype=np.float32).ravel()
    top = np.argsort(-sims)[:k]
    return [(int(rows[i]["chunk_id"]), float(sims[i])) for i in top]
