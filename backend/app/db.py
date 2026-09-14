"""SQLite storage, including the vector index.

Design note
-----------
The whole persistence layer is one SQLite file. That is a deliberate choice,
not a shortcut: `sqlite-vec` gives us a genuinely indexed ANN search that
supports *metadata-filtered* KNN (`... where embedding match ? and k = ? and
doc_id in (...)`), which is the one capability this product actually needs
from a vector database. Running it in-process removes a network hop from the
hot path and lets the reviewer launch the app without Docker, Postgres or a
vector-DB server.

If the extension cannot be loaded (unusual platform, locked-down build), the
app degrades to a NumPy brute-force scan in `vectorstore.py` rather than
failing -- correctness is preserved, only latency changes.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from .config import get_settings

_local = threading.local()
VEC_AVAILABLE = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    title           TEXT NOT NULL,
    sha256          TEXT NOT NULL UNIQUE,
    byte_size       INTEGER NOT NULL DEFAULT 0,
    page_count      INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'pending',   -- pending|parsing|embedding|ready|failed
    error           TEXT,
    -- Structured metadata extracted at ingest. Promoted to real columns
    -- (rather than living only in meta_json) because every one of them is a
    -- filter the retriever pushes down into the ANN query.
    client          TEXT,
    industry        TEXT,
    doc_type        TEXT,          -- proposal|case_study|sow|capability|rfp|other
    engagement_model TEXT,         -- fixed_bid|time_and_materials|dedicated_team|other
    year            INTEGER,
    value_usd       INTEGER,
    duration_months INTEGER,
    tech_stack      TEXT,          -- JSON array, also mirrored into doc_tech
    summary         TEXT,
    meta_json       TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Normalised tech tags. A separate table because "has Node AND AWS" is a
-- set-membership question, and LIKE '%node%' over a JSON blob gets that
-- wrong (it matches "nodemon", and cannot express AND without N scans).
CREATE TABLE IF NOT EXISTS doc_tech (
    doc_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tech    TEXT NOT NULL,
    PRIMARY KEY (doc_id, tech)
);
CREATE INDEX IF NOT EXISTS idx_doc_tech_tech ON doc_tech(tech);

CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL,
    section_path    TEXT NOT NULL DEFAULT '',
    page_from       INTEGER NOT NULL DEFAULT 0,
    page_to         INTEGER NOT NULL DEFAULT 0,
    text            TEXT NOT NULL,
    context_header  TEXT NOT NULL DEFAULT '',
    char_count      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL DEFAULT 'New conversation',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,          -- user|assistant
    content         TEXT NOT NULL,
    mode            TEXT NOT NULL DEFAULT 'answer',
    citations_json  TEXT NOT NULL DEFAULT '[]',
    gaps_json       TEXT NOT NULL DEFAULT '[]',
    confidence      REAL NOT NULL DEFAULT 0,
    latency_ms      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);

-- Observability: one row per retrieval, so the eval harness and the stats
-- endpoint report on real traffic rather than on a synthetic replay.
CREATE TABLE IF NOT EXISTS query_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query       TEXT NOT NULL,
    mode        TEXT NOT NULL,
    filters_json TEXT NOT NULL DEFAULT '{}',
    provider    TEXT NOT NULL DEFAULT '',
    n_candidates INTEGER NOT NULL DEFAULT 0,
    n_returned  INTEGER NOT NULL DEFAULT 0,
    confidence  REAL NOT NULL DEFAULT 0,
    retrieval_ms INTEGER NOT NULL DEFAULT 0,
    total_ms    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS eval_cases (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    question        TEXT NOT NULL,
    expect_keywords TEXT NOT NULL DEFAULT '[]',  -- JSON array, answer must contain
    expect_doc      TEXT,                        -- filename substring that should be retrieved
    note            TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    detail_json  TEXT NOT NULL DEFAULT '[]',
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _connect() -> sqlite3.Connection:
    global VEC_AVAILABLE
    settings = get_settings()
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        VEC_AVAILABLE = True
    except Exception:  # pragma: no cover - platform dependent
        VEC_AVAILABLE = False
    return conn


def get_conn() -> sqlite3.Connection:
    """One connection per thread. FastAPI runs sync handlers in a threadpool."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    if VEC_AVAILABLE:
        dim = get_settings().embedding_dim
        # `doc_id` is a vec0 metadata column, which is what makes
        # metadata-filtered KNN possible in a single indexed query.
        conn.execute(
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    doc_id   INTEGER,
                    embedding FLOAT[{dim}] distance_metric=cosine
                )"""
        )
    else:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS vec_chunks_fallback ("
            " chunk_id INTEGER PRIMARY KEY, doc_id INTEGER, embedding BLOB)"
        )
    conn.commit()


def query(sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
    return get_conn().execute(sql, params).fetchall()


def query_one(sql: str, params: tuple | list = ()) -> sqlite3.Row | None:
    return get_conn().execute(sql, params).fetchone()


def execute(sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
    conn = get_conn()
    cur = conn.execute(sql, params)
    conn.commit()
    return cur


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None
