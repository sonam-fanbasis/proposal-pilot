"""Report Lab storage and ingestion.

Reports live in their own tables, separate from the proposal library. The two
collections need different things from a store -- proposals are filtered by
client/industry/price, reports are navigated by chapter and section -- and
keeping them apart means neither retriever can be polluted by the other's
chunks.

    reports          one row per uploaded report, with its shape and profile
    report_sections  the heading tree (ordinal, parent, level, path, pages)
    report_chunks    typed chunks: text | table | figure | footnote | note
    vec_report_chunks  sqlite-vec index; report_id and section_ordinal are
                       metadata columns, so section-routed KNN is one query

Ingestion runs on a background thread and advances `stage` so the UI can show
which step of the pipeline a long document is in.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from . import db
from .config import get_settings
from .report_chunking import STRATEGIES, chunk_report, classify_shape
from .report_models import embed_passages, embedder_name, embedding_dim
from .report_parsing import parse_report
from .vectorstore import _pack, _unpack

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filename     TEXT NOT NULL,
    title        TEXT NOT NULL,
    sha256       TEXT NOT NULL UNIQUE,
    byte_size    INTEGER NOT NULL DEFAULT 0,
    page_count   INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending|processing|ready|failed
    stage        TEXT NOT NULL DEFAULT 'queued',   -- parsing|vision|chunking|embedding|done
    stage_detail TEXT NOT NULL DEFAULT '',
    error        TEXT,
    shape        TEXT NOT NULL DEFAULT '',         -- narrative|long_structured|visual
    profile_json TEXT NOT NULL DEFAULT '{}',
    ingest_ms    INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS report_sections (
    report_id  INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    parent     INTEGER,
    level      INTEGER NOT NULL,
    title      TEXT NOT NULL,
    path       TEXT NOT NULL,
    page_from  INTEGER NOT NULL,
    page_to    INTEGER NOT NULL,
    PRIMARY KEY (report_id, ordinal)
);

CREATE TABLE IF NOT EXISTS report_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id       INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL,
    section_ordinal INTEGER NOT NULL,
    section_path    TEXT NOT NULL,
    content_type    TEXT NOT NULL,
    page_from       INTEGER NOT NULL,
    page_to         INTEGER NOT NULL,
    text            TEXT NOT NULL,
    header          TEXT NOT NULL,
    note_text       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_report_chunks ON report_chunks(report_id, ordinal);
"""

_versions: dict[int, float] = {}   # report_id -> last write, for retrieval caches


def init() -> None:
    conn = db.get_conn()
    conn.executescript(SCHEMA)
    if db.VEC_AVAILABLE:
        conn.execute(
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS vec_report_chunks USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    report_id INTEGER,
                    section_ordinal INTEGER,
                    embedding FLOAT[{embedding_dim()}] distance_metric=cosine
                )"""
        )
    else:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS vec_report_fallback ("
            " chunk_id INTEGER PRIMARY KEY, report_id INTEGER, section_ordinal INTEGER, embedding BLOB)"
        )
    # Reports interrupted by a restart would otherwise spin forever in the UI.
    conn.execute(
        "UPDATE reports SET status='failed', error='Interrupted by a server restart; re-upload to retry.' "
        "WHERE status IN ('pending','processing')"
    )
    conn.commit()


def version(report_id: int) -> float:
    return _versions.get(report_id, 0.0)


def _set(report_id: int, **fields) -> None:
    keys = ", ".join(f"{k} = ?" for k in fields)
    db.execute(f"UPDATE reports SET {keys} WHERE id = ?", (*fields.values(), report_id))


class ReportError(RuntimeError):
    pass


def register(file_bytes: bytes, filename: str) -> tuple[int, Path]:
    if not filename.lower().endswith(".pdf"):
        raise ReportError("The Report Lab ingests PDF reports. Use the proposal library for DOCX/MD/TXT.")
    digest = hashlib.sha256(file_bytes).hexdigest()
    existing = db.query_one("SELECT id, title FROM reports WHERE sha256 = ?", (digest,))
    if existing:
        raise ReportError(f"This report is already loaded as '{existing['title']}' (report {existing['id']}).")
    settings = get_settings()
    folder = settings.upload_dir / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest[:12]}_{Path(filename).name}"
    path.write_bytes(file_bytes)
    cursor = db.execute(
        "INSERT INTO reports(filename, title, sha256, byte_size) VALUES (?, ?, ?, ?)",
        (filename, Path(filename).stem.replace("_", " ").replace("-", " "), digest, len(file_bytes)),
    )
    return int(cursor.lastrowid), path


def process(report_id: int, path: Path, use_vision: bool | None = None) -> dict:
    started = time.perf_counter()
    timings: dict[str, int] = {}
    mark = time.perf_counter()

    def progress(stage: str, detail: str) -> None:
        _set(report_id, stage=stage, stage_detail=detail)

    try:
        _set(report_id, status="processing", stage="parsing", stage_detail="Reading page layout")
        parsed = parse_report(path, use_vision=use_vision, progress=progress)
        if not any(e.kind in {"text", "table", "figure"} for e in parsed.elements):
            raise ReportError("No extractable text. A scanned PDF needs OCR before it can be indexed.")
        timings["parse_ms"] = int((time.perf_counter() - mark) * 1000)

        mark = time.perf_counter()
        shape = classify_shape(parsed.stats)
        progress("chunking", f"Building section tree ({shape.replace('_', ' ')} recipe)")
        sections, chunks = chunk_report(parsed.elements, parsed.title, shape)
        if not chunks:
            raise ReportError("Parsing produced no usable chunks.")
        timings["chunk_ms"] = int((time.perf_counter() - mark) * 1000)

        conn = db.get_conn()
        conn.execute("DELETE FROM report_sections WHERE report_id = ?", (report_id,))
        conn.execute("DELETE FROM report_chunks WHERE report_id = ?", (report_id,))
        _delete_vectors(report_id)
        conn.executemany(
            "INSERT INTO report_sections(report_id, ordinal, parent, level, title, path, page_from, page_to) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(report_id, s.ordinal, s.parent, s.level, s.title, s.path, s.page_from, s.page_to) for s in sections],
        )
        ids: list[int] = []
        for c in chunks:
            cur = conn.execute(
                "INSERT INTO report_chunks(report_id, ordinal, section_ordinal, section_path, content_type, "
                "page_from, page_to, text, header, note_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (report_id, c.ordinal, c.section_ordinal, c.section_path, c.content_type,
                 c.page_from, c.page_to, c.text, c.header, c.note_text),
            )
            ids.append(int(cur.lastrowid))
        conn.commit()

        mark = time.perf_counter()
        progress("embedding", f"Embedding {len(chunks)} chunks locally")
        vectors = embed_passages([c.embedding_text for c in chunks])
        _store_vectors(report_id, ids, [c.section_ordinal for c in chunks], vectors)
        timings["embed_ms"] = int((time.perf_counter() - mark) * 1000)

        counts: dict[str, int] = {}
        for c in chunks:
            counts[c.content_type] = counts.get(c.content_type, 0) + 1
        profile = {
            "stats": parsed.stats,
            "chunk_types": counts,
            "sections": len(sections) - 1,
            "chunks": len(chunks),
            "strategy": STRATEGIES[shape],
            "embedder": embedder_name(),
            "timings": timings,
        }
        _set(
            report_id, status="ready", stage="done", stage_detail="", title=parsed.title,
            page_count=parsed.page_count, shape=shape, profile_json=json.dumps(profile),
            ingest_ms=int((time.perf_counter() - started) * 1000), error=None,
        )
        _versions[report_id] = time.time()
        return {"report_id": report_id, "shape": shape, **profile}
    except Exception as exc:
        traceback.print_exc()
        _set(report_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
        raise


def ingest_async(report_id: int, path: Path) -> None:
    def run() -> None:
        try:
            process(report_id, path)
        except Exception:
            pass  # recorded on the row by process()

    threading.Thread(target=run, daemon=True, name=f"report-{report_id}").start()


def ingest_file(path: Path, use_vision: bool | None = None) -> dict:
    """Synchronous ingest (CLI/eval). Re-uses an existing copy of the same file."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    existing = db.query_one("SELECT id, status FROM reports WHERE sha256 = ?", (digest,))
    if existing and existing["status"] == "ready":
        return {"report_id": int(existing["id"]), "reused": True}
    if existing:
        delete(int(existing["id"]))
    report_id, stored = register(data, path.name)
    return process(report_id, stored, use_vision=use_vision)


def delete(report_id: int) -> None:
    _delete_vectors(report_id)
    db.execute("DELETE FROM report_chunks WHERE report_id = ?", (report_id,))
    db.execute("DELETE FROM report_sections WHERE report_id = ?", (report_id,))
    db.execute("DELETE FROM reports WHERE id = ?", (report_id,))
    _versions[report_id] = time.time()


# --- vectors ----------------------------------------------------------------

def _delete_vectors(report_id: int) -> None:
    table = "vec_report_chunks" if db.VEC_AVAILABLE else "vec_report_fallback"
    db.execute(f"DELETE FROM {table} WHERE report_id = ?", (report_id,))


def _store_vectors(report_id: int, chunk_ids: list[int], sections: list[int], vectors: np.ndarray) -> None:
    table = "vec_report_chunks" if db.VEC_AVAILABLE else "vec_report_fallback"
    conn = db.get_conn()
    conn.executemany(
        f"INSERT INTO {table}(chunk_id, report_id, section_ordinal, embedding) VALUES (?, ?, ?, ?)",
        [(cid, report_id, sec, _pack(vec)) for cid, sec, vec in zip(chunk_ids, sections, vectors)],
    )
    conn.commit()


def vector_search(
    query_vector: np.ndarray, report_id: int, k: int, sections: list[int] | None = None
) -> list[tuple[int, float]]:
    """KNN inside one report, optionally restricted to routed sections."""
    if sections is not None and not sections:
        return []
    if db.VEC_AVAILABLE:
        sql = "SELECT chunk_id, distance FROM vec_report_chunks WHERE embedding MATCH ? AND k = ? AND report_id = ?"
        params: list = [_pack(query_vector), int(k), int(report_id)]
        if sections is not None:
            sql += f" AND section_ordinal IN ({','.join('?' for _ in sections)})"
            params.extend(int(s) for s in sections)
        rows = db.query(sql, params)
        return sorted(((int(r["chunk_id"]), 1.0 - float(r["distance"])) for r in rows), key=lambda x: -x[1])[:k]

    sql = "SELECT chunk_id, embedding FROM vec_report_fallback WHERE report_id = ?"
    params = [int(report_id)]
    if sections is not None:
        sql += f" AND section_ordinal IN ({','.join('?' for _ in sections)})"
        params.extend(int(s) for s in sections)
    rows = db.query(sql, params)
    if not rows:
        return []
    matrix = np.vstack([_unpack(r["embedding"]) for r in rows])
    sims = matrix @ np.asarray(query_vector, dtype=np.float32)
    top = np.argsort(-sims)[:k]
    return [(int(rows[i]["chunk_id"]), float(sims[i])) for i in top]


# --- read helpers -----------------------------------------------------------

def report_row(report_id: int) -> dict | None:
    row = db.query_one("SELECT * FROM reports WHERE id = ?", (report_id,))
    return dict(row) if row else None


def list_reports() -> list[dict]:
    rows = db.query(
        "SELECT r.*, (SELECT COUNT(*) FROM report_chunks c WHERE c.report_id = r.id) AS chunk_count "
        "FROM reports r ORDER BY r.id DESC"
    )
    out = []
    for row in rows:
        data = dict(row)
        data["profile"] = json.loads(data.pop("profile_json") or "{}")
        data.pop("sha256", None)
        out.append(data)
    return out


def sections(report_id: int) -> list[dict]:
    rows = db.query(
        "SELECT s.*, (SELECT COUNT(*) FROM report_chunks c WHERE c.report_id = s.report_id "
        "AND c.section_ordinal = s.ordinal) AS chunk_count "
        "FROM report_sections s WHERE s.report_id = ? ORDER BY s.ordinal",
        (report_id,),
    )
    return [dict(r) for r in rows]


def chunks(report_id: int, content_type: str | None = None, section: int | None = None) -> list[dict]:
    sql = "SELECT * FROM report_chunks WHERE report_id = ?"
    params: list = [report_id]
    if content_type:
        sql += " AND content_type = ?"
        params.append(content_type)
    if section is not None:
        sql += " AND section_ordinal = ?"
        params.append(section)
    return [dict(r) for r in db.query(sql + " ORDER BY ordinal", params)]
