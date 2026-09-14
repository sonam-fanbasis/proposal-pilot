"""The write path: file -> parse -> metadata -> chunk -> embed -> index.

Ingestion runs on a background thread and the document's `status` column is
advanced at each stage (`pending` -> `parsing` -> `embedding` -> `ready`),
so the UI can show real progress and a failure lands as `failed` with the
reason attached rather than as a silent gap in the corpus.

Ordering note: metadata extraction happens *before* chunking, because the
contextual header prepended to every chunk is built from that metadata. Chunk
first and every chunk would be missing the client, industry and tech tags that
make it retrievable.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import traceback
from pathlib import Path

from . import db, metadata as metadata_mod, vectorstore
from .chunking import chunk_document
from .config import get_settings
from .embeddings import DOCUMENT, embed
from .parsing import SUPPORTED_EXTENSIONS, parse


class IngestionError(RuntimeError):
    pass


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_status(doc_id: int, status: str, error: str | None = None) -> None:
    db.execute(
        "UPDATE documents SET status = ?, error = ? WHERE id = ?", (status, error, doc_id)
    )


def register(path: Path, original_name: str) -> int:
    """Record the document and return its id. Raises on duplicates."""
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    digest = file_digest(path)
    existing = db.query_one("SELECT id, title FROM documents WHERE sha256 = ?", (digest,))
    if existing:
        raise IngestionError(
            f"This file is already in the library as '{existing['title']}' (document {existing['id']})."
        )

    cursor = db.execute(
        "INSERT INTO documents(filename, title, sha256, byte_size, status) VALUES (?, ?, ?, ?, 'pending')",
        (original_name, Path(original_name).stem.replace("_", " ").title(), digest, path.stat().st_size),
    )
    return int(cursor.lastrowid)


def process(doc_id: int, path: Path) -> dict:
    """Run the full pipeline for one document. Returns a small summary."""
    try:
        _set_status(doc_id, "parsing")
        parsed = parse(path)
        if not parsed.blocks:
            raise IngestionError(
                "No extractable text found. If this is a scanned PDF it needs OCR first."
            )

        meta = metadata_mod.extract(parsed.full_text, path.name)
        tech = metadata_mod.normalise_tech(meta.get("tech_stack") or [])

        db.execute(
            """UPDATE documents SET title = ?, client = ?, industry = ?, doc_type = ?,
                   engagement_model = ?, year = ?, value_usd = ?, duration_months = ?,
                   tech_stack = ?, summary = ?, meta_json = ?, page_count = ?
               WHERE id = ?""",
            (
                meta.get("title") or path.stem,
                meta.get("client") or "",
                meta.get("industry") or "other",
                meta.get("doc_type") or "other",
                meta.get("engagement_model") or "other",
                int(meta.get("year") or 0),
                int(meta.get("value_usd") or 0),
                int(meta.get("duration_months") or 0),
                json.dumps(tech),
                meta.get("summary") or "",
                json.dumps(meta),
                parsed.page_count,
                doc_id,
            ),
        )
        db.execute("DELETE FROM doc_tech WHERE doc_id = ?", (doc_id,))
        for name in tech:
            db.execute(
                "INSERT OR IGNORE INTO doc_tech(doc_id, tech) VALUES (?, ?)", (doc_id, name)
            )

        doc_meta = dict(meta)
        doc_meta["tech_stack"] = tech
        doc_meta["filename"] = path.name
        chunks = chunk_document(parsed.blocks, doc_meta)
        if not chunks:
            raise IngestionError("Document produced no usable chunks after parsing.")

        _set_status(doc_id, "embedding")
        db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        vectorstore.delete_document(doc_id)

        chunk_ids: list[int] = []
        for chunk in chunks:
            cursor = db.execute(
                """INSERT INTO chunks(doc_id, ordinal, section_path, page_from, page_to,
                                      text, context_header, char_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc_id, chunk.ordinal, chunk.section_path, chunk.page_from,
                    chunk.page_to, chunk.text, chunk.context_header, len(chunk.text),
                ),
            )
            chunk_ids.append(int(cursor.lastrowid))

        vectors = embed([c.embedding_text for c in chunks], DOCUMENT)
        vectorstore.upsert(chunk_ids, [doc_id] * len(chunk_ids), vectors)

        _set_status(doc_id, "ready")
        return {
            "document_id": doc_id,
            "chunks": len(chunks),
            "pages": parsed.page_count,
            "metadata": meta,
        }
    except Exception as exc:
        _set_status(doc_id, "failed", f"{type(exc).__name__}: {exc}"[:500])
        traceback.print_exc()
        raise


def save_upload(file_bytes: bytes, original_name: str) -> Path:
    settings = get_settings()
    safe = Path(original_name).name.replace("/", "_")
    target = settings.upload_dir / f"{hashlib.sha256(file_bytes).hexdigest()[:12]}_{safe}"
    target.write_bytes(file_bytes)
    return target


def ingest_async(doc_id: int, path: Path) -> threading.Thread:
    """Embedding a long document takes seconds to minutes; never block the request."""

    def run() -> None:
        try:
            process(doc_id, path)
        except Exception:
            pass  # status/error already persisted by process()

    thread = threading.Thread(target=run, daemon=True, name=f"ingest-{doc_id}")
    thread.start()
    return thread


def ingest_path(path: Path) -> dict:
    """Synchronous ingest of a file already on disk. Used by the seed script."""
    settings = get_settings()
    target = settings.upload_dir / path.name
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
    doc_id = register(target, path.name)
    return process(doc_id, target)


def delete_document(doc_id: int) -> None:
    vectorstore.delete_document(doc_id)
    db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
