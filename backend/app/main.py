"""ProposalPilot HTTP API.

Routes are grouped as: documents (the write path), ask (the read path),
conversations, facets/stats (observability), and eval (quality).
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import db, eval as eval_mod, ingestion, llm, vectorstore
from .config import get_settings
from .embeddings import active_backend
from .generation import generate, generate_stream, verify_citations
from .parsing import SUPPORTED_EXTENSIONS
from .retrieval import retrieve
from .schemas import (
    AskRequest, AskResponse, ConversationOut, DocumentOut, EvalRunRequest,
    FacetsOut, MessageOut, StatsOut, UploadResponse,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create the schema and seed evaluation cases before serving traffic."""
    db.init_db()
    if not db.query_one("SELECT 1 FROM eval_cases LIMIT 1"):
        try:
            eval_mod.seed_cases()
        except Exception:
            pass
    yield


app = FastAPI(
    title="ProposalPilot",
    description="RAG over a consulting firm's proposal and case-study library.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- helpers ----------------------------------------------------------------

def _document_row_to_out(row) -> DocumentOut:
    """Map a document row to its API shape.

    A document that is still being ingested has NULL metadata columns -- it has
    been registered but not yet catalogued -- so every nullable column is
    coerced to its zero value here. The API never emits `null` for these
    fields, which means the UI can render a pending document without
    null-guarding every badge.
    """
    data = dict(row)
    try:
        data["tech_stack"] = json.loads(data.get("tech_stack") or "[]")
    except json.JSONDecodeError:
        data["tech_stack"] = []

    for key in ("client", "industry", "doc_type", "engagement_model", "summary", "created_at"):
        if data.get(key) is None:
            data[key] = ""
    for key in ("year", "value_usd", "duration_months", "page_count", "byte_size", "chunk_count"):
        if data.get(key) is None:
            data[key] = 0

    data.pop("meta_json", None)
    data.pop("sha256", None)
    return DocumentOut(**{k: v for k, v in data.items() if k in DocumentOut.model_fields})


def _history(conversation_id: int | None) -> list[dict]:
    if not conversation_id:
        return []
    rows = db.query(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT 6",
        (conversation_id,),
    )
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def _ensure_conversation(conversation_id: int | None, question: str) -> int:
    if conversation_id:
        if db.query_one("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)):
            return conversation_id
        raise HTTPException(404, f"Conversation {conversation_id} not found.")
    title = question[:70] + ("…" if len(question) > 70 else "")
    return int(db.execute("INSERT INTO conversations(title) VALUES (?)", (title,)).lastrowid)


def _retrieval_debug(result, standalone: str) -> dict:
    return {
        "search_query": result.plan.search_query,
        "standalone_question": standalone,
        "intent": result.plan.intent,
        "planned_by": result.plan.planned_by,
        "filters": result.plan.filters.to_dict(),
        "filtered_document_count": result.filtered_doc_count,
        "filters_relaxed": result.relaxed,
        "candidates": result.candidates,
        "retrieval_ms": result.took_ms,
        "embedding_backend": active_backend(),
        "llm": llm.provider_name(),
    }


def _log_query(question: str, mode: str, result, total_ms: int) -> None:
    db.execute(
        """INSERT INTO query_log(query, mode, filters_json, provider, n_candidates,
                                 n_returned, confidence, retrieval_ms, total_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            question, mode, json.dumps(result.plan.filters.to_dict()), llm.provider_name(),
            result.candidates, len(result.hits), result.confidence, result.took_ms, total_ms,
        ),
    )


# --- documents --------------------------------------------------------------

@app.post("/api/documents", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Uploaded file is empty.")
    limit = settings.max_upload_mb * 1024 * 1024
    if len(raw) > limit:
        raise HTTPException(413, f"File exceeds the {settings.max_upload_mb} MB limit.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            415, f"Unsupported type '{suffix or 'unknown'}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    path = ingestion.save_upload(raw, file.filename or "upload")
    try:
        doc_id = ingestion.register(path, file.filename or path.name)
    except ingestion.IngestionError as exc:
        raise HTTPException(409, str(exc)) from exc

    ingestion.ingest_async(doc_id, path)
    return UploadResponse(
        document_id=doc_id,
        filename=file.filename or path.name,
        status="pending",
        message="Upload accepted. Parsing, cataloguing and embedding are running in the background.",
    )


@app.get("/api/documents", response_model=list[DocumentOut])
def list_documents() -> list[DocumentOut]:
    rows = db.query(
        """SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS chunk_count
           FROM documents d ORDER BY d.id DESC"""
    )
    return [_document_row_to_out(r) for r in rows]


@app.get("/api/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int) -> DocumentOut:
    row = db.query_one(
        """SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS chunk_count
           FROM documents d WHERE d.id = ?""",
        (doc_id,),
    )
    if not row:
        raise HTTPException(404, f"Document {doc_id} not found.")
    return _document_row_to_out(row)


@app.get("/api/documents/{doc_id}/chunks")
def get_document_chunks(doc_id: int, limit: int = Query(200, ge=1, le=1000)) -> list[dict]:
    """Exposed so the chunking strategy can be inspected rather than trusted."""
    rows = db.query(
        """SELECT id, ordinal, section_path, page_from, page_to, char_count, text, context_header
           FROM chunks WHERE doc_id = ? ORDER BY ordinal LIMIT ?""",
        (doc_id, limit),
    )
    return [dict(r) for r in rows]


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int) -> dict:
    if not db.query_one("SELECT 1 FROM documents WHERE id = ?", (doc_id,)):
        raise HTTPException(404, f"Document {doc_id} not found.")
    ingestion.delete_document(doc_id)
    return {"deleted": doc_id}


# --- ask --------------------------------------------------------------------

@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    started = time.perf_counter()
    conversation_id = _ensure_conversation(request.conversation_id, request.question)
    history = _history(request.conversation_id)

    db.execute(
        "INSERT INTO messages(conversation_id, role, content, mode) VALUES (?, 'user', ?, ?)",
        (conversation_id, request.question, request.mode),
    )

    result = retrieve(request.question, history, request.top_k, request.document_ids or None)

    answer = generate(request.question, result, request.mode, history)
    total_ms = int((time.perf_counter() - started) * 1000)

    citations = [c.to_dict() for c in answer.citations]
    cursor = db.execute(
        """INSERT INTO messages(conversation_id, role, content, mode, citations_json,
                                gaps_json, confidence, latency_ms)
           VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?)""",
        (
            conversation_id, answer.text, request.mode, json.dumps(citations),
            json.dumps(answer.gaps), result.confidence, total_ms,
        ),
    )
    _log_query(request.question, request.mode, result, total_ms)

    return AskResponse(
        answer=answer.text,
        mode=request.mode,
        citations=citations,
        gaps=answer.gaps,
        confidence=result.confidence,
        ungrounded_sentences=answer.ungrounded_sentences,
        invalid_citations=answer.invalid_citations,
        conversation_id=conversation_id,
        message_id=int(cursor.lastrowid),
        total_ms=total_ms,
        retrieval=_retrieval_debug(result, result.plan.original),
    )


@app.post("/api/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    """Server-sent events: `meta` (citations up front), then `token`s, then `done`.

    Citations are sent *before* the first token deliberately -- the sources
    panel fills in while the answer is still being written, so the user can
    start checking provenance immediately rather than waiting for the model to
    finish.
    """

    def event(name: str, payload: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    def stream():
        started = time.perf_counter()
        conversation_id = _ensure_conversation(request.conversation_id, request.question)
        history = _history(request.conversation_id)
        db.execute(
            "INSERT INTO messages(conversation_id, role, content, mode) VALUES (?, 'user', ?, ?)",
            (conversation_id, request.question, request.mode),
        )

        result = retrieve(request.question, history, request.top_k, request.document_ids or None)

        from .generation import _citations_from_hits

        yield event(
            "meta",
            {
                "conversation_id": conversation_id,
                "confidence": result.confidence,
                "citations": [c.to_dict() for c in _citations_from_hits(result.hits)],
                "retrieval": _retrieval_debug(result, result.plan.original),
            },
        )

        # No evidence, weak evidence, or no model: fall back to the same
        # non-streaming path so the guardrails apply identically.
        if not result.hits or result.confidence < settings.min_confidence or llm.is_offline():
            answer = generate(request.question, result, request.mode, history)
            yield event("token", {"text": answer.text})
            total_ms = int((time.perf_counter() - started) * 1000)
            cursor = db.execute(
                """INSERT INTO messages(conversation_id, role, content, mode, citations_json,
                                        gaps_json, confidence, latency_ms)
                   VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?)""",
                (
                    conversation_id, answer.text, request.mode,
                    json.dumps([c.to_dict() for c in answer.citations]),
                    json.dumps(answer.gaps), result.confidence, total_ms,
                ),
            )
            _log_query(request.question, request.mode, result, total_ms)
            yield event(
                "done",
                {
                    "message_id": int(cursor.lastrowid),
                    "citations": [c.to_dict() for c in answer.citations],
                    "gaps": answer.gaps,
                    "ungrounded_sentences": answer.ungrounded_sentences,
                    "invalid_citations": answer.invalid_citations,
                    "total_ms": total_ms,
                },
            )
            return

        pieces: list[str] = []
        try:
            for piece in generate_stream(request.question, result, request.mode, history):
                pieces.append(piece)
                yield event("token", {"text": piece})
        except Exception as exc:
            yield event("error", {"message": str(exc)[:300]})

        raw = "".join(pieces)
        from .generation import _split_gaps

        body, gaps = _split_gaps(raw)
        cleaned, citations, ungrounded, invalid = verify_citations(body, result.hits)
        total_ms = int((time.perf_counter() - started) * 1000)
        cursor = db.execute(
            """INSERT INTO messages(conversation_id, role, content, mode, citations_json,
                                    gaps_json, confidence, latency_ms)
               VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?)""",
            (
                conversation_id, cleaned, request.mode,
                json.dumps([c.to_dict() for c in citations]),
                json.dumps(gaps), result.confidence, total_ms,
            ),
        )
        _log_query(request.question, request.mode, result, total_ms)
        yield event(
            "done",
            {
                "message_id": int(cursor.lastrowid),
                # The verified text replaces what was streamed: tokens arrive
                # raw, but what gets stored and displayed has had invented
                # citations stripped.
                "final_text": cleaned,
                "citations": [c.to_dict() for c in citations],
                "gaps": gaps,
                "ungrounded_sentences": ungrounded,
                "invalid_citations": invalid,
                "total_ms": total_ms,
            },
        )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- conversations ----------------------------------------------------------

@app.get("/api/conversations", response_model=list[ConversationOut])
def list_conversations() -> list[ConversationOut]:
    rows = db.query(
        """SELECT c.id, c.title, c.created_at,
                  (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
           FROM conversations c ORDER BY c.id DESC LIMIT 100"""
    )
    return [ConversationOut(**dict(r)) for r in rows]


@app.get("/api/conversations/{conversation_id}/messages", response_model=list[MessageOut])
def get_messages(conversation_id: int) -> list[MessageOut]:
    rows = db.query(
        """SELECT id, role, content, mode, citations_json, gaps_json, confidence, created_at
           FROM messages WHERE conversation_id = ? ORDER BY id""",
        (conversation_id,),
    )
    out: list[MessageOut] = []
    for row in rows:
        out.append(
            MessageOut(
                id=row["id"], role=row["role"], content=row["content"], mode=row["mode"],
                citations=json.loads(row["citations_json"] or "[]"),
                gaps=json.loads(row["gaps_json"] or "[]"),
                confidence=row["confidence"], created_at=row["created_at"],
            )
        )
    return out


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: int) -> dict:
    db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    return {"deleted": conversation_id}


# --- facets and stats -------------------------------------------------------

@app.get("/api/facets", response_model=FacetsOut)
def facets() -> FacetsOut:
    def counts(column: str) -> list[dict]:
        rows = db.query(
            f"""SELECT {column} AS value, COUNT(*) AS count FROM documents
                WHERE status = 'ready' AND {column} IS NOT NULL AND {column} != ''
                GROUP BY {column} ORDER BY count DESC"""
        )
        return [{"value": r["value"], "count": r["count"]} for r in rows]

    tech_rows = db.query(
        """SELECT t.tech AS value, COUNT(*) AS count FROM doc_tech t
           JOIN documents d ON d.id = t.doc_id WHERE d.status = 'ready'
           GROUP BY t.tech ORDER BY count DESC, value"""
    )
    year_rows = db.query(
        "SELECT DISTINCT year FROM documents WHERE status = 'ready' AND year > 0 ORDER BY year DESC"
    )
    return FacetsOut(
        industries=counts("industry"),
        doc_types=counts("doc_type"),
        engagement_models=counts("engagement_model"),
        tech=[{"value": r["value"], "count": r["count"]} for r in tech_rows],
        years=[int(r["year"]) for r in year_rows],
    )


@app.get("/api/stats", response_model=StatsOut)
def stats() -> StatsOut:
    def scalar(sql: str, params: tuple = ()) -> float | None:
        row = db.query_one(sql, params)
        value = row[0] if row else None
        return float(value) if value is not None else None

    latencies = [r["retrieval_ms"] for r in db.query("SELECT retrieval_ms FROM query_log")]
    median = float(sorted(latencies)[len(latencies) // 2]) if latencies else None
    asked = int(scalar("SELECT COUNT(*) FROM query_log") or 0)
    low = int(scalar("SELECT COUNT(*) FROM query_log WHERE confidence < ?", (settings.min_confidence,)) or 0)

    return StatsOut(
        documents=int(scalar("SELECT COUNT(*) FROM documents") or 0),
        documents_ready=int(scalar("SELECT COUNT(*) FROM documents WHERE status = 'ready'") or 0),
        documents_failed=int(scalar("SELECT COUNT(*) FROM documents WHERE status = 'failed'") or 0),
        chunks=int(scalar("SELECT COUNT(*) FROM chunks") or 0),
        vectors=vectorstore.count(),
        conversations=int(scalar("SELECT COUNT(*) FROM conversations") or 0),
        questions_asked=asked,
        median_retrieval_ms=median,
        mean_confidence=round(scalar("SELECT AVG(confidence) FROM query_log") or 0, 3) if asked else None,
        low_confidence_rate=round(low / asked, 3) if asked else None,
        embedding_backend=active_backend(),
        llm=llm.provider_name(),
        vector_index="sqlite-vec (indexed ANN)" if db.VEC_AVAILABLE else "numpy brute-force fallback",
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "llm": llm.provider_name(), "embeddings": active_backend()}


# --- eval -------------------------------------------------------------------

@app.post("/api/eval/seed")
def eval_seed() -> dict:
    return {"seeded": eval_mod.seed_cases()}


@app.post("/api/eval/run")
def eval_run(request: EvalRunRequest = Body(default=EvalRunRequest())) -> dict:
    return eval_mod.run(k=request.k, generate_answers=request.generate_answers)


@app.get("/api/eval/runs")
def eval_runs(limit: int = Query(10, ge=1, le=50)) -> list[dict]:
    rows = db.query(
        "SELECT id, metrics_json, created_at FROM eval_runs ORDER BY id DESC LIMIT ?", (limit,)
    )
    return [
        {"id": r["id"], "created_at": r["created_at"], "metrics": json.loads(r["metrics_json"])}
        for r in rows
    ]


@app.get("/api/eval/runs/{run_id}")
def eval_run_detail(run_id: int) -> dict:
    row = db.query_one("SELECT * FROM eval_runs WHERE id = ?", (run_id,))
    if not row:
        raise HTTPException(404, f"Evaluation run {run_id} not found.")
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "metrics": json.loads(row["metrics_json"]),
        "details": json.loads(row["detail_json"]),
    }


# --- static frontend --------------------------------------------------------
# Mounted last so it never shadows an /api route. Present only after the
# frontend has been built; in development Vite serves it on port 5173 instead.

_FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str) -> FileResponse:
        candidate = _FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")
