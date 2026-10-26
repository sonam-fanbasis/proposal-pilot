"""Report Lab HTTP API: upload, inspect, ask, evaluate.

Mounted under /api/reports by `main.py`. Kept as its own router because the
report pipeline shares nothing with the proposal library's request shapes --
no metadata filters, no draft mode -- and gains things the library does not
need: a section map, typed chunks and page images for visual grounding.
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from . import llm, report_eval, report_store
from .config import get_settings
from .report_generation import citation_dicts, finalize, generate, low_evidence_message, stream
from .report_models import embedder_name, reranker_name
from .report_retrieval import ReportRetrieval, retrieve

router = APIRouter(prefix="/api/reports", tags=["reports"])


class ReportAsk(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    report_ids: list[int] = Field(min_length=1)
    history: list[dict] = Field(default_factory=list)


class ReportEvalRequest(BaseModel):
    report_ids: list[int] = Field(default_factory=list)
    generate_answers: bool = True


def _require(report_id: int) -> dict:
    row = report_store.report_row(report_id)
    if not row:
        raise HTTPException(404, f"Report {report_id} not found.")
    return row


def _trace(result: ReportRetrieval) -> dict:
    """Everything the UI needs to show *how* the evidence was chosen."""
    return {
        "standalone_question": result.standalone,
        "route": {"kind": result.route.kind, "reason": result.route.reason, "sections": result.route.labels},
        "quantitative": result.quantitative,
        "candidates": result.candidates,
        "context_chars": result.context_chars,
        "timings": result.timings,
        "hits": [h.to_trace() for h in result.hits],
        "expanded_neighbours": sum(len(h.expanded_with) for h in result.hits),
        "embedder": embedder_name(),
        "reranker": reranker_name(),
        "llm": llm.provider_name(),
    }


# --- documents --------------------------------------------------------------

@router.post("")
async def upload(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    limit = get_settings().max_upload_mb * 1024 * 1024 * 2  # reports are allowed 2x
    if len(raw) > limit:
        raise HTTPException(413, f"File exceeds {limit // (1024 * 1024)} MB.")
    try:
        report_id, path = report_store.register(raw, file.filename or "report.pdf")
    except report_store.ReportError as exc:
        raise HTTPException(400, str(exc)) from exc
    report_store.ingest_async(report_id, path)
    return {"report_id": report_id, "filename": file.filename, "status": "pending"}


@router.get("")
def list_reports() -> list[dict]:
    return report_store.list_reports()


@router.get("/{report_id}")
def get_report(report_id: int) -> dict:
    _require(report_id)
    return next(r for r in report_store.list_reports() if r["id"] == report_id)


@router.delete("/{report_id}")
def delete_report(report_id: int) -> dict:
    _require(report_id)
    report_store.delete(report_id)
    return {"deleted": report_id}


@router.get("/{report_id}/sections")
def report_sections(report_id: int) -> list[dict]:
    _require(report_id)
    return report_store.sections(report_id)


@router.get("/{report_id}/chunks")
def report_chunks(
    report_id: int,
    type: str | None = Query(None, pattern="^(text|table|figure|footnote|note)$"),
    section: int | None = None,
) -> list[dict]:
    _require(report_id)
    return report_store.chunks(report_id, type, section)


@router.get("/{report_id}/samples")
def report_samples(report_id: int) -> dict:
    row = _require(report_id)
    suggestions = {
        "visual": ["What does the main chart on page 2 show?", "Which footnote qualifies the energy figure?"],
        "long_structured": ["Summarise the executive summary.", "Which chapter discusses fiscal policy?"],
        "narrative": ["What were the headline results this year?", "Who wrote the opening message?"],
    }.get(row["shape"], [])
    return {"benchmark": report_eval.samples_for(row), "suggested": suggestions}


@router.get("/{report_id}/pages/{page}.png")
def page_image(report_id: int, page: int, zoom: float = Query(0.9, ge=0.3, le=2.0)) -> Response:
    """Render a page so the UI can show the evidence where it actually sits."""
    import pymupdf

    row = _require(report_id)
    folder = get_settings().upload_dir / "reports"
    matches = list(folder.glob(f"*_{row['filename']}"))
    if not matches:
        raise HTTPException(404, "Original file is no longer on disk.")
    with pymupdf.open(str(matches[0])) as doc:
        if not 1 <= page <= doc.page_count:
            raise HTTPException(404, f"Page {page} is outside 1-{doc.page_count}.")
        png = doc[page - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
    return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


# --- ask --------------------------------------------------------------------

@router.post("/ask")
def ask(request: ReportAsk) -> dict:
    started = time.perf_counter()
    result = retrieve(request.question, request.report_ids, request.history)
    answer = generate(result)
    return {**answer, "confidence": result.confidence, "trace": _trace(result),
            "total_ms": int((time.perf_counter() - started) * 1000)}


@router.post("/ask/stream")
def ask_stream(request: ReportAsk) -> StreamingResponse:
    """SSE: `meta` (trace + all evidence) -> `token`s -> `done` (verified answer)."""

    def event(name: str, payload: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    def run():
        started = time.perf_counter()
        yield event("stage", {"stage": "retrieving"})
        result = retrieve(request.question, request.report_ids, request.history)
        yield event("meta", {
            "confidence": result.confidence,
            "evidence": citation_dicts(result.hits, set(range(1, len(result.hits) + 1))),
            "trace": _trace(result),
        })

        if low_evidence_message(result) or llm.is_offline():
            answer = generate(result)
            yield event("token", {"text": answer["answer"]})
            yield event("done", {**answer, "final_text": answer["answer"],
                                 "total_ms": int((time.perf_counter() - started) * 1000)})
            return

        yield event("stage", {"stage": "generating"})
        pieces: list[str] = []
        try:
            for piece in stream(result):
                pieces.append(piece)
                yield event("token", {"text": piece})
        except Exception as exc:
            if not pieces:
                answer = generate(result)  # extractive fallback with the real reason
                yield event("done", {**answer, "final_text": answer["answer"],
                                     "total_ms": int((time.perf_counter() - started) * 1000)})
                return
            yield event("error", {"message": str(exc)[:300]})
        answer = finalize("".join(pieces), result)
        yield event("done", {**answer, "final_text": answer["answer"],
                             "total_ms": int((time.perf_counter() - started) * 1000)})

    return StreamingResponse(run(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --- evaluation -------------------------------------------------------------

@router.post("/eval")
def run_eval(request: ReportEvalRequest = Body(default=ReportEvalRequest())) -> dict:
    return report_eval.run(request.report_ids or None, request.generate_answers)
