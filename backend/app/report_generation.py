"""Grounded answer generation over report evidence.

Reuses the library's grounding machinery (`verify_citations` strips citations
to evidence that was never retrieved and counts uncited factual sentences),
with a prompt written for the failure modes of long and visual reports:

* **scope bleed** -- borrowing a figure from the neighbouring region or chapter;
* **lost qualifiers** -- quoting "58 percent less energy" without the footnote
  that says which configuration and which standard it was measured against;
* **visual guesswork** -- inventing a chart value that was never transcribed.
"""

from __future__ import annotations

from collections.abc import Iterator

from . import llm
from .generation import _extractive, verify_citations
from .report_retrieval import ReportHit, ReportRetrieval

REPORT_SYSTEM = """You answer questions about a specific published report, using ONLY numbered \
excerpts retrieved from it. Each excerpt states its section path, page and type: body text, \
table (rebuilt as "row | column: value" records), chart/figure (transcribed from the page image), \
figure note, or footnote.

Rules:
- Use only the excerpts. If they do not contain the answer, say so in one sentence and name what is missing.
- End every sentence that states a fact with its citation, e.g. [2]. Cite several as [1][3].
- Copy numbers, units, percentages, years and names exactly. Never round, estimate or combine figures \
  unless the question asks for arithmetic, and then show the inputs.
- Respect scope. If the question names a region, chapter, section, product configuration or year, use \
  only excerpts about that scope. Never transfer a figure from one region, chapter or configuration to another.
- Prefer the table or chart value when a number appears in several places; mention that it comes from a table/chart.
- If a footnote qualifies a number (configuration, method, baseline, date), state the qualification briefly.
- When the excerpts state an overall assessment (e.g. "risks are tilted to the downside", "growth is \
  projected to slow"), lead with it before listing specifics.
- Format: lead with a direct 1–2 sentence answer. Then, only if there are several distinct facts, add short \
  bullet points. Keep it under 220 words. No preamble."""


def build_evidence(hits: list[ReportHit]) -> str:
    kinds = {
        "text": "body text", "table": "table", "figure": "chart/figure (transcribed)",
        "note": "figure note/caption", "footnote": "footnote",
    }
    blocks = []
    for index, hit in enumerate(hits, start=1):
        pages = f"p.{hit.page_from}" if hit.page_from == hit.page_to else f"pp.{hit.page_from}-{hit.page_to}"
        body = hit.context or hit.text
        if hit.note_text:
            body += f"\n    Linked footnotes: {hit.note_text}"
        blocks.append(
            f"[{index}] Section: {hit.section_path or 'front matter'} | {pages} | {kinds.get(hit.content_type, hit.content_type)}\n"
            f"{body}"
        )
    return "\n\n".join(blocks)


def build_prompt(result: ReportRetrieval) -> str:
    title = result.hits[0].title if result.hits else "the report"
    scope = ""
    if result.route.labels:
        scope = f"The question was routed to: {'; '.join(result.route.labels)}.\n"
    return (
        f"REPORT: {title}\n{scope}\nEXCERPTS\n{build_evidence(result.hits)}\n\n"
        f"QUESTION: {result.standalone}\n\nAnswer using only the excerpts above."
    )


def citation_dicts(hits: list[ReportHit], used: set[int]) -> list[dict]:
    out = []
    for index, hit in enumerate(hits, start=1):
        if index not in used:
            continue
        out.append({
            "index": index, "chunk_id": hit.chunk_id, "doc_id": hit.doc_id, "title": hit.title,
            "filename": hit.filename, "section_path": hit.section_path, "page_from": hit.page_from,
            "page_to": hit.page_to, "content_type": hit.content_type, "snippet": hit.text[:600],
            "note_text": hit.note_text[:400], "routed": hit.routed,
            "rerank": None if hit.rerank is None else round(hit.rerank, 2),
        })
    return out


def low_evidence_message(result: ReportRetrieval) -> str | None:
    if not result.hits:
        return "Nothing in the selected report matches this question."
    if result.confidence < 0.12:
        return (
            "The report does not appear to cover this. The closest passages scored too low "
            f"to answer from (confidence {result.confidence:.2f}): "
            + "; ".join(f"{h.section_path} (p.{h.page_from})" for h in result.hits[:3])
        )
    return None


def finalize(raw: str, result: ReportRetrieval) -> dict:
    text, citations, ungrounded, invalid = verify_citations(raw, result.hits)
    used = {c.index for c in citations}
    return {
        "answer": text,
        "citations": citation_dicts(result.hits, used),
        "ungrounded_sentences": ungrounded,
        "invalid_citations": invalid,
    }


def generate(result: ReportRetrieval) -> dict:
    message = low_evidence_message(result)
    if message:
        return {"answer": message, "citations": citation_dicts(result.hits, set(range(1, 4))),
                "ungrounded_sentences": 0, "invalid_citations": []}
    if llm.is_offline():
        return _offline(result, "offline")
    try:
        raw = llm.complete(build_prompt(result), system=REPORT_SYSTEM, temperature=0.1, max_output_tokens=1200)
    except Exception as exc:
        reason = "rate_limit" if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc) else str(exc)[:80]
        return _offline(result, reason)
    return finalize(raw, result)


def stream(result: ReportRetrieval) -> Iterator[str]:
    yield from llm.stream(build_prompt(result), system=REPORT_SYSTEM, temperature=0.1, max_output_tokens=1200)


def _offline(result: ReportRetrieval, reason: str) -> dict:
    answer = _extractive(result.standalone, result.hits, "answer", reason)
    used = {c.index for c in answer.citations}
    return {"answer": answer.text, "citations": citation_dicts(result.hits, used),
            "ungrounded_sentences": answer.ungrounded_sentences, "invalid_citations": answer.invalid_citations}
