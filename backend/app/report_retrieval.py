"""Retrieval for reports: route -> search -> fuse -> rerank -> build context.

    question (+ history)
      -> resolve_followup  : standalone question (shared with the library path)
      -> route             : which chapter / box / region / summary is it about?
      -> dense + BM25      : inside the routed sections AND across the report
      -> RRF fusion        : routed lists weighted above global ones
      -> cross-encoder     : reads question and passage together
      -> type guarantee    : a "how much" question gets a table/figure in context
      -> context builder   : neighbour expansion, overlap removal, linked
                             footnotes, hard character budget

Routing is what makes a 200-page report answerable. Every regional section of
an economic outlook says "growth is projected to slow to N percent"; to a
bi-encoder these passages are near-duplicates, so a plain top-k over the whole
document returns a mix of regions and the model blends them. When the question
names a section -- "Chapter 3", "the executive summary", "South Asia" -- the
search is scoped to that section's subtree first, so the candidates are right
before ranking even starts.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

from . import db, report_store
from .config import get_settings
from .report_models import embed_query, rerank
from .retrieval import _STOPWORDS, _WORD, resolve_followup

_QUANT = re.compile(
    r"\b(how (much|many)|percent(age)?|share|total|number of|amount|rate|kg|tonnes?|co2|emissions?|"
    r"footprint|energy|consumption|cost|price|revenue|budget|growth|forecast|figure|table|chart)\b|%|\$",
    re.I,
)
_STRUCTURAL = (
    "executive summary", "foreword", "preface", "introduction", "conclusion", "conclusions",
    "statistical annex", "statistical appendix", "appendix", "annex", "endnotes", "definitions",
    "glossary", "overview", "methodology", "message from",
)


@dataclass
class ReportHit:
    chunk_id: int
    doc_id: int                 # report id; named for compatibility with generation helpers
    title: str
    filename: str
    section_path: str
    section_ordinal: int
    ordinal: int
    page_from: int
    page_to: int
    text: str
    content_type: str
    note_text: str = ""
    header: str = ""
    routed: bool = False
    dense_rank: int | None = None
    keyword_rank: int | None = None
    fused: float = 0.0
    rerank: float | None = None
    context: str = ""           # text after neighbour expansion
    expanded_with: list[int] = field(default_factory=list)

    # generation.verify_citations / _citations_from_hits read these
    client: str = ""
    industry: str = ""
    year: int = 0
    doc_type: str = "report"

    def to_trace(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "section": self.section_path, "type": self.content_type,
            "pages": [self.page_from, self.page_to], "routed": self.routed,
            "dense_rank": self.dense_rank, "keyword_rank": self.keyword_rank,
            "rerank": None if self.rerank is None else round(self.rerank, 3),
        }


@dataclass
class Route:
    kind: str = "none"          # explicit | title | none
    reason: str = ""
    sections: dict[int, list[int]] = field(default_factory=dict)   # report -> section ordinals (subtree)
    labels: list[str] = field(default_factory=list)


@dataclass
class ReportRetrieval:
    question: str
    standalone: str
    hits: list[ReportHit]
    route: Route
    confidence: float
    candidates: int
    timings: dict[str, int]
    quantitative: bool
    context_chars: int


# --- caches -----------------------------------------------------------------

_bm25: dict[int, tuple[float, object, list[int]]] = {}
_section_cache: dict[int, tuple[float, list[dict]]] = {}


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]


def _bm25_for(report_id: int):
    from rank_bm25 import BM25Okapi

    version = report_store.version(report_id)
    cached = _bm25.get(report_id)
    if cached and cached[0] == version:
        return cached[1], cached[2]
    rows = db.query(
        "SELECT id, header, text, note_text FROM report_chunks WHERE report_id = ? ORDER BY ordinal", (report_id,)
    )
    if not rows:
        return None, []
    index = BM25Okapi([_tokens(f"{r['header']} {r['text']} {r['note_text']}") for r in rows])
    ids = [int(r["id"]) for r in rows]
    _bm25[report_id] = (version, index, ids)
    return index, ids


def _sections_for(report_id: int) -> list[dict]:
    version = report_store.version(report_id)
    cached = _section_cache.get(report_id)
    if cached and cached[0] == version:
        return cached[1]
    rows = [dict(r) for r in db.query(
        "SELECT ordinal, parent, level, title, path, page_from, page_to FROM report_sections "
        "WHERE report_id = ? ORDER BY ordinal", (report_id,)
    )]
    _section_cache[report_id] = (version, rows)
    return rows


def _subtree(sections: list[dict], root: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for s in sections:
        if s["parent"] is not None:
            children.setdefault(int(s["parent"]), []).append(int(s["ordinal"]))
    out, stack = [], [root]
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(children.get(node, []))
    return out


# --- routing ----------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9. ]+", " ", text.lower()).strip()


def route(question: str, report_ids: list[int]) -> Route:
    """Map a question onto the report's section tree.

    Two tiers:

    * **explicit** references -- "Chapter 3", "Box 1.1", "the executive
      summary", "page 42". The user told us where to look; the search is
      hard-scoped to that subtree.
    * **title** matches -- every content word of a section title appears in
      the question ("growth outlook for *South Asia*"). Titles made of a
      single generic word ("Outlook", "Use") only count when they are rare in
      the document, otherwise every regional "Outlook" subsection would match.
    """
    q = _norm(question)
    q_words = set(_tokens(question))
    result = Route()
    explicit: dict[int, set[int]] = {}
    titled: dict[int, set[int]] = {}
    explicit_labels: list[str] = []
    title_labels: list[str] = []

    chapter = re.search(r"\b(?:chapter|ch\.?)\s*(\d+)\b", q)
    box = re.search(r"\bbox\s*([a-z]?\d+(?:\.\d+)*)", q)
    structural = [p for p in _STRUCTURAL if p in q]

    for report_id in report_ids:
        sections = _sections_for(report_id)
        title_freq: dict[str, int] = {}
        for s in sections:
            for w in set(_tokens(s["title"])):
                title_freq[w] = title_freq.get(w, 0) + 1

        for s in sections:
            if not s["ordinal"]:
                continue
            title = _norm(s["title"])
            if chapter and re.match(rf"^(ch\.?|chapter)\s*{chapter.group(1)}\b", title):
                explicit.setdefault(report_id, set()).add(s["ordinal"]); explicit_labels.append(s["title"])
            elif box and re.match(rf"^box\s*{re.escape(box.group(1))}\b", title):
                explicit.setdefault(report_id, set()).add(s["ordinal"]); explicit_labels.append(s["title"])
            elif structural and any(title.startswith(p) or title == p for p in structural) and s["level"] <= 2:
                explicit.setdefault(report_id, set()).add(s["ordinal"]); explicit_labels.append(s["title"])
            else:
                words = set(_tokens(s["title"]))
                if not words or not words <= q_words:
                    continue
                distinctive = len(words) >= 2 or any(len(w) >= 5 and title_freq.get(w, 0) <= 2 for w in words)
                if distinctive:
                    titled.setdefault(report_id, set()).add(s["ordinal"]); title_labels.append(s["title"])

    page = re.search(r"\bpage\s*(\d{1,4})\b", q)
    chosen = explicit or titled
    if chosen:
        result.kind = "explicit" if explicit else "title"
        result.reason = (
            "question names a chapter/box/summary" if explicit else "question names a section title"
        )
        for report_id, roots in chosen.items():
            sections = _sections_for(report_id)
            ordinals: set[int] = set()
            for root in roots:
                ordinals.update(_subtree(sections, root))
            result.sections[report_id] = sorted(ordinals)
        result.labels = list(dict.fromkeys(explicit_labels if explicit else title_labels))[:6]
    elif page:
        number = int(page.group(1))
        result.kind = "explicit"
        result.reason = f"question names page {number}"
        for report_id in report_ids:
            rows = db.query(
                "SELECT DISTINCT section_ordinal FROM report_chunks WHERE report_id = ? AND page_from <= ? AND page_to >= ?",
                (report_id, number, number),
            )
            if rows:
                result.sections[report_id] = [int(r["section_ordinal"]) for r in rows]
        result.labels = [f"page {number}"]
    return result


# --- search -----------------------------------------------------------------

def _rrf(lists: list[tuple[list[tuple[int, float]], float, str]], k: int) -> dict[int, dict]:
    fused: dict[int, dict] = {}
    for hits, weight, kind in lists:
        for rank, (chunk_id, _score) in enumerate(hits, start=1):
            entry = fused.setdefault(chunk_id, {"score": 0.0, "dense_rank": None, "keyword_rank": None, "routed": False})
            entry["score"] += weight / (k + rank)
            key = "dense_rank" if kind.startswith("dense") else "keyword_rank"
            if entry[key] is None or rank < entry[key]:
                entry[key] = rank
            if kind.endswith("routed"):
                entry["routed"] = True
    return fused


def _load(ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    rows = db.query(
        f"""SELECT c.*, r.title AS report_title, r.filename FROM report_chunks c
            JOIN reports r ON r.id = c.report_id WHERE c.id IN ({','.join('?' for _ in ids)})""",
        ids,
    )
    return {int(r["id"]): dict(r) for r in rows}


def _hit(row: dict, meta: dict) -> ReportHit:
    return ReportHit(
        chunk_id=int(row["id"]), doc_id=int(row["report_id"]), title=row["report_title"],
        filename=row["filename"], section_path=row["section_path"], section_ordinal=int(row["section_ordinal"]),
        ordinal=int(row["ordinal"]), page_from=int(row["page_from"]), page_to=int(row["page_to"]),
        text=row["text"], content_type=row["content_type"], note_text=row["note_text"] or "",
        header=row["header"], routed=meta.get("routed", False), dense_rank=meta.get("dense_rank"),
        keyword_rank=meta.get("keyword_rank"), fused=meta.get("score", 0.0),
    )


def _strip_overlap(previous: str, following: str) -> str:
    """Drop the carried-over tail that the chunker prepended to `following`."""
    body = following.lstrip("…").lstrip()
    for size in range(min(len(previous), len(body), 260), 24, -1):
        if previous.endswith(body[:size]):
            return body[size:].lstrip()
    return body


def _expand(hits: list[ReportHit], budget: int) -> int:
    """Give each text hit its neighbours from the same section, within budget.

    A retrieved chunk is chosen for being *relevant*, not for being
    *complete*: the sentence that answers the question often continues into
    the next chunk, or depends on the one before ("This reflects..."). Pulling
    the adjacent chunks of the same section restores that context without
    widening the search.
    """
    used = sum(len(h.text) + len(h.note_text) for h in hits)
    for hit in hits:
        hit.context = hit.text
    taken = {(h.doc_id, h.ordinal) for h in hits}
    for hit in hits:
        if hit.content_type not in {"text", "note"}:
            continue
        rows = db.query(
            "SELECT id, ordinal, text FROM report_chunks WHERE report_id = ? AND section_ordinal = ? "
            "AND content_type = ? AND ordinal IN (?, ?) ORDER BY ordinal",
            (hit.doc_id, hit.section_ordinal, hit.content_type, hit.ordinal - 1, hit.ordinal + 1),
        )
        for row in rows:
            key = (hit.doc_id, int(row["ordinal"]))
            if key in taken or used + len(row["text"]) > budget:
                continue
            taken.add(key)
            used += len(row["text"])
            hit.expanded_with.append(int(row["id"]))
            if int(row["ordinal"]) < hit.ordinal:
                hit.context = row["text"] + "\n" + _strip_overlap(row["text"], hit.context)
            else:
                hit.context = hit.context + "\n" + _strip_overlap(hit.context, row["text"])
    return used


def _confidence(top: float | None, fused_top: float) -> float:
    if top is None:
        return round(min(1.0, fused_top * 30), 3)
    return round(1.0 / (1.0 + math.exp(-(top + 3.0) / 2.0)), 3)


def retrieve(question: str, report_ids: list[int], history: list[dict] | None = None) -> ReportRetrieval:
    settings = get_settings()
    timings: dict[str, int] = {}
    started = time.perf_counter()

    standalone = resolve_followup(question, history or [])
    timings["rewrite_ms"] = int((time.perf_counter() - started) * 1000)

    mark = time.perf_counter()
    plan = route(standalone, report_ids)
    query_vector = embed_query(standalone)
    terms = _tokens(standalone)
    k = settings.report_candidates

    lists: list[tuple[list[tuple[int, float]], float, str]] = []
    for report_id in report_ids:
        scoped = plan.sections.get(report_id)
        # Global lists are always present: routing narrows, it never blinds.
        # When the route is explicit they are down-weighted heavily.
        global_weight = 0.35 if (scoped and plan.kind == "explicit") else 1.0
        lists.append((report_store.vector_search(query_vector, report_id, k), global_weight, "dense"))
        index, ids = _bm25_for(report_id)
        scores = index.get_scores(terms) if index is not None and terms else []
        keyword = sorted(((cid, float(s)) for cid, s in zip(ids, scores) if s > 0), key=lambda x: -x[1])
        lists.append((keyword[:k], global_weight, "keyword"))
        if scoped:
            lists.append((report_store.vector_search(query_vector, report_id, k, scoped), 2.0, "dense_routed"))
            allowed = {int(r["id"]) for r in db.query(
                f"SELECT id FROM report_chunks WHERE report_id = ? AND section_ordinal IN ({','.join('?' for _ in scoped)})",
                [report_id, *scoped],
            )}
            lists.append(([p for p in keyword if p[0] in allowed][:k], 2.0, "keyword_routed"))
            # The opening of a routed chapter is usually its abstract; make
            # sure it is a candidate for "what does chapter N say" questions.
            lead = db.query(
                f"SELECT id FROM report_chunks WHERE report_id = ? AND content_type = 'text' AND section_ordinal IN "
                f"({','.join('?' for _ in scoped)}) ORDER BY ordinal LIMIT 3",
                [report_id, *scoped],
            )
            lists.append(([(int(r["id"]), 0.0) for r in lead], 1.0, "dense_routed"))
    fused = _rrf(lists, settings.rrf_k)
    timings["search_ms"] = int((time.perf_counter() - mark) * 1000)

    mark = time.perf_counter()
    # The cross-encoder is the slow step (~40ms a passage on a laptop CPU), so
    # it sees the top of the fused list, trimmed to the part it can read.
    pool_ids = [cid for cid, _ in sorted(fused.items(), key=lambda kv: -kv[1]["score"])[: settings.report_rerank_pool]]
    rows = _load(pool_ids)
    pool = [_hit(rows[cid], fused[cid]) for cid in pool_ids if cid in rows]
    scored = rerank(standalone, [f"{h.section_path}\n{h.text}\n{h.note_text}"[:1200] for h in pool])
    if scored is not None:
        for hit, score in zip(pool, scored):
            # A routed chunk gets a fixed bonus: the user named this section.
            hit.rerank = score + (1.5 if hit.routed and plan.kind == "explicit" else 0.75 if hit.routed else 0.0)
        pool.sort(key=lambda h: -(h.rerank or 0.0))
    timings["rerank_ms"] = int((time.perf_counter() - mark) * 1000)

    final_k = settings.report_final_k
    routed = [h for h in pool if h.routed]
    if plan.kind == "explicit" and routed and scored is not None:
        # The user named the section, so the answer must come from it.
        # In-scope chunks within reach of the best one are kept; an
        # out-of-scope chunk gets in only by beating the best in-scope chunk,
        # or as filler when the section is too small to fill three slots.
        best_routed = routed[0].rerank or 0.0
        keep = [h for h in routed if (h.rerank or -99) > best_routed - 8][:final_k]
        outside = [h for h in pool if not h.routed]
        extra = [h for h in outside if (h.rerank or -99) > best_routed]
        if len(keep) + len(extra) < 3:
            extra += [h for h in outside if h not in extra][: 3 - len(keep) - len(extra)]
        pool = sorted(keep + extra, key=lambda h: -(h.rerank or 0.0))
    top = pool[:final_k]
    quantitative = bool(_QUANT.search(standalone))
    if quantitative and top and not any(h.content_type in {"table", "figure"} for h in top):
        # Numbers in visual reports live in tables and chart transcriptions. If
        # one is a plausible candidate, it displaces the weakest text hit.
        best = top[0].rerank or 0.0
        visual = [h for h in pool[final_k:] if h.content_type in {"table", "figure"} and (h.rerank or -99) > best - 7]
        if visual:
            top[-1] = visual[0]

    mark = time.perf_counter()
    used = _expand(top, settings.report_context_chars)
    timings["context_ms"] = int((time.perf_counter() - mark) * 1000)
    timings["total_ms"] = int((time.perf_counter() - started) * 1000)

    best = max((h.rerank for h in top if h.rerank is not None), default=None)
    return ReportRetrieval(
        question=question, standalone=standalone, hits=top, route=plan,
        confidence=_confidence(best, top[0].fused if top else 0.0), candidates=len(fused),
        timings=timings, quantitative=quantitative, context_chars=used,
    )
