"""Query understanding and hybrid retrieval.

The pipeline for one question:

    question (+ chat history)
        -> resolve_followup   : "and the pricing?" -> a standalone question
        -> plan_query         : split into structured FILTERS + a semantic query
        -> filters -> doc_ids : one indexed SQL pass over documents/doc_tech
        -> vector search      : metadata-filtered KNN inside those documents
        -> keyword search     : BM25 over the same filtered chunk set
        -> RRF fusion         : rank-based merge, no score normalisation needed
        -> rerank             : metadata-agreement and section-prior boosts
        -> confidence         : does this evidence justify answering at all?

Why filters *and* vectors, rather than vectors alone: "fixed-bid healthcare
projects under $200k" contains almost no semantic signal -- every proposal in
the corpus talks about scope and delivery in similar language. Embeddings
cannot represent "< $200,000". A WHERE clause can, exactly, so the structured
half of the question is answered structurally and only the rest is left to
similarity.

Why RRF rather than a weighted score blend: cosine similarity and BM25 live on
different, corpus-dependent scales, so any fixed `0.7*vec + 0.3*bm25` needs
retuning per corpus. Reciprocal Rank Fusion only reads the *ranks*, so it is
scale-free and needs no tuning.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field

from . import db, llm, vectorstore
from .config import get_settings
from .embeddings import QUERY, embed_one
from .metadata import DOC_TYPES, ENGAGEMENT_MODELS, INDUSTRIES, TECH_ALIASES, find_tech_in_text

# Includes the framing vocabulary of questions as well as ordinary function
# words. "Describe your QA process" and "have we ever done automotive work"
# are both mostly scaffolding: the words that decide the answer are "QA" and
# "automotive". Confidence scoring reads the absence of a query term from the
# corpus as evidence it cannot answer, so leaving framing verbs in would make
# every politely-phrased question look unanswerable -- "describe" appears in no
# proposal ever written.
_STOPWORDS = {
    # function words
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "we", "our", "us", "have", "has", "had", "do", "did", "does", "is",
    "are", "was", "were", "be", "been", "what", "which", "who", "whom", "how",
    "when", "where", "why", "any", "some", "that", "this", "it", "its", "as",
    "at", "by", "from", "can", "could", "would", "should", "you", "your",
    "there", "their", "them", "they", "than", "then", "into", "about", "also",
    "more", "most", "other", "such", "only", "same", "too", "very", "just",
    "not", "all", "been", "over", "under", "between", "during", "before",
    "after", "each", "both", "will", "may", "might", "must", "shall",
    # question framing
    "describe", "tell", "explain", "provide", "outline", "summarise",
    "summarize", "list", "show", "give", "detail", "details", "please",
    "need", "want", "looking", "know", "ever", "anything", "something",
    "like", "used", "using", "use", "get", "got", "make", "made", "take",
    # generic business filler that carries no discriminative signal here
    "work", "works", "worked", "working", "done", "doing", "project",
    "projects", "engagement", "engagements", "client", "clients", "company",
    "experience", "capability", "capabilities", "approach", "thing", "things",
}
_WORD = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")
# "200k", "2022", "15" -- quantities the structured filters already handle.
# They are never in the prose, so counting them as unknown content words
# would penalise every question that mentions a number.
_NUMERIC_TOKEN = re.compile(r"^\d[\d.,]*[a-z]{0,2}$")

# Sections that usually answer a given kind of question. A small, honest
# prior -- it nudges ranking, it does not filter anything out.
_SECTION_PRIORS: dict[str, tuple[str, ...]] = {
    "price": ("pricing", "commercial", "cost", "fee", "budget", "investment"),
    "timeline": ("timeline", "schedule", "milestone", "phase", "delivery plan"),
    "team": ("team", "staffing", "resource", "roles"),
    "tech": ("architecture", "technology", "tech stack", "solution", "approach"),
    "risk": ("risk", "assumption", "dependency", "mitigation"),
    "outcome": ("result", "outcome", "impact", "benefit", "success"),
    "scope": ("scope", "deliverable", "requirement", "feature"),
    "quality": ("quality", "testing", "qa", "test", "assurance"),
    "security": ("security", "compliance", "privacy", "audit", "governance"),
    "process": ("approach", "methodology", "process", "ways of working", "delivery model"),
}
_INTENT_WORDS: dict[str, tuple[str, ...]] = {
    "price": ("price", "pricing", "cost", "budget", "quote", "quoted", "rate",
              "fee", "charge", "$", "usd", "bid", "commercial", "invoice",
              "payment", "milestone payment", "how much"),
    "timeline": ("timeline", "duration", "how long", "schedule", "milestone", "deadline", "weeks", "months"),
    "team": ("team", "how many people", "staffing", "headcount", "developers", "engineers", "roles"),
    "tech": ("tech", "stack", "technology", "architecture", "built with", "framework", "database"),
    "risk": ("risk", "assumption", "dependency", "challenge", "mitigat"),
    "outcome": ("outcome", "result", "impact", "roi", "benefit", "improved", "reduced", "increase"),
    "scope": ("scope", "deliverable", "included", "requirement", "feature"),
    # RFP response questions cluster heavily on these three, so they get
    # first-class intents rather than falling through to "general".
    "quality": ("qa", "quality assurance", "quality", "testing", "test coverage",
                "automation", "defect", "regression"),
    "security": ("security", "secure", "compliance", "compliant", "soc 2", "soc2",
                 "iso 27001", "hipaa", "gdpr", "pci", "penetration test", "pen test",
                 "vulnerability", "encryption", "audit"),
    "process": ("process", "methodology", "approach", "how do you", "ways of working",
                "agile", "scrum", "sprint", "governance", "handover", "knowledge transfer"),
}

_bm25_cache: dict = {"version": None, "index": None, "chunk_ids": []}


# --- plan -------------------------------------------------------------------

@dataclass
class Filters:
    industry: list[str] = field(default_factory=list)
    tech: list[str] = field(default_factory=list)
    doc_type: list[str] = field(default_factory=list)
    engagement_model: list[str] = field(default_factory=list)
    client: str = ""
    year_min: int = 0
    year_max: int = 0
    value_max: int = 0
    value_min: int = 0
    doc_ids: list[int] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(
            [
                self.industry, self.tech, self.doc_type, self.engagement_model,
                self.client, self.year_min, self.year_max, self.value_max,
                self.value_min, self.doc_ids,
            ]
        )

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}


@dataclass
class QueryPlan:
    original: str
    search_query: str
    filters: Filters = field(default_factory=Filters)
    intent: str = "general"
    planned_by: str = "heuristic"


@dataclass
class Hit:
    chunk_id: int
    doc_id: int
    text: str
    section_path: str
    page_from: int
    page_to: int
    title: str
    filename: str
    client: str
    industry: str
    doc_type: str
    year: int
    score: float = 0.0
    vector_rank: int | None = None
    keyword_rank: int | None = None

    def to_dict(self) -> dict:
        data = dict(self.__dict__)
        data["score"] = round(self.score, 4)
        return data


_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "search_query": {"type": "string"},
        "industry": {"type": "array", "items": {"type": "string", "enum": INDUSTRIES}},
        "tech": {"type": "array", "items": {"type": "string"}},
        "doc_type": {"type": "array", "items": {"type": "string", "enum": DOC_TYPES}},
        "engagement_model": {"type": "array", "items": {"type": "string", "enum": ENGAGEMENT_MODELS}},
        "client": {"type": "string"},
        "year_min": {"type": "integer"},
        "year_max": {"type": "integer"},
        "value_min": {"type": "integer"},
        "value_max": {"type": "integer"},
        "intent": {
            "type": "string",
            "enum": [
                "price", "timeline", "team", "tech", "risk", "outcome", "scope",
                "quality", "security", "process", "general",
            ],
        },
    },
    "required": ["search_query", "intent"],
}

_PLAN_SYSTEM = """You turn a question about a consulting firm's proposal library into a search plan.

Split the question into two parts:
1. FILTERS - hard constraints that can be checked against document metadata
   (industry, technology, document type, engagement model, client name, year range, contract value range).
2. `search_query` - the remaining meaning, rewritten as a dense keyword-rich phrase for semantic search.

Rules:
- Only set a filter the question actually states. Never infer one. An empty list is the correct answer most of the time.
- "under $200k" -> value_max 200000. "since 2022" -> year_min 2022.
- `search_query` must never be empty. If the question is pure filtering ("show me all healthcare work"),
  use the topic words themselves as the search query.
- Use 0 for any numeric filter that does not apply."""


def _guess_intent(text: str) -> str:
    lowered = text.lower()
    best, best_score = "general", 0
    for intent, words in _INTENT_WORDS.items():
        score = sum(1 for w in words if w in lowered)
        if score > best_score:
            best, best_score = intent, score
    return best


def _heuristic_plan(question: str) -> QueryPlan:
    lowered = question.lower()
    filters = Filters()

    for industry in INDUSTRIES:
        if industry == "other":
            continue
        if re.search(rf"(?<![a-z]){re.escape(industry.replace('_', ' '))}(?![a-z])", lowered):
            filters.industry.append(industry)

    filters.tech = find_tech_in_text(question)

    for doc_type, words in {
        "case_study": ["case study", "case studies"],
        "proposal": ["proposal", "proposals"],
        "sow": ["sow", "statement of work"],
        "rfp": ["rfp"],
        "capability": ["capability deck", "capabilities"],
    }.items():
        if any(w in lowered for w in words):
            filters.doc_type.append(doc_type)

    for model, words in {
        "fixed_bid": ["fixed bid", "fixed-bid", "fixed price"],
        "time_and_materials": ["time and materials", "t&m"],
        "dedicated_team": ["dedicated team", "staff augmentation"],
        "retainer": ["retainer"],
    }.items():
        if any(w in lowered for w in words):
            filters.engagement_model.append(model)

    if m := re.search(r"(?:since|after|from)\s+(20[0-2]\d)", lowered):
        filters.year_min = int(m.group(1))
    if m := re.search(r"(?:before|until|up to)\s+(20[0-2]\d)", lowered):
        filters.year_max = int(m.group(1))
    if m := re.search(r"\bin\s+(20[0-2]\d)\b", lowered):
        filters.year_min = filters.year_max = int(m.group(1))

    def _amount(text: str, suffix: str | None) -> int:
        value = float(text.replace(",", ""))
        return int(value * {"k": 1_000, "m": 1_000_000}.get((suffix or "").lower(), 1))

    if m := re.search(r"(?:under|below|less than|<)\s*\$?\s*([\d,.]+)\s*(k|m)?", lowered):
        filters.value_max = _amount(m.group(1), m.group(2))
    if m := re.search(r"(?:over|above|more than|>)\s*\$?\s*([\d,.]+)\s*(k|m)?", lowered):
        filters.value_min = _amount(m.group(1), m.group(2))

    terms = [w for w in _WORD.findall(lowered) if w not in _STOPWORDS]
    return QueryPlan(
        original=question,
        search_query=" ".join(terms) or question,
        filters=filters,
        intent=_guess_intent(question),
        planned_by="heuristic",
    )


def plan_query(question: str) -> QueryPlan:
    fallback = _heuristic_plan(question)
    if llm.is_offline():
        return fallback
    try:
        data = llm.complete_json(
            f"Question: {question}", system=_PLAN_SYSTEM, json_schema=_PLAN_SCHEMA
        )
    except Exception:
        return fallback

    from .metadata import normalise_tech

    filters = Filters(
        industry=[i for i in (data.get("industry") or []) if i in INDUSTRIES and i != "other"],
        tech=normalise_tech(data.get("tech") or []),
        doc_type=[d for d in (data.get("doc_type") or []) if d in DOC_TYPES],
        engagement_model=[
            e for e in (data.get("engagement_model") or []) if e in ENGAGEMENT_MODELS
        ],
        client=(data.get("client") or "").strip(),
        year_min=int(data.get("year_min") or 0),
        year_max=int(data.get("year_max") or 0),
        value_min=int(data.get("value_min") or 0),
        value_max=int(data.get("value_max") or 0),
    )
    return QueryPlan(
        original=question,
        search_query=(data.get("search_query") or "").strip() or fallback.search_query,
        filters=filters,
        intent=data.get("intent") or fallback.intent,
        planned_by="llm",
    )


def resolve_followup(question: str, history: list[dict]) -> str:
    """Rewrite a context-dependent follow-up into a standalone question.

    "And what did it cost?" retrieves nothing on its own -- the subject lives
    in the previous turn. Rewriting before retrieval (rather than pasting the
    whole history into the search query, which drags in stale topics) is what
    makes multi-turn conversation work.
    """
    if not history:
        return question
    words = _WORD.findall(question.lower())
    # Long, self-contained questions rarely need rewriting; skip the round trip.
    if len(words) > 12 and not re.match(
        r"^\s*(and|what about|how about|also|then|ok|okay|it|that|those|they|he|she)\b",
        question.strip().lower(),
    ):
        return question

    recent = history[-4:]
    if llm.is_offline():
        # Carry the nouns from the last user turn, which is the cheap
        # approximation of coreference resolution.
        previous = next(
            (m["content"] for m in reversed(recent) if m.get("role") == "user"), ""
        )
        carry = [w for w in _WORD.findall(previous.lower()) if w not in _STOPWORDS]
        if not carry:
            return question
        return f"{question} {' '.join(carry[:12])}"

    transcript = "\n".join(f"{m.get('role')}: {m.get('content', '')[:400]}" for m in recent)
    try:
        rewritten = llm.complete(
            f"Conversation so far:\n{transcript}\n\nFollow-up question: {question}\n\n"
            "Rewrite the follow-up as a standalone question that makes sense with no "
            "conversation history. Resolve every pronoun and implicit reference using "
            "the conversation. Return only the rewritten question.",
            temperature=0.0,
            max_output_tokens=200,
        )
        return rewritten.strip().strip('"') or question
    except Exception:
        return question


# --- filters -> document ids -------------------------------------------------

def resolve_filters(filters: Filters) -> list[int] | None:
    """Turn structured filters into a document id list (None = no filter)."""
    if filters.is_empty():
        return None

    clauses: list[str] = ["d.status = 'ready'"]
    params: list = []

    def in_clause(column: str, values: list[str]) -> None:
        clauses.append(f"d.{column} IN ({','.join('?' for _ in values)})")
        params.extend(values)

    if filters.industry:
        in_clause("industry", filters.industry)
    if filters.doc_type:
        in_clause("doc_type", filters.doc_type)
    if filters.engagement_model:
        in_clause("engagement_model", filters.engagement_model)
    if filters.client:
        clauses.append("(LOWER(d.client) LIKE ? OR LOWER(d.title) LIKE ?)")
        needle = f"%{filters.client.lower()}%"
        params.extend([needle, needle])
    if filters.year_min:
        clauses.append("d.year >= ?")
        params.append(filters.year_min)
    if filters.year_max:
        clauses.append("d.year <= ?")
        params.append(filters.year_max)
    # A document with no recorded value must not be excluded by a price
    # filter -- "unknown" is not "too expensive".
    if filters.value_max:
        clauses.append("(d.value_usd = 0 OR d.value_usd <= ?)")
        params.append(filters.value_max)
    if filters.value_min:
        clauses.append("(d.value_usd = 0 OR d.value_usd >= ?)")
        params.append(filters.value_min)
    if filters.doc_ids:
        clauses.append(f"d.id IN ({','.join('?' for _ in filters.doc_ids)})")
        params.extend(filters.doc_ids)

    # Every requested technology must be present: "Node AND AWS", not "or".
    if filters.tech:
        clauses.append(
            "(SELECT COUNT(DISTINCT t.tech) FROM doc_tech t "
            f"WHERE t.doc_id = d.id AND t.tech IN ({','.join('?' for _ in filters.tech)})) = ?"
        )
        params.extend(filters.tech)
        params.append(len(filters.tech))

    rows = db.query(f"SELECT d.id FROM documents d WHERE {' AND '.join(clauses)}", params)
    return [int(r["id"]) for r in rows]


def relax_filters(filters: Filters) -> Filters:
    """Drop the most brittle constraints for a second attempt.

    When a strict filter matches nothing, returning "no results" is usually
    the wrong answer -- the user asked for React Native *and* AWS *and*
    fintech, and we have two of the three. Widening tech to ANY-of and
    dropping the value band recovers those near-misses, and the response
    labels them as relaxed so the user is not misled.
    """
    relaxed = Filters(
        industry=list(filters.industry),
        tech=list(filters.tech[:1]),
        doc_type=list(filters.doc_type),
        engagement_model=[],
        client=filters.client,
        year_min=0,
        year_max=0,
        value_max=0,
        value_min=0,
        doc_ids=list(filters.doc_ids),
    )
    return relaxed


# --- keyword search ----------------------------------------------------------

def _corpus_version() -> str:
    row = db.query_one("SELECT COUNT(*) AS n, COALESCE(MAX(id), 0) AS m FROM chunks")
    return f"{row['n']}:{row['m']}" if row else "0:0"


def _bm25_index():
    from rank_bm25 import BM25Okapi

    version = _corpus_version()
    if _bm25_cache["version"] == version and _bm25_cache["index"] is not None:
        return _bm25_cache["index"], _bm25_cache["chunk_ids"]

    rows = db.query(
        "SELECT c.id, c.text, c.context_header FROM chunks c "
        "JOIN documents d ON d.id = c.doc_id WHERE d.status = 'ready'"
    )
    if not rows:
        _bm25_cache.update({"version": version, "index": None, "chunk_ids": []})
        return None, []

    corpus = [
        [w for w in _WORD.findall(f"{r['context_header']} {r['text']}".lower()) if w not in _STOPWORDS]
        for r in rows
    ]
    index = BM25Okapi(corpus)
    chunk_ids = [int(r["id"]) for r in rows]
    _bm25_cache.update({"version": version, "index": index, "chunk_ids": chunk_ids})
    return index, chunk_ids


def keyword_search(query: str, k: int, allowed_chunk_ids: set[int] | None) -> list[tuple[int, float]]:
    index, chunk_ids = _bm25_index()
    if index is None:
        return []
    terms = [w for w in _WORD.findall(query.lower()) if w not in _STOPWORDS]
    if not terms:
        return []
    scores = index.get_scores(terms)
    pairs = [
        (cid, float(score))
        for cid, score in zip(chunk_ids, scores)
        if score > 0 and (allowed_chunk_ids is None or cid in allowed_chunk_ids)
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]


# --- fusion and rerank -------------------------------------------------------

def _rrf_fuse(
    vector_hits: list[tuple[int, float]],
    keyword_hits: list[tuple[int, float]],
    rrf_k: int,
) -> dict[int, dict]:
    fused: dict[int, dict] = {}
    for rank, (chunk_id, _score) in enumerate(vector_hits, start=1):
        entry = fused.setdefault(chunk_id, {"score": 0.0, "vector_rank": None, "keyword_rank": None})
        entry["score"] += 1.0 / (rrf_k + rank)
        entry["vector_rank"] = rank
    for rank, (chunk_id, _score) in enumerate(keyword_hits, start=1):
        entry = fused.setdefault(chunk_id, {"score": 0.0, "vector_rank": None, "keyword_rank": None})
        entry["score"] += 1.0 / (rrf_k + rank)
        entry["keyword_rank"] = rank
    return fused


def _rerank(hits: list[Hit], plan: QueryPlan) -> list[Hit]:
    """Cheap, explainable reranking. No second model, no extra latency.

    Three signals, all of which a reviewer can reason about:
      * section prior   -- a pricing question is likely answered in a pricing section
      * metadata agreement -- the document matches the filters the user asked for
      * agreement bonus -- both retrievers independently surfaced this chunk
    """
    priors = _SECTION_PRIORS.get(plan.intent, ())
    for hit in hits:
        boost = 1.0
        section = hit.section_path.lower()
        if priors and any(word in section for word in priors):
            boost += 0.35
        if plan.filters.industry and hit.industry in plan.filters.industry:
            boost += 0.10
        if plan.filters.doc_type and hit.doc_type in plan.filters.doc_type:
            boost += 0.10
        if plan.filters.client and plan.filters.client.lower() in (hit.client or "").lower():
            boost += 0.20
        if hit.vector_rank is not None and hit.keyword_rank is not None:
            boost += 0.25
        hit.score *= boost
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def _load_hits(fused: dict[int, dict]) -> list[Hit]:
    if not fused:
        return []
    ids = list(fused.keys())
    rows = db.query(
        f"""SELECT c.id AS chunk_id, c.doc_id, c.text, c.section_path, c.page_from, c.page_to,
                   d.title, d.filename, d.client, d.industry, d.doc_type, d.year
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE c.id IN ({','.join('?' for _ in ids)})""",
        ids,
    )
    hits: list[Hit] = []
    for row in rows:
        meta = fused[int(row["chunk_id"])]
        hits.append(
            Hit(
                chunk_id=int(row["chunk_id"]),
                doc_id=int(row["doc_id"]),
                text=row["text"],
                section_path=row["section_path"] or "",
                page_from=int(row["page_from"]),
                page_to=int(row["page_to"]),
                title=row["title"],
                filename=row["filename"],
                client=row["client"] or "",
                industry=row["industry"] or "",
                doc_type=row["doc_type"] or "",
                year=int(row["year"] or 0),
                score=meta["score"],
                vector_rank=meta["vector_rank"],
                keyword_rank=meta["keyword_rank"],
            )
        )
    return hits


def _term_present(term: str, evidence: str) -> bool:
    """Is this query term present in the evidence, allowing for hyphenation?

    Writers are inconsistent about compounds: one proposal says "fixed-bid",
    the next says "fixed bid", and a user may type either. A literal substring
    test scores the retrieved Commercials section as not containing the term
    the query was about, which collapses coverage and refuses a question whose
    filters and retrieval were both correct.
    """
    if term in evidence:
        return True
    if "-" in term:
        return term.replace("-", " ") in evidence or term.replace("-", "") in evidence
    return False


def _split_unknown_compounds(terms: set[str], idf: dict[str, float]) -> set[str]:
    """Break a hyphenated term the corpus does not know into its parts.

    "fixed-bid" tokenises as a single word, and no document writes it that way
    -- proposals say "fixed bid". Left whole it counts as vocabulary the corpus
    has never seen, so "what fixed-bid work have we done under $200k?" scored
    0.08 and was refused even though the filters and retrieval were both
    correct. Splitting only when the compound is unknown keeps genuine
    hyphenated names intact wherever the corpus actually uses them.
    """
    expanded: set[str] = set()
    for term in terms:
        if term in idf or "-" not in term:
            expanded.add(term)
            continue
        parts = [p for p in term.split("-") if len(p) >= 2]
        if any(p in idf for p in parts):
            expanded.update(parts)
        else:
            expanded.add(term)
    return expanded


def _term_coverage(terms: set[str], evidence: str) -> tuple[float, float]:
    """Measure how well the corpus covers a query's informative words.

    Returns `(coverage, unknown_fraction)`, two signals that must stay
    separate because they answer different questions:

    * **coverage** -- of the query terms the corpus *does* know, how much of
      that weight appears in the evidence we actually retrieved? This measures
      the retrieval, and it is computed over known terms only. Letting unknown
      terms into this denominator double-counts them, since they are already
      penalised by `unknown_fraction`: "describe your QA process" retrieved the
      Quality Assurance section correctly, yet scored 0.28 coverage because
      "process" is a synonym the corpus spells "approach". Punishing good
      retrieval for the user's choice of wording is precisely backwards.
    * **unknown_fraction** -- share of the query's term weight sitting on words
      that appear in *no chunk of the corpus at all*. This measures the corpus,
      not the retrieval.

    The second number is the one that reliably detects a question the library
    simply cannot answer, and it needs to be measured separately. In a small
    corpus, IDF alone does not separate the cases: "done" occurs in a single
    chunk and therefore scores the same IDF as a word occurring nowhere, so
    "have we done automotive work" looked well covered on the strength of
    "done" and "work" while "automotive" -- the only word that decides the
    answer -- was absent from every document. A term with zero document
    frequency is categorically different from a rare one, and treating it that
    way is what turns "no automotive experience" into a refusal rather than a
    confident answer about unrelated projects.
    """
    if not terms:
        return 0.0, 0.0

    index, chunk_ids = _bm25_index()
    idf: dict[str, float] = getattr(index, "idf", {}) or {}
    # The IDF a term with zero document frequency would receive, on BM25's own
    # formula -- strictly above the highest IDF any present term can reach.
    corpus_size = max(1, len(chunk_ids))
    unseen_weight = math.log(1 + (corpus_size + 0.5) / 0.5)

    total = known_total = covered = unknown = 0.0
    for term in _split_unknown_compounds(terms, idf):
        known = term in idf
        # BM25Okapi floors negative IDFs for very common terms; clamp so a
        # stopword-ish term can never contribute negative weight.
        weight = max(0.0, idf.get(term, unseen_weight)) if known else unseen_weight
        if weight == 0.0:
            continue
        total += weight
        if known:
            known_total += weight
            if _term_present(term, evidence):
                covered += weight
        else:
            unknown += weight

    if not total:
        return 0.0, 0.0
    # No known terms at all: coverage is 0, and unknown is 1.0, so the caller
    # discounts the score to near nothing -- which is the right answer for a
    # question written entirely in vocabulary the library has never seen.
    coverage = (covered / known_total) if known_total else 0.0
    return coverage, unknown / total


def _confidence(hits: list[Hit], vector_hits: list[tuple[int, float]], search_query: str) -> float:
    """How much should the generator trust this evidence?

    Two independent relevance signals, then a corroboration *multiplier*:

    * **best cosine similarity** -- is anything semantically close to the query?
    * **term coverage** -- what fraction of the query's content words actually
      occur in the retrieved text? This is the signal that catches a question
      about a subject the corpus simply does not contain. It also keeps the
      score meaningful under the offline hashing embedder, whose cosine
      similarities are compressed into a narrow band.

    Corroboration (how many distinct documents agree) *scales* that relevance
    rather than adding to it. As an additive term it could carry the score on
    its own: a question about spacecraft avionics against a corpus of banking
    proposals retrieved three unrelated documents, scored 0.25 x 1.0 on
    corroboration alone, cleared the threshold and got a confident answer
    assembled from irrelevant evidence. Evidence that is not relevant cannot
    become trustworthy by being plentiful, so it multiplies.
    """
    if not hits:
        return 0.0

    best_similarity = max((score for _cid, score in vector_hits), default=0.0)

    # Length 2 is deliberate: "QA", "AI" and "Go" are real content terms here.
    terms = {
        w
        for w in _WORD.findall(search_query.lower())
        if w not in _STOPWORDS and len(w) >= 2 and not _NUMERIC_TOKEN.match(w)
    }
    evidence = " ".join(h.text for h in hits).lower()
    coverage, unknown = _term_coverage(terms, evidence)

    relevance = 0.6 * max(0.0, best_similarity) + 0.4 * coverage
    # Discount by the share of the question the corpus has never seen. A
    # question made mostly of words absent from every document is a question
    # this library cannot answer, however similar the nearest chunks look.
    # The coefficient is 0.5 rather than 1.0 because an unseen word is
    # ambiguous evidence -- it may mean the subject is absent ("automotive"),
    # or merely that the user picked a synonym ("outcomes" for "results").
    # Halving keeps a single synonym survivable while a query made entirely of
    # unknown words still collapses.
    relevance *= 1.0 - 0.5 * unknown
    corroboration = min(1.0, len({h.doc_id for h in hits}) / 3.0)
    return round(max(0.0, min(1.0, relevance * (0.8 + 0.2 * corroboration))), 4)


@dataclass
class RetrievalResult:
    hits: list[Hit]
    plan: QueryPlan
    confidence: float
    filtered_doc_count: int | None
    relaxed: bool
    took_ms: int
    candidates: int


def retrieve(
    question: str,
    history: list[dict] | None = None,
    top_k: int | None = None,
    document_ids: list[int] | None = None,
) -> RetrievalResult:
    """Run the full retrieval pipeline for one question.

    `document_ids` scopes the search to an explicit selection made in the UI.
    It is intersected with, not replaced by, whatever the query planner infers,
    so "what did this cost?" asked against two selected documents stays scoped
    to those two.
    """
    settings = get_settings()
    started = time.perf_counter()

    standalone = resolve_followup(question, history or [])
    plan = plan_query(standalone)
    plan.original = question
    if document_ids:
        plan.filters.doc_ids = list(document_ids)

    doc_ids = resolve_filters(plan.filters)
    relaxed = False
    if doc_ids is not None and not doc_ids:
        relaxed_filters = relax_filters(plan.filters)
        widened = resolve_filters(relaxed_filters)
        if widened:
            doc_ids, relaxed = widened, True
        else:
            doc_ids, relaxed = None, True

    query_vector = embed_one(plan.search_query, QUERY)
    vector_hits = vectorstore.search(query_vector, settings.vector_top_k, doc_ids)

    allowed: set[int] | None = None
    if doc_ids is not None:
        rows = db.query(
            f"SELECT id FROM chunks WHERE doc_id IN ({','.join('?' for _ in doc_ids)})",
            doc_ids,
        ) if doc_ids else []
        allowed = {int(r["id"]) for r in rows}

    keyword_hits = keyword_search(plan.search_query, settings.keyword_top_k, allowed)

    fused = _rrf_fuse(vector_hits, keyword_hits, settings.rrf_k)
    hits = _rerank(_load_hits(fused), plan)
    limit = top_k or settings.final_top_k
    top_hits = hits[:limit]

    return RetrievalResult(
        hits=top_hits,
        plan=plan,
        confidence=_confidence(top_hits, vector_hits, plan.search_query),
        filtered_doc_count=None if doc_ids is None else len(doc_ids),
        relaxed=relaxed,
        took_ms=int((time.perf_counter() - started) * 1000),
        candidates=len(fused),
    )
