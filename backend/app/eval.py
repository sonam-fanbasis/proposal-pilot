"""Retrieval and grounding evaluation.

"The answers look good" is not a quality claim, and it is the only claim most
RAG demos can make. This harness turns a set of golden questions into numbers
that move when the pipeline changes:

* **Recall@k** -- did the document that actually contains the answer appear in
  the retrieved set at all? If it did not, no amount of prompt engineering
  downstream can save the answer. This is the metric that tells you whether a
  bad answer is a *retrieval* bug or a *generation* bug, which is the single
  most useful thing to know when debugging RAG.
* **MRR** -- how high did it rank? Evidence at position 1 gets used; evidence
  at position 8 competes with seven distractors for the model's attention.
* **Keyword coverage** -- did the generated answer actually contain the facts
  the question was asking for?
* **Citation validity and ungrounded-sentence rate** -- did the answer cite
  evidence that exists, and did any factual sentence go uncited?

The golden set ships in `eval_cases.json` so the numbers are reproducible on a
fresh checkout, and results are written to `eval_runs` so runs are comparable
over time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import db
from .generation import generate
from .retrieval import retrieve

DEFAULT_CASES_PATH = Path(__file__).resolve().parent.parent / "eval_cases.json"


def load_cases_from_file(path: Path | None = None) -> list[dict]:
    path = path or DEFAULT_CASES_PATH
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def seed_cases(path: Path | None = None) -> int:
    cases = load_cases_from_file(path)
    db.execute("DELETE FROM eval_cases")
    for case in cases:
        db.execute(
            "INSERT INTO eval_cases(question, expect_keywords, expect_doc, note) VALUES (?, ?, ?, ?)",
            (
                case["question"],
                json.dumps(case.get("expect_keywords", [])),
                case.get("expect_doc", ""),
                case.get("note", ""),
            ),
        )
    return len(cases)


def _document_rank(hits, expect_doc: str) -> int | None:
    """1-based rank of the first hit from the expected document."""
    if not expect_doc:
        return None
    needle = expect_doc.lower()
    for rank, hit in enumerate(hits, start=1):
        if needle in hit.filename.lower() or needle in hit.title.lower():
            return rank
    return None


def run(k: int = 6, generate_answers: bool = True, cases: list[dict] | None = None) -> dict:
    if cases is None:
        rows = db.query("SELECT question, expect_keywords, expect_doc, note FROM eval_cases")
        cases = [
            {
                "question": r["question"],
                "expect_keywords": json.loads(r["expect_keywords"] or "[]"),
                "expect_doc": r["expect_doc"] or "",
                "note": r["note"] or "",
            }
            for r in rows
        ]
    if not cases:
        return {"error": "No evaluation cases. POST /api/eval/seed first.", "cases": 0}

    started = time.perf_counter()
    details: list[dict] = []
    reciprocal_ranks: list[float] = []
    recall_hits = 0
    recall_total = 0
    keyword_scores: list[float] = []
    ungrounded_total = 0
    invalid_citation_cases = 0
    answered = 0

    for case in cases:
        result = retrieve(case["question"], top_k=k)
        rank = _document_rank(result.hits, case.get("expect_doc", ""))

        if case.get("expect_doc"):
            recall_total += 1
            if rank is not None:
                recall_hits += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        detail = {
            "question": case["question"],
            "expect_doc": case.get("expect_doc", ""),
            "retrieved_rank": rank,
            "confidence": result.confidence,
            "retrieval_ms": result.took_ms,
            "filters": result.plan.filters.to_dict(),
            "intent": result.plan.intent,
            "top_documents": list(dict.fromkeys(h.title for h in result.hits))[:3],
        }

        if generate_answers:
            answer = generate(case["question"], result)
            lowered = answer.text.lower()
            expected = [kw.lower() for kw in case.get("expect_keywords", [])]
            found = [kw for kw in expected if kw in lowered]
            coverage = len(found) / len(expected) if expected else None
            if coverage is not None:
                keyword_scores.append(coverage)
            ungrounded_total += answer.ungrounded_sentences
            if answer.invalid_citations:
                invalid_citation_cases += 1
            answered += 1
            detail.update(
                {
                    "keyword_coverage": None if coverage is None else round(coverage, 3),
                    "missing_keywords": [kw for kw in expected if kw not in found],
                    "citations": len(answer.citations),
                    "ungrounded_sentences": answer.ungrounded_sentences,
                    "invalid_citations": answer.invalid_citations,
                    "answer_preview": answer.text[:300],
                }
            )

        details.append(detail)

    metrics = {
        "cases": len(cases),
        f"recall_at_{k}": round(recall_hits / recall_total, 3) if recall_total else None,
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3) if reciprocal_ranks else None,
        "keyword_coverage": round(sum(keyword_scores) / len(keyword_scores), 3) if keyword_scores else None,
        "mean_confidence": round(
            sum(d["confidence"] for d in details) / len(details), 3
        ) if details else None,
        "ungrounded_sentences_total": ungrounded_total,
        "cases_with_invalid_citations": invalid_citation_cases,
        "answers_generated": answered,
        "median_retrieval_ms": sorted(d["retrieval_ms"] for d in details)[len(details) // 2] if details else None,
        "total_seconds": round(time.perf_counter() - started, 2),
    }

    db.execute(
        "INSERT INTO eval_runs(metrics_json, detail_json) VALUES (?, ?)",
        (json.dumps(metrics), json.dumps(details)),
    )
    return {"metrics": metrics, "details": details}
