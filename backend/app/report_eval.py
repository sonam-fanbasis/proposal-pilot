"""Benchmark the Report Lab against the use-case sample questions.

Each case is scored on three independent axes, because "the answer was
right" hides *why* a pipeline fails:

* **retrieval hit** -- did any retrieved chunk come from a page that holds
  the answer? A miss here is a parsing/chunking/retrieval problem.
* **scope precision** -- for questions that name a section ("South Asia",
  "Chapter 3"), what share of the evidence came from that section? Low
  precision is how region-mixing hallucinations start.
* **answer keywords** -- does the generated answer contain the figures the
  reference answer requires ("346", "45", "50")? A miss with a retrieval hit
  is a generation problem.

A case passes when retrieval hits and every keyword is present.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import db
from .config import BACKEND_DIR
from .report_generation import generate
from .report_retrieval import retrieve

CASES_PATH = BACKEND_DIR / "report_eval_cases.json"


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text())


def _match_report(pattern: str, reports: list[dict]) -> dict | None:
    pattern = pattern.lower()
    for report in reports:
        haystack = f"{report['filename']} {report['title']}".lower().replace("-", " ").replace("_", " ")
        if pattern in haystack or (pattern == "gep" and "global economic prospects" in haystack):
            return report
    return None


def samples_for(report: dict) -> list[str]:
    reports = [report]
    return [c["question"] for c in load_cases() if _match_report(c["report"], reports)]


def run(report_ids: list[int] | None = None, generate_answers: bool = True) -> dict:
    reports = [dict(r) for r in db.query("SELECT id, filename, title FROM reports WHERE status = 'ready'")]
    if report_ids:
        reports = [r for r in reports if r["id"] in report_ids]
    started = time.perf_counter()
    details = []
    for case in load_cases():
        report = _match_report(case["report"], reports)
        if report is None:
            continue
        t0 = time.perf_counter()
        result = retrieve(case["question"], [report["id"]])
        expected_pages = set(case.get("expect_pages") or [])
        hit_pages = [p for h in result.hits for p in range(h.page_from, h.page_to + 1)]
        retrieval_hit = not expected_pages or bool(expected_pages & set(hit_pages))
        first_rank = next(
            (i for i, h in enumerate(result.hits, 1) if expected_pages & set(range(h.page_from, h.page_to + 1))), None
        )
        scope = case.get("expect_section") or ""
        precision = None
        if scope:
            precision = round(sum(1 for h in result.hits if scope.lower() in h.section_path.lower()) / max(1, len(result.hits)), 2)

        answer = {"answer": "", "citations": [], "ungrounded_sentences": 0}
        if generate_answers:
            answer = generate(result)
        text = answer["answer"].lower()
        missing = [k for k in case["expect_keywords"] if not re.search(re.escape(k.lower()), text)] if generate_answers else []
        details.append({
            "report": report["title"], "use_case": case["use_case"], "question": case["question"],
            "reference": case.get("reference", ""), "route": result.route.kind, "routed_to": result.route.labels,
            "retrieval_hit": retrieval_hit, "first_relevant_rank": first_rank, "scope_precision": precision,
            "content_types": sorted({h.content_type for h in result.hits}),
            "confidence": result.confidence, "answer": answer["answer"], "citations": len(answer["citations"]),
            "missing_keywords": missing, "ungrounded_sentences": answer.get("ungrounded_sentences", 0),
            "passed": retrieval_hit and not missing and generate_answers,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        })
    n = len(details) or 1
    precisions = [d["scope_precision"] for d in details if d["scope_precision"] is not None]
    ranks = [d["first_relevant_rank"] for d in details]
    metrics = {
        "cases": len(details),
        "passed": sum(d["passed"] for d in details),
        "retrieval_hit_rate": round(sum(d["retrieval_hit"] for d in details) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        "scope_precision": round(sum(precisions) / len(precisions), 3) if precisions else None,
        "answer_keyword_rate": round(sum(not d["missing_keywords"] for d in details) / n, 3) if generate_answers else None,
        "ungrounded_sentences": sum(d["ungrounded_sentences"] for d in details),
        "seconds": round(time.perf_counter() - started, 1),
    }
    return {"metrics": metrics, "details": details}


def to_markdown(result: dict) -> str:
    m = result["metrics"]
    lines = [
        "# Report Lab — sample question benchmark", "",
        f"**{m['passed']}/{m['cases']} passed** · retrieval hit rate {m['retrieval_hit_rate']:.0%} · "
        f"MRR {m['mrr']} · scope precision {m['scope_precision']} · answer keyword rate {m['answer_keyword_rate']} · "
        f"{m['seconds']}s", "",
    ]
    for d in result["details"]:
        lines += [
            f"## {'✅' if d['passed'] else '❌'} {d['question']}",
            f"*{d['use_case']} — {d['report']}*  ",
            f"route: `{d['route']}` {', '.join(d['routed_to'])} · first relevant rank: {d['first_relevant_rank']} · "
            f"scope precision: {d['scope_precision']} · evidence types: {', '.join(d['content_types'])} · "
            f"confidence {d['confidence']}", "",
            d["answer"], "",
            f"> **Reference:** {d['reference']}", "",
        ]
        if d["missing_keywords"]:
            lines += [f"Missing expected terms: {', '.join(d['missing_keywords'])}", ""]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI
    import argparse

    from . import report_store

    parser = argparse.ArgumentParser(description="Run the Report Lab sample-question benchmark.")
    parser.add_argument("--ingest", nargs="*", default=[], help="PDF files to ingest first")
    parser.add_argument("--no-answers", action="store_true", help="retrieval metrics only (no LLM calls)")
    parser.add_argument("--out", default="", help="write a markdown report here")
    args = parser.parse_args()

    db.init_db()
    report_store.init()
    for path in args.ingest:
        print(f"ingesting {path} ...", flush=True)
        report_store.ingest_file(Path(path))
    result = run(generate_answers=not args.no_answers)
    markdown = to_markdown(result)
    if args.out:
        Path(args.out).write_text(markdown)
        print(f"wrote {args.out}")
    print(json.dumps(result["metrics"], indent=2))
    for d in result["details"]:
        print(f"{'PASS' if d['passed'] else 'FAIL'}  {d['question'][:80]}  missing={d['missing_keywords']} hit={d['retrieval_hit']}")


if __name__ == "__main__":  # pragma: no cover
    main()
