#!/usr/bin/env python3
"""Load the sample proposal library and seed the evaluation cases.

Run this for a working demo without hunting for documents:

    .venv/bin/python seed.py            # ingest ../sample_docs
    .venv/bin/python seed.py --eval     # ingest, then run the eval harness
    .venv/bin/python seed.py --reset    # wipe the index first
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import db, eval as eval_mod  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.ingestion import IngestionError, ingest_path  # noqa: E402
from app.llm import provider_name  # noqa: E402
from app.parsing import SUPPORTED_EXTENSIONS  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_docs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=SAMPLE_DIR, help="Directory of documents to ingest")
    parser.add_argument("--reset", action="store_true", help="Delete the existing index first")
    parser.add_argument("--eval", action="store_true", help="Run the evaluation harness afterwards")
    args = parser.parse_args()

    settings = get_settings()
    if args.reset:
        for suffix in ("", "-wal", "-shm"):
            target = Path(str(settings.db_path) + suffix)
            target.unlink(missing_ok=True)
        print("Index reset.")

    db.init_db()
    print(f"LLM: {provider_name()}")

    if not args.dir.is_dir():
        print(f"No such directory: {args.dir}", file=sys.stderr)
        return 1

    files = sorted(f for f in args.dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        print(f"No supported documents in {args.dir}", file=sys.stderr)
        return 1

    ingested = 0
    for path in files:
        try:
            result = ingest_path(path)
            meta = result["metadata"]
            print(
                f"  {path.name[:42]:44} {result['chunks']:>3} chunks  "
                f"{meta['industry']:<11} {meta['doc_type']:<11} "
                f"{meta['engagement_model']:<19} "
                f"{('$' + format(meta['value_usd'], ',')) if meta['value_usd'] else '—':>10}"
            )
            ingested += 1
        except IngestionError as exc:
            print(f"  {path.name[:42]:44} skipped — {exc}")
        except Exception as exc:  # pragma: no cover
            print(f"  {path.name[:42]:44} FAILED — {type(exc).__name__}: {exc}")

    print(f"\n{ingested} document(s) indexed.")
    print(f"Evaluation cases seeded: {eval_mod.seed_cases()}")

    if args.eval:
        print("\nRunning evaluation…")
        print(json.dumps(eval_mod.run()["metrics"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
