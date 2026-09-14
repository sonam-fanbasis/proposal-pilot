"""Test fixtures.

Every test runs against a throwaway SQLite file in a tmp directory, so the
suite never touches a developer's real index and tests cannot leak state into
one another.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def app_db(tmp_path, monkeypatch):
    from app import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "offline")

    settings = config.get_settings()
    monkeypatch.setattr(settings, "db_path", tmp_path / "test.db")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    (tmp_path / "uploads").mkdir(exist_ok=True)

    from app import db, retrieval

    db._local.__dict__.pop("conn", None)
    db.init_db()
    retrieval._bm25_cache.update({"version": None, "index": None, "chunk_ids": []})

    yield settings

    db._local.__dict__.pop("conn", None)
    config.get_settings.cache_clear()


@pytest.fixture
def seeded(app_db, tmp_path):
    """A small two-document corpus with deliberately contrasting metadata."""
    from app.ingestion import ingest_path

    alpha = tmp_path / "alpha_bank_proposal.md"
    alpha.write_text(
        "# Alpha Bank Payments Proposal\n\n"
        "## Executive Summary\n"
        "Alpha Bank asked us to rebuild the payments hub serving nine million customers. "
        "The legacy system settled transactions in batches overnight.\n\n"
        "## Technology Stack\n"
        "The platform is built in Node.js with React Native clients on AWS, backed by PostgreSQL.\n\n"
        "## Commercials\n"
        "This is a fixed bid of USD 450,000 covering all roles. Delivered in 2023 over 9 months.\n",
        encoding="utf-8",
    )
    beta = tmp_path / "beta_health_case_study.md"
    beta.write_text(
        "# Beta Health Telemedicine Case Study\n\n"
        "## Engagement Model\n"
        "This was a dedicated team engagement rather than a fixed bid, because the clinical "
        "workflow had never been mapped. Total engagement value was USD 680,000.\n\n"
        "## Technology Stack\n"
        "Python with Django on Azure, using PostgreSQL for patient records.\n\n"
        "## Results\n"
        "Administrative overhead per consultation fell from 7 minutes to 40 seconds. "
        "The platform met HIPAA requirements in month 14.\n",
        encoding="utf-8",
    )
    ingest_path(alpha)
    ingest_path(beta)
    return app_db
