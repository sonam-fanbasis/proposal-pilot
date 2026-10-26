"""Runtime configuration.

Everything is read from environment variables (optionally via a local `.env`),
so the same image runs in dev and prod with no code change. Defaults are
chosen so that `python -m app.main` works on a machine with no API key at
all -- the LLM and embedding layers each have an offline fallback.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


class Settings(BaseSettings):
    # --- storage ---
    data_dir: Path = BACKEND_DIR / "data"
    db_path: Path = BACKEND_DIR / "data" / "proposalpilot.db"
    upload_dir: Path = BACKEND_DIR / "data" / "uploads"

    # --- LLM provider ---
    # "gemini" uses GEMINI_API_KEY; "offline" is a deterministic extractive
    # responder that needs no network, used automatically when no key is set.
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    # Chosen for free-tier headroom, which is the binding constraint here.
    # "gemini-flash-latest" tracks the newest model and is capped at 20
    # requests/day -- one re-index plus a few questions exhausts it. The
    # 2.5 series is closed to new API keys entirely. The flash-lite alias
    # has real headroom and, being an alias, survives model retirement.
    # For higher-quality RFP drafts on a paid key, set GEMINI_MODEL to
    # gemini-3.6-flash in backend/.env.
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    gemini_embed_model: str = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

    # --- embeddings ---
    # gemini-embedding-001 defaults to 3072-d but accepts an explicit
    # output_dimensionality; 768 is requested to keep the index compact and
    # to match the offline hashing embedder's width, so a corpus stays
    # queryable if the key is removed (results degrade, nothing crashes).
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "768"))

    # --- chunking ---
    chunk_target_chars: int = int(os.getenv("CHUNK_TARGET_CHARS", "1100"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "180"))
    chunk_min_chars: int = int(os.getenv("CHUNK_MIN_CHARS", "120"))

    # --- retrieval ---
    vector_top_k: int = int(os.getenv("VECTOR_TOP_K", "20"))
    keyword_top_k: int = int(os.getenv("KEYWORD_TOP_K", "20"))
    final_top_k: int = int(os.getenv("FINAL_TOP_K", "6"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.18"))

    # --- Report Lab (long / visual reports) ---
    # Reports are embedded locally (fastembed, ONNX on CPU): a 200-page report
    # is ~1,000 chunks, which would take minutes against the free-tier
    # embedding quota and seconds locally.
    # arctic-embed-xs: 384-d, 512-token window, ~2.3x faster than bge-small on
    # CPU at comparable retrieval quality (measured: 200 GEP chunks in 16s vs 37s).
    report_embed_model: str = os.getenv("REPORT_EMBED_MODEL", "snowflake/snowflake-arctic-embed-xs")
    report_embed_dim: int = int(os.getenv("REPORT_EMBED_DIM", "384"))
    report_rerank_model: str = os.getenv("REPORT_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    model_cache_dir: Path = BACKEND_DIR / "data" / "models"
    report_chunk_chars: int = int(os.getenv("REPORT_CHUNK_CHARS", "1200"))
    report_chunk_overlap: int = int(os.getenv("REPORT_CHUNK_OVERLAP", "200"))
    report_candidates: int = int(os.getenv("REPORT_CANDIDATES", "40"))
    report_rerank_pool: int = int(os.getenv("REPORT_RERANK_POOL", "28"))
    report_final_k: int = int(os.getenv("REPORT_FINAL_K", "7"))
    report_context_chars: int = int(os.getenv("REPORT_CONTEXT_CHARS", "14000"))
    # Vision transcription of charts/call-outs: only for short documents,
    # where facts live in graphics and the page count keeps the cost bounded.
    vision_enabled: bool = os.getenv("VISION_ENABLED", "true").lower() == "true"
    vision_max_pages: int = int(os.getenv("VISION_MAX_PAGES", "30"))
    vision_dpi: int = int(os.getenv("VISION_DPI", "110"))
    vision_workers: int = int(os.getenv("VISION_WORKERS", "4"))

    # --- server ---
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def resolved_provider(self) -> str:
        """`auto` resolves to gemini when a key exists, else offline."""
        if self.llm_provider != "auto":
            return self.llm_provider
        return "gemini" if self.gemini_api_key else "offline"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.upload_dir.mkdir(parents=True, exist_ok=True)
    return s
