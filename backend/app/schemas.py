"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Mode = Literal["answer", "draft"]


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    mode: Mode = "answer"
    conversation_id: int | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[int] = Field(default_factory=list)


class CitationOut(BaseModel):
    index: int
    chunk_id: int
    doc_id: int
    title: str
    filename: str
    section_path: str
    page_from: int
    page_to: int
    snippet: str


class RetrievalDebug(BaseModel):
    """Surfaced in the UI so the pipeline is inspectable, not a black box."""

    search_query: str
    standalone_question: str
    intent: str
    planned_by: str
    filters: dict[str, Any]
    filtered_document_count: int | None
    filters_relaxed: bool
    candidates: int
    retrieval_ms: int
    embedding_backend: str
    llm: str


class AskResponse(BaseModel):
    answer: str
    mode: Mode
    citations: list[CitationOut]
    gaps: list[str]
    confidence: float
    ungrounded_sentences: int
    invalid_citations: list[int]
    conversation_id: int
    message_id: int
    total_ms: int
    retrieval: RetrievalDebug


class DocumentOut(BaseModel):
    id: int
    filename: str
    title: str
    status: str
    error: str | None = None
    client: str = ""
    industry: str = ""
    doc_type: str = ""
    engagement_model: str = ""
    year: int = 0
    value_usd: int = 0
    duration_months: int = 0
    tech_stack: list[str] = Field(default_factory=list)
    summary: str = ""
    page_count: int = 0
    byte_size: int = 0
    chunk_count: int = 0
    created_at: str = ""


class UploadResponse(BaseModel):
    document_id: int
    filename: str
    status: str
    message: str


class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: str
    message_count: int = 0


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    mode: str
    citations: list[CitationOut] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    created_at: str = ""


class FacetsOut(BaseModel):
    industries: list[dict[str, Any]]
    doc_types: list[dict[str, Any]]
    engagement_models: list[dict[str, Any]]
    tech: list[dict[str, Any]]
    years: list[int]


class StatsOut(BaseModel):
    documents: int
    documents_ready: int
    documents_failed: int
    chunks: int
    vectors: int
    conversations: int
    questions_asked: int
    median_retrieval_ms: float | None
    mean_confidence: float | None
    low_confidence_rate: float | None
    embedding_backend: str
    llm: str
    vector_index: str


class EvalRunRequest(BaseModel):
    k: int = Field(default=6, ge=1, le=20)
    generate_answers: bool = True
