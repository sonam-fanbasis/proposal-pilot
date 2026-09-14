export type Mode = 'answer' | 'draft'

export interface Citation {
  index: number
  chunk_id: number
  doc_id: number
  title: string
  filename: string
  section_path: string
  page_from: number
  page_to: number
  snippet: string
}

export interface RetrievalDebug {
  search_query: string
  standalone_question: string
  intent: string
  planned_by: string
  filters: Record<string, unknown>
  filtered_document_count: number | null
  filters_relaxed: boolean
  candidates: number
  retrieval_ms: number
  embedding_backend: string
  llm: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  mode: Mode
  citations: Citation[]
  gaps: string[]
  confidence: number
  ungrounded: number
  invalidCitations: number[]
  retrieval?: RetrievalDebug
  totalMs?: number
  streaming?: boolean
}

export interface DocumentRecord {
  id: number
  filename: string
  title: string
  status: 'pending' | 'parsing' | 'embedding' | 'ready' | 'failed'
  error: string | null
  client: string
  industry: string
  doc_type: string
  engagement_model: string
  year: number
  value_usd: number
  duration_months: number
  tech_stack: string[]
  summary: string
  page_count: number
  byte_size: number
  chunk_count: number
  created_at: string
}

export interface Stats {
  documents: number
  documents_ready: number
  documents_failed: number
  chunks: number
  vectors: number
  conversations: number
  questions_asked: number
  median_retrieval_ms: number | null
  mean_confidence: number | null
  low_confidence_rate: number | null
  embedding_backend: string
  llm: string
  vector_index: string
}

export interface EvalMetrics {
  cases: number
  mrr: number | null
  keyword_coverage: number | null
  mean_confidence: number | null
  ungrounded_sentences_total: number
  cases_with_invalid_citations: number
  median_retrieval_ms: number | null
  total_seconds: number
  [key: string]: number | null | undefined
}

export interface EvalDetail {
  question: string
  expect_doc: string
  retrieved_rank: number | null
  confidence: number
  intent: string
  filters: Record<string, unknown>
  top_documents: string[]
  keyword_coverage: number | null
  missing_keywords: string[]
  citations: number
  ungrounded_sentences: number
  answer_preview: string
}
