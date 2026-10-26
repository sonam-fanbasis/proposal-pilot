import type { Citation } from '../types'

export type Shape = 'narrative' | 'long_structured' | 'visual' | ''
export type ContentType = 'text' | 'table' | 'figure' | 'footnote' | 'note'

export interface ReportProfile {
  stats?: Record<string, number>
  chunk_types?: Partial<Record<ContentType, number>>
  sections?: number
  chunks?: number
  strategy?: string[]
  embedder?: string
  timings?: Record<string, number>
}

export interface Report {
  id: number
  filename: string
  title: string
  byte_size: number
  page_count: number
  status: 'pending' | 'processing' | 'ready' | 'failed'
  stage: string
  stage_detail: string
  error: string | null
  shape: Shape
  profile: ReportProfile
  ingest_ms: number
  chunk_count: number
  created_at: string
}

export interface Evidence extends Citation {
  content_type: ContentType
  note_text: string
  routed: boolean
  rerank: number | null
}

export interface TraceHit {
  chunk_id: number
  section: string
  type: ContentType
  pages: [number, number]
  routed: boolean
  dense_rank: number | null
  keyword_rank: number | null
  rerank: number | null
}

export interface Trace {
  standalone_question: string
  route: { kind: 'explicit' | 'title' | 'none'; reason: string; sections: string[] }
  quantitative: boolean
  candidates: number
  context_chars: number
  timings: Record<string, number>
  hits: TraceHit[]
  expanded_neighbours: number
  embedder: string
  reranker: string
  llm: string
}

export interface ReportMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  evidence: Evidence[]
  citations: Evidence[]
  confidence: number
  trace?: Trace
  stage?: 'retrieving' | 'generating'
  ungrounded: number
  totalMs?: number
  streaming?: boolean
}

export interface SectionRow {
  ordinal: number
  parent: number | null
  level: number
  title: string
  path: string
  page_from: number
  page_to: number
  chunk_count: number
}

export interface BenchmarkDetail {
  report: string
  use_case: string
  question: string
  reference: string
  route: string
  routed_to: string[]
  retrieval_hit: boolean
  first_relevant_rank: number | null
  scope_precision: number | null
  content_types: string[]
  confidence: number
  answer: string
  citations: number
  missing_keywords: string[]
  passed: boolean
  latency_ms: number
}

export interface BenchmarkResult {
  metrics: {
    cases: number
    passed: number
    retrieval_hit_rate: number
    mrr: number
    scope_precision: number | null
    answer_keyword_rate: number | null
    ungrounded_sentences: number
    seconds: number
  }
  details: BenchmarkDetail[]
}
