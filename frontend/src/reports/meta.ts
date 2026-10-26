import type { ContentType, Shape } from './types'

export const SHAPES: Record<Exclude<Shape, ''>, { label: string; blurb: string }> = {
  narrative: { label: 'Narrative', blurb: 'Facts live in paragraphs — the baseline recipe.' },
  long_structured: { label: 'Long & structured', blurb: 'Chapters and sections — routed retrieval.' },
  visual: { label: 'Visual', blurb: 'Charts, tables and footnotes — vision + layout parsing.' },
}

export const TYPES: Record<ContentType, { label: string; short: string }> = {
  text: { label: 'Body text', short: 'TEXT' },
  table: { label: 'Table', short: 'TABLE' },
  figure: { label: 'Chart / figure', short: 'CHART' },
  note: { label: 'Figure note', short: 'NOTE' },
  footnote: { label: 'Footnote', short: 'FOOTNOTE' },
}

export const TYPE_ORDER: ContentType[] = ['text', 'table', 'figure', 'note', 'footnote']

export const STAGES = ['parsing', 'vision', 'chunking', 'embedding'] as const

export const pages = (from: number, to: number) => (from === to ? `p.${from}` : `pp.${from}–${to}`)

export const seconds = (ms: number) => (ms >= 10_000 ? `${Math.round(ms / 1000)}s` : `${(ms / 1000).toFixed(1)}s`)
