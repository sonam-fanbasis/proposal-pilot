import type { Citation, DocumentRecord, EvalDetail, EvalMetrics, Mode, RetrievalDebug, Stats } from './types'

async function json<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? `${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  documents: () => json<DocumentRecord[]>('/api/documents'),
  deleteDocument: (id: number) => json<{ deleted: number }>(`/api/documents/${id}`, { method: 'DELETE' }),
  chunks: (id: number) =>
    json<{ id: number; ordinal: number; section_path: string; page_from: number; char_count: number; text: string; context_header: string }[]>(
      `/api/documents/${id}/chunks`,
    ),
  stats: () => json<Stats>('/api/stats'),
  facets: () => json<{ industries: { value: string; count: number }[]; tech: { value: string; count: number }[] }>('/api/facets'),
  runEval: () => json<{ metrics: EvalMetrics; details: EvalDetail[] }>('/api/eval/run', { method: 'POST', body: JSON.stringify({ k: 6 }) }),

  async upload(file: File) {
    const form = new FormData()
    form.append('file', file)
    const response = await fetch('/api/documents', { method: 'POST', body: form })
    if (!response.ok) {
      const body = await response.json().catch(() => ({}))
      throw new Error(body.detail ?? `Upload failed (${response.status})`)
    }
    return response.json() as Promise<{ document_id: number; filename: string; message: string }>
  },
}

export interface StreamHandlers {
  onMeta: (meta: { conversation_id: number; confidence: number; citations: Citation[]; retrieval: RetrievalDebug }) => void
  onToken: (text: string) => void
  onDone: (done: {
    message_id: number
    final_text?: string
    citations: Citation[]
    gaps: string[]
    ungrounded_sentences: number
    invalid_citations: number[]
    total_ms: number
  }) => void
  onError: (message: string) => void
}

/**
 * Consumes the SSE stream from POST /api/ask/stream.
 *
 * EventSource cannot issue a POST, so the stream is read from a fetch body and
 * the SSE framing is parsed by hand. Events are separated by a blank line, and
 * a partial event left in the buffer between reads must be carried forward --
 * chunk boundaries fall in the middle of events routinely.
 */
export async function askStream(
  body: { question: string; mode: Mode; conversation_id: number | null; document_ids: number[] },
  handlers: StreamHandlers,
  signal?: AbortSignal,
) {
  const response = await fetch('/api/ask/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!response.ok || !response.body) {
    handlers.onError(`Request failed (${response.status})`)
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let split = buffer.indexOf('\n\n')
    while (split !== -1) {
      const raw = buffer.slice(0, split)
      buffer = buffer.slice(split + 2)
      split = buffer.indexOf('\n\n')

      let event = 'message'
      const dataLines: string[] = []
      for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
      }
      if (!dataLines.length) continue

      let payload: any
      try {
        payload = JSON.parse(dataLines.join('\n'))
      } catch {
        continue
      }

      if (event === 'meta') handlers.onMeta(payload)
      else if (event === 'token') handlers.onToken(payload.text ?? '')
      else if (event === 'done') handlers.onDone(payload)
      else if (event === 'error') handlers.onError(payload.message ?? 'Stream error')
    }
  }
}
