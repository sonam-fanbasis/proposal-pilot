import type { BenchmarkResult, Evidence, Report, SectionRow, Trace } from './types'

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

export const reportApi = {
  list: () => json<Report[]>('/api/reports'),
  remove: (id: number) => json<{ deleted: number }>(`/api/reports/${id}`, { method: 'DELETE' }),
  sections: (id: number) => json<SectionRow[]>(`/api/reports/${id}/sections`),
  samples: (id: number) => json<{ benchmark: string[]; suggested: string[] }>(`/api/reports/${id}/samples`),
  benchmark: (reportIds: number[]) =>
    json<BenchmarkResult>('/api/reports/eval', {
      method: 'POST',
      body: JSON.stringify({ report_ids: reportIds, generate_answers: true }),
    }),
  pageUrl: (id: number, page: number) => `/api/reports/${id}/pages/${page}.png`,

  async upload(file: File) {
    const form = new FormData()
    form.append('file', file)
    const response = await fetch('/api/reports', { method: 'POST', body: form })
    if (!response.ok) {
      const body = await response.json().catch(() => ({}))
      throw new Error(body.detail ?? `Upload failed (${response.status})`)
    }
    return response.json() as Promise<{ report_id: number }>
  },
}

export interface ReportStreamHandlers {
  onStage: (stage: 'retrieving' | 'generating') => void
  onMeta: (meta: { confidence: number; evidence: Evidence[]; trace: Trace }) => void
  onToken: (text: string) => void
  onDone: (done: {
    final_text: string
    citations: Evidence[]
    ungrounded_sentences: number
    invalid_citations: number[]
    total_ms: number
  }) => void
  onError: (message: string) => void
}

/** SSE over a POST body: framing is parsed by hand, partial events carried forward. */
export async function askReport(
  body: { question: string; report_ids: number[]; history: { role: string; content: string }[] },
  handlers: ReportStreamHandlers,
  signal?: AbortSignal,
) {
  const response = await fetch('/api/reports/ask/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!response.ok || !response.body) {
    const detail = await response.json().catch(() => ({}))
    handlers.onError(detail.detail ? JSON.stringify(detail.detail) : `Request failed (${response.status})`)
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
      const data: string[] = []
      for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data.push(line.slice(5).trim())
      }
      if (!data.length) continue
      let payload: any
      try {
        payload = JSON.parse(data.join('\n'))
      } catch {
        continue
      }
      if (event === 'stage') handlers.onStage(payload.stage)
      else if (event === 'meta') handlers.onMeta(payload)
      else if (event === 'token') handlers.onToken(payload.text ?? '')
      else if (event === 'done') handlers.onDone(payload)
      else if (event === 'error') handlers.onError(payload.message ?? 'Stream error')
    }
  }
}
