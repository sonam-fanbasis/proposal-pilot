import { useCallback, useEffect, useRef, useState } from 'react'
import { AnswerText } from '../components/AnswerText'
import { askReport, reportApi } from './api'
import { Benchmark } from './Benchmark'
import { EvidencePanel } from './EvidencePanel'
import { IndexProfile } from './IndexProfile'
import { pages, SHAPES, TYPES } from './meta'
import { ReportShelf } from './ReportShelf'
import type { Report, ReportMessage, Trace } from './types'
import './reportlab.css'

type View = 'ask' | 'profile' | 'benchmark'

interface Props {
  notify: (text: string, tone?: string) => void
}

export function ReportLab({ notify }: Props) {
  const [reports, setReports] = useState<Report[]>([])
  const [activeId, setActiveId] = useState<number | null>(null)
  const [view, setView] = useState<View>('ask')
  // Conversations are per report: switching reports must not carry one
  // report's history into another's follow-up rewriting.
  const [threads, setThreads] = useState<Record<number, ReportMessage[]>>({})
  const [samples, setSamples] = useState<{ benchmark: string[]; suggested: string[] }>({ benchmark: [], suggested: [] })
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [highlighted, setHighlighted] = useState<number | null>(null)
  const [preview, setPreview] = useState<{ page: number; reportId: number } | null>(null)
  const endRef = useRef<HTMLDivElement>(null)

  const active = reports.find((r) => r.id === activeId) ?? null
  const messages = activeId ? threads[activeId] ?? [] : []
  const lastAnswer = [...messages].reverse().find((m) => m.role === 'assistant')

  const refresh = useCallback(async () => {
    try {
      const list = await reportApi.list()
      setReports(list)
      setActiveId((current) => current ?? list.find((r) => r.status === 'ready')?.id ?? list[0]?.id ?? null)
    } catch {
      /* next poll retries */
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  useEffect(() => {
    if (!reports.some((r) => r.status === 'pending' || r.status === 'processing')) return
    const timer = window.setInterval(refresh, 1500)
    return () => window.clearInterval(timer)
  }, [reports, refresh])

  const activeReady = active?.status === 'ready'
  useEffect(() => {
    if (!activeId || !activeReady) return
    reportApi.samples(activeId).then(setSamples).catch(() => setSamples({ benchmark: [], suggested: [] }))
  }, [activeId, activeReady])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, lastAnswer?.content])

  const upload = async (files: FileList) => {
    for (const file of Array.from(files)) {
      try {
        const { report_id } = await reportApi.upload(file)
        setActiveId(report_id)
        setView('profile')
      } catch (error) {
        notify(error instanceof Error ? error.message : `Could not upload ${file.name}`, 'error')
      }
    }
    refresh()
  }

  const remove = async (id: number) => {
    try {
      await reportApi.remove(id)
      setThreads(({ [id]: _drop, ...rest }) => rest)
      if (id === activeId) setActiveId(null)
      setPreview(null)
      refresh()
    } catch (error) {
      notify(error instanceof Error ? error.message : 'Delete failed', 'error')
    }
  }

  const jump = (index: number) => {
    setHighlighted(index)
    document.getElementById(`rl-ev-${index}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    const item = lastAnswer?.evidence.find((e) => e.index === index)
    if (item) setPreview({ page: item.page_from, reportId: item.doc_id })
    window.setTimeout(() => setHighlighted(null), 1800)
  }

  const send = async (question: string) => {
    const text = question.trim()
    if (!text || busy || !active) return
    if (active.status !== 'ready') {
      notify('This report is still being indexed.', 'error')
      return
    }
    const reportId = active.id
    const history = (threads[reportId] ?? []).slice(-4).map((m) => ({ role: m.role, content: m.content.slice(0, 600) }))
    const assistantId = `a-${Date.now()}`
    const blank = { evidence: [], citations: [], confidence: 0, ungrounded: 0 }
    setThreads((t) => ({
      ...t,
      [reportId]: [
        ...(t[reportId] ?? []),
        { id: `u-${Date.now()}`, role: 'user', content: text, ...blank },
        { id: assistantId, role: 'assistant', content: '', streaming: true, stage: 'retrieving', ...blank },
      ],
    }))
    setInput('')
    setView('ask')
    setBusy(true)
    setPreview(null)

    const patch = (fn: (m: ReportMessage) => Partial<ReportMessage>) =>
      setThreads((t) => ({
        ...t,
        [reportId]: (t[reportId] ?? []).map((m) => (m.id === assistantId ? { ...m, ...fn(m) } : m)),
      }))

    try {
      await askReport({ question: text, report_ids: [reportId], history }, {
        onStage: (stage) => patch(() => ({ stage })),
        onMeta: (meta) => patch(() => ({ evidence: meta.evidence, confidence: meta.confidence, trace: meta.trace })),
        onToken: (token) => patch((m) => ({ content: m.content + token })),
        onDone: (done) =>
          patch(() => ({
            content: done.final_text,
            citations: done.citations,
            ungrounded: done.ungrounded_sentences,
            totalMs: done.total_ms,
            streaming: false,
            stage: undefined,
          })),
        onError: (message) => {
          patch(() => ({ streaming: false, stage: undefined }))
          notify(message, 'error')
        },
      })
    } catch (error) {
      notify(error instanceof Error ? error.message : 'Request failed', 'error')
      patch(() => ({ streaming: false, stage: undefined }))
    } finally {
      setBusy(false)
    }
  }

  const cited = new Set((lastAnswer?.citations ?? []).map((c) => c.index))

  return (
    <div className="layout rl">
      <ReportShelf reports={reports} activeId={activeId} onSelect={(id) => { setActiveId(id); setPreview(null) }} onUpload={upload} onDelete={remove} />

      <main className="panel rl-main">
        {active ? (
          <>
            <header className="rl-head">
              <div className="rl-head-text">
                <div className="rl-head-title">{active.title}</div>
                <div className="rl-head-sub">
                  {active.shape && <span className="rl-shape" data-shape={active.shape}>{SHAPES[active.shape].label}</span>}
                  {active.filename} · {active.page_count || '…'} pages
                </div>
              </div>
              <nav className="rl-views">
                {(['ask', 'profile', 'benchmark'] as View[]).map((v) => (
                  <button key={v} data-active={view === v} onClick={() => setView(v)}>
                    {v === 'ask' ? 'Ask' : v === 'profile' ? 'How it was indexed' : 'Benchmark'}
                  </button>
                ))}
              </nav>
            </header>

            {view === 'profile' && (
              <div className="scroll">
                {active.status === 'ready' ? <IndexProfile report={active} /> : <Indexing report={active} />}
              </div>
            )}

            {view === 'benchmark' && (
              <div className="scroll"><Benchmark reports={reports} onError={(m) => notify(m, 'error')} /></div>
            )}

            {view === 'ask' && (
              <>
                <div className="scroll rl-thread">
                  {messages.length === 0 && (
                    <div className="rl-intro">
                      <h2>Ask this report</h2>
                      <p>
                        Answers are built only from retrieved passages, tables, chart transcriptions and
                        footnotes, and every claim links to its page.
                      </p>
                      {samples.benchmark.length > 0 && (
                        <>
                          <div className="rl-intro-label">Use-case sample questions</div>
                          <div className="rl-chips">
                            {samples.benchmark.map((q) => <button key={q} className="rl-chip" data-primary onClick={() => send(q)}>{q}</button>)}
                          </div>
                        </>
                      )}
                      {samples.suggested.length > 0 && (
                        <>
                          <div className="rl-intro-label">Also try</div>
                          <div className="rl-chips">
                            {samples.suggested.map((q) => <button key={q} className="rl-chip" onClick={() => send(q)}>{q}</button>)}
                          </div>
                        </>
                      )}
                    </div>
                  )}

                  {messages.map((m) =>
                    m.role === 'user' ? (
                      <div key={m.id} className="rl-q">{m.content}</div>
                    ) : (
                      <Answer key={m.id} message={m} onCite={jump} />
                    ),
                  )}
                  <div ref={endRef} />
                </div>

                <div className="composer">
                  <div className="composer-row">
                    <textarea
                      value={input}
                      rows={1}
                      placeholder={active.status === 'ready' ? `Ask about ${active.title}…` : 'Indexing… you can ask once it is ready'}
                      disabled={active.status !== 'ready'}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault()
                          send(input)
                        }
                      }}
                    />
                    <button className="send" onClick={() => send(input)} disabled={busy || !input.trim() || active.status !== 'ready'}>
                      {busy ? '…' : 'Ask'}
                    </button>
                  </div>
                </div>
              </>
            )}
          </>
        ) : (
          <div className="rl-intro rl-intro-empty">
            <h2>Report Lab</h2>
            <p>
              Upload a PDF report. ProposalPilot reads its layout — chapters, tables, charts and
              footnotes — picks an indexing recipe for its shape, and answers questions with
              page-level citations.
            </p>
          </div>
        )}
      </main>

      <EvidencePanel
        report={active}
        evidence={lastAnswer?.evidence ?? []}
        cited={cited}
        pending={Boolean(lastAnswer?.streaming)}
        highlighted={highlighted}
        preview={preview}
        onPreview={setPreview}
      />
    </div>
  )
}

function Indexing({ report }: { report: Report }) {
  if (report.status === 'failed') return <div className="rl-intro"><h2>Indexing failed</h2><p>{report.error}</p></div>
  return (
    <div className="rl-intro">
      <h2><span className="spinner" /> Indexing {report.filename}</h2>
      <p>{report.stage_detail || 'Queued'} — the recipe and parsing results appear here when it finishes.</p>
    </div>
  )
}

function Answer({ message, onCite }: { message: ReportMessage; onCite: (index: number) => void }) {
  const trace = message.trace
  return (
    <div className="rl-a">
      {trace && trace.route.kind !== 'none' && (
        <div className="rl-route">
          <span className="rl-route-kind">{trace.route.kind === 'explicit' ? 'Scoped to' : 'Focused on'}</span>
          {trace.route.sections.slice(0, 3).map((s) => <span key={s} className="rl-route-sec">{s}</span>)}
        </div>
      )}

      {message.content ? (
        <AnswerText text={message.content} citations={message.citations.length ? message.citations : message.evidence} onCitationClick={onCite} />
      ) : (
        <div className="rl-pending">
          <span className="spinner" />
          {message.stage === 'generating'
            ? `Writing from ${message.evidence.length} pieces of evidence…`
            : 'Routing, searching and reranking…'}
        </div>
      )}

      {!message.streaming && message.content && (
        <div className="answer-meta">
          <span className="pill" data-tone={message.confidence >= 0.5 ? 'good' : 'warn'}>
            <span className="dot" /> evidence {message.confidence.toFixed(2)}
          </span>
          {[...new Set(message.citations.map((c) => c.content_type))].map((t) => (
            <span key={t} className="rl-type" data-type={t}>{TYPES[t]?.short ?? t}</span>
          ))}
          {message.citations.length > 0 && (
            <span className="pill">
              {[...new Set(message.citations.map((c) => pages(c.page_from, c.page_to)))].slice(0, 4).join(', ')}
            </span>
          )}
          {message.ungrounded > 0 && <span className="pill" data-tone="warn"><span className="dot" />{message.ungrounded} uncited</span>}
          {message.totalMs !== undefined && <span className="pill">{(message.totalMs / 1000).toFixed(1)}s</span>}
        </div>
      )}

      {trace && !message.streaming && <TraceView trace={trace} />}
    </div>
  )
}

function TraceView({ trace }: { trace: Trace }) {
  const steps: [string, number][] = [
    ['rewrite', trace.timings.rewrite_ms ?? 0],
    ['search', trace.timings.search_ms ?? 0],
    ['rerank', trace.timings.rerank_ms ?? 0],
    ['context', trace.timings.context_ms ?? 0],
  ]
  const total = Math.max(1, steps.reduce((s, [, ms]) => s + ms, 0))
  return (
    <details className="inspector">
      <summary>Retrieval trace</summary>
      <div className="inspector-body">
        <div className="rl-timeline">
          {steps.map(([name, ms]) => (
            <span key={name} data-step={name} style={{ flexGrow: Math.max(ms, total * 0.04) }} title={`${name}: ${ms}ms`}>
              {name} {ms}ms
            </span>
          ))}
        </div>
        <dl className="kv">
          {trace.standalone_question && <><dt>standalone question</dt><dd>{trace.standalone_question}</dd></>}
          <dt>routing</dt>
          <dd>{trace.route.kind === 'none' ? 'whole report (no section named)' : `${trace.route.reason}: ${trace.route.sections.join(', ')}`}</dd>
          <dt>quantitative</dt><dd>{trace.quantitative ? 'yes — a table/chart is guaranteed a slot' : 'no'}</dd>
          <dt>candidates fused</dt><dd>{trace.candidates} (dense + BM25{trace.route.kind !== 'none' ? ', global + routed' : ''})</dd>
          <dt>neighbours added</dt><dd>{trace.expanded_neighbours} · {trace.context_chars.toLocaleString()} chars of context</dd>
          <dt>models</dt><dd>{trace.embedder} · {trace.reranker} · {trace.llm}</dd>
        </dl>
        <table className="rl-trace-table">
          <thead><tr><th>#</th><th>type</th><th>section</th><th>page</th><th>dense</th><th>bm25</th><th>rerank</th></tr></thead>
          <tbody>
            {trace.hits.map((h, i) => (
              <tr key={h.chunk_id} data-routed={h.routed}>
                <td>{i + 1}</td>
                <td>{h.type}</td>
                <td title={h.section}>{h.section.split(' > ').slice(-1)[0]}</td>
                <td>{pages(h.pages[0], h.pages[1])}</td>
                <td>{h.dense_rank ?? '—'}</td>
                <td>{h.keyword_rank ?? '—'}</td>
                <td>{h.rerank ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  )
}
