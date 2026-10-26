import { useState } from 'react'
import { reportApi } from './api'
import type { BenchmarkResult, Report } from './types'

const pct = (value: number | null | undefined) => (value === null || value === undefined ? '—' : `${Math.round(value * 100)}%`)

/** Runs the use-case sample questions and scores retrieval, scope and answer separately. */
export function Benchmark({ reports, onError }: { reports: Report[]; onError: (message: string) => void }) {
  const [result, setResult] = useState<BenchmarkResult | null>(null)
  const [running, setRunning] = useState(false)
  const [open, setOpen] = useState<number | null>(null)
  const ready = reports.filter((r) => r.status === 'ready')

  const run = async () => {
    setRunning(true)
    try {
      setResult(await reportApi.benchmark(ready.map((r) => r.id)))
    } catch (error) {
      onError(error instanceof Error ? error.message : 'Benchmark failed')
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="rl-bench">
      <div className="rl-bench-intro">
        <div>
          <h2>Sample-question benchmark</h2>
          <p>
            Runs every use-case question that matches a loaded report and scores three things
            separately: whether retrieval reached the page with the answer, whether the evidence
            stayed inside the section the question names, and whether the answer contains the
            required figures.
          </p>
        </div>
        <button className="btn" onClick={run} disabled={running || ready.length === 0}>
          {running ? <><span className="spinner" /> Running…</> : result ? 'Run again' : 'Run benchmark'}
        </button>
      </div>

      {running && !result && <div className="empty">Asking each question end-to-end. This takes about 20–40 seconds per question on the Gemini free tier.</div>}

      {result && (
        <>
          <div className="metric-grid">
            <Metric label="Passed" value={`${result.metrics.passed}/${result.metrics.cases}`} />
            <Metric label="Retrieval hit" value={pct(result.metrics.retrieval_hit_rate)} note="answer page retrieved" />
            <Metric label="MRR" value={result.metrics.mrr.toFixed(2)} note="rank of first relevant chunk" />
            <Metric label="Scope precision" value={pct(result.metrics.scope_precision)} note="evidence inside named section" />
            <Metric label="Answer figures" value={pct(result.metrics.answer_keyword_rate)} note="required facts present" />
            <Metric label="Uncited claims" value={String(result.metrics.ungrounded_sentences)} />
          </div>

          <div className="rl-bench-list">
            {result.details.map((d, i) => (
              <div key={d.question} className="rl-bench-row" data-pass={d.passed}>
                <button className="rl-bench-head" onClick={() => setOpen(open === i ? null : i)}>
                  <span className="rl-bench-mark">{d.passed ? '✓' : '✕'}</span>
                  <span className="rl-bench-q">
                    {d.question}
                    <small>{d.use_case}</small>
                  </span>
                  <span className="rl-bench-tags">
                    {d.route !== 'none' && <span className="rl-flag">routed</span>}
                    {d.content_types.map((t) => <span key={t} className="rl-type" data-type={t}>{t}</span>)}
                  </span>
                </button>
                {open === i && (
                  <div className="rl-bench-body">
                    <div className="rl-bench-answer">{d.answer}</div>
                    <div className="rl-bench-ref"><strong>Reference</strong> {d.reference}</div>
                    <dl className="kv">
                      <dt>route</dt><dd>{d.route}{d.routed_to.length ? ` → ${d.routed_to.join(', ')}` : ''}</dd>
                      <dt>first relevant rank</dt><dd>{d.first_relevant_rank ?? 'not retrieved'}</dd>
                      <dt>scope precision</dt><dd>{pct(d.scope_precision)}</dd>
                      <dt>missing figures</dt><dd>{d.missing_keywords.length ? d.missing_keywords.join(', ') : 'none'}</dd>
                      <dt>latency</dt><dd>{(d.latency_ms / 1000).toFixed(1)}s</dd>
                    </dl>
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

function Metric({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {note && <div className="metric-note">{note}</div>}
    </div>
  )
}
