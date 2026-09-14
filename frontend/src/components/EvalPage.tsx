import { useState } from 'react'
import { api } from '../api'
import type { EvalDetail, EvalMetrics } from '../types'

export function EvalPage({ onError }: { onError: (message: string) => void }) {
  const [running, setRunning] = useState(false)
  const [metrics, setMetrics] = useState<EvalMetrics | null>(null)
  const [details, setDetails] = useState<EvalDetail[]>([])

  const run = async () => {
    setRunning(true)
    try {
      const result = await api.runEval()
      setMetrics(result.metrics)
      setDetails(result.details)
    } catch (error) {
      onError(error instanceof Error ? error.message : 'Evaluation failed')
    } finally {
      setRunning(false)
    }
  }

  const recall = metrics ? (metrics['recall_at_6'] as number | null) : null

  return (
    <div className="page">
      <h2>Evaluation</h2>
      <p className="lede">
        A golden set of questions with a known correct source document. Recall tells you whether the
        right document was retrieved at all — if it was not, no prompt change can fix the answer, so
        this is what separates a retrieval bug from a generation bug. MRR tells you how highly it
        ranked. Grounding checks that every citation in the answer points at evidence that really
        exists.
      </p>

      <button className="btn" onClick={run} disabled={running}>
        {running ? 'Running…' : 'Run evaluation'}
      </button>

      {metrics && (
        <>
          <div className="metric-grid" style={{ marginTop: 20 }}>
            <Metric label="Recall@6" value={fmt(recall)} tone={tone(recall, 0.9, 0.7)} note={`${metrics.cases} cases`} />
            <Metric label="MRR" value={fmt(metrics.mrr)} tone={tone(metrics.mrr, 0.8, 0.5)} note="rank of correct doc" />
            <Metric label="Keyword coverage" value={fmt(metrics.keyword_coverage)} tone={tone(metrics.keyword_coverage, 0.85, 0.6)} note="expected facts present" />
            <Metric label="Mean confidence" value={fmt(metrics.mean_confidence)} />
            <Metric
              label="Ungrounded claims"
              value={metrics.ungrounded_sentences_total}
              tone={metrics.ungrounded_sentences_total === 0 ? 'good' : 'warn'}
              note="uncited factual sentences"
            />
            <Metric
              label="Invalid citations"
              value={metrics.cases_with_invalid_citations}
              tone={metrics.cases_with_invalid_citations === 0 ? 'good' : 'bad'}
              note="cases citing nonexistent evidence"
            />
          </div>

          <h3>Per-question results</h3>
          <table>
            <thead>
              <tr>
                <th>Question</th>
                <th style={{ width: 62 }}>Rank</th>
                <th style={{ width: 74 }}>Intent</th>
                <th style={{ width: 74 }}>Coverage</th>
                <th style={{ width: 74 }}>Conf.</th>
              </tr>
            </thead>
            <tbody>
              {details.map((detail, i) => (
                <tr key={i}>
                  <td>
                    {detail.question}
                    {detail.missing_keywords.length > 0 && (
                      <div style={{ color: 'var(--warn)', fontSize: 11.5, marginTop: 3 }}>
                        missing: {detail.missing_keywords.join(', ')}
                      </div>
                    )}
                    {Object.keys(detail.filters).length > 0 && (
                      <div style={{ color: 'var(--text-faint)', fontSize: 11, marginTop: 3, fontFamily: 'var(--mono)' }}>
                        {JSON.stringify(detail.filters)}
                      </div>
                    )}
                  </td>
                  <td className="num" style={{ color: rankColor(detail.retrieved_rank, detail.expect_doc) }}>
                    {detail.expect_doc ? (detail.retrieved_rank ?? 'miss') : 'n/a'}
                  </td>
                  <td style={{ color: 'var(--text-dim)' }}>{detail.intent}</td>
                  <td className="num">{detail.keyword_coverage === null ? '—' : `${Math.round(detail.keyword_coverage * 100)}%`}</td>
                  <td className="num">{detail.confidence.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  )
}

const fmt = (value: number | null | undefined) => (value === null || value === undefined ? '—' : value.toFixed(2))

const tone = (value: number | null | undefined, good: number, ok: number) =>
  value === null || value === undefined ? undefined : value >= good ? 'good' : value >= ok ? 'warn' : 'bad'

const rankColor = (rank: number | null, expected: string) => {
  if (!expected) return 'var(--text-faint)'
  if (rank === null) return 'var(--bad)'
  return rank <= 2 ? 'var(--good)' : 'var(--warn)'
}

function Metric({ label, value, note, tone }: { label: string; value: number | string; note?: string; tone?: string }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-value" data-tone={tone}>{value}</div>
      {note && <div className="metric-note">{note}</div>}
    </div>
  )
}
