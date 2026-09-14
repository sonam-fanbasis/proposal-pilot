import { useEffect, useState } from 'react'
import { api } from '../api'
import type { Stats } from '../types'

export function StatsPage() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [facets, setFacets] = useState<{ industries: { value: string; count: number }[]; tech: { value: string; count: number }[] } | null>(null)

  useEffect(() => {
    api.stats().then(setStats).catch(() => {})
    api.facets().then(setFacets).catch(() => {})
  }, [])

  if (!stats) return <div className="page"><div className="empty">Loading…</div></div>

  return (
    <div className="page">
      <h2>Corpus & runtime</h2>
      <p className="lede">
        What is indexed, which backends are actually in use, and how retrieval has performed on real
        traffic. Every question asked through the app is logged, so these are measurements rather
        than estimates.
      </p>

      <div className="metric-grid">
        <Metric label="Documents" value={stats.documents_ready} note={stats.documents_failed ? `${stats.documents_failed} failed` : 'all indexed'} />
        <Metric label="Chunks" value={stats.chunks} note={`${stats.vectors} vectors`} />
        <Metric label="Questions" value={stats.questions_asked} note={`${stats.conversations} conversations`} />
        <Metric
          label="Median retrieval"
          value={stats.median_retrieval_ms === null ? '—' : `${stats.median_retrieval_ms}ms`}
          tone={stats.median_retrieval_ms !== null && stats.median_retrieval_ms < 150 ? 'good' : undefined}
        />
        <Metric
          label="Mean confidence"
          value={stats.mean_confidence === null ? '—' : stats.mean_confidence.toFixed(2)}
          note="0–1, blended similarity"
        />
        <Metric
          label="Low-confidence"
          value={stats.low_confidence_rate === null ? '—' : `${Math.round(stats.low_confidence_rate * 100)}%`}
          note="declined to answer"
          tone={stats.low_confidence_rate !== null && stats.low_confidence_rate > 0.4 ? 'warn' : undefined}
        />
      </div>

      <h3>Active backends</h3>
      <table>
        <tbody>
          <tr><th style={{ width: 190 }}>Language model</th><td>{stats.llm}</td></tr>
          <tr><th>Embeddings</th><td>{stats.embedding_backend === 'gemini' ? 'Gemini text-embedding-004 (768d, asymmetric)' : 'Offline hashing embedder (768d) — set GEMINI_API_KEY for semantic embeddings'}</td></tr>
          <tr><th>Vector index</th><td>{stats.vector_index}</td></tr>
        </tbody>
      </table>

      {facets && facets.industries.length > 0 && (
        <>
          <h3>Coverage by industry</h3>
          <table>
            <thead><tr><th>Industry</th><th style={{ width: 90 }}>Documents</th></tr></thead>
            <tbody>
              {facets.industries.map((row) => (
                <tr key={row.value}>
                  <td style={{ textTransform: 'capitalize' }}>{row.value.replace('_', ' ')}</td>
                  <td className="num">{row.count}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h3>Technology coverage</h3>
          <div className="tags" style={{ gap: 6 }}>
            {facets.tech.map((row) => (
              <span key={row.value} className="tag" data-kind="tech">
                {row.value} · {row.count}
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  )
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
