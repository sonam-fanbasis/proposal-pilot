import { useEffect, useState } from 'react'
import { api } from '../api'
import type { DocumentRecord } from '../types'

interface Chunk {
  id: number
  ordinal: number
  section_path: string
  page_from: number
  char_count: number
  text: string
  context_header: string
}

/**
 * Shows exactly what was indexed for a document.
 *
 * Chunking is where most RAG quality is won or lost and it is normally
 * invisible, so this exposes the real stored chunks -- including the generated
 * context header that gets embedded with each one.
 */
export function DocumentModal({ doc, onClose }: { doc: DocumentRecord; onClose: () => void }) {
  const [chunks, setChunks] = useState<Chunk[] | null>(null)

  useEffect(() => {
    api.chunks(doc.id).then(setChunks).catch(() => setChunks([]))
  }, [doc.id])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,.62)', zIndex: 40,
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24,
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: 'var(--bg)', border: '1px solid var(--border-strong)', borderRadius: 10,
          width: 'min(880px, 100%)', maxHeight: '86vh', display: 'flex', flexDirection: 'column',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="section-head" style={{ justifyContent: 'space-between' }}>
          <span>{doc.title}</span>
          <button className="icon-btn" onClick={onClose}>✕</button>
        </div>

        <div className="scroll" style={{ padding: 16 }}>
          <div className="metric-grid" style={{ marginBottom: 18 }}>
            <Meta label="Client" value={doc.client || '—'} />
            <Meta label="Industry" value={doc.industry || '—'} />
            <Meta label="Type" value={doc.doc_type?.replace('_', ' ') || '—'} />
            <Meta label="Model" value={doc.engagement_model?.replace(/_/g, ' ') || '—'} />
            <Meta label="Value" value={doc.value_usd ? `$${doc.value_usd.toLocaleString()}` : '—'} />
            <Meta label="Duration" value={doc.duration_months ? `${doc.duration_months} mo` : '—'} />
          </div>

          {doc.summary && <p className="lede" style={{ marginBottom: 18 }}>{doc.summary}</p>}

          <h3 style={{ marginTop: 0 }}>
            Indexed chunks {chunks ? `(${chunks.length})` : ''}
          </h3>
          <p className="lede" style={{ fontSize: 12.5, marginBottom: 12 }}>
            The blue line is the context header prepended to each chunk before embedding — it is
            what makes an isolated passage retrievable by client, industry and section.
          </p>

          {!chunks && <div className="empty">Loading…</div>}
          {chunks?.map((chunk) => (
            <div key={chunk.id} className="chunk">
              <div className="chunk-head">
                <span>#{chunk.ordinal}</span>
                <span>{chunk.char_count} chars</span>
                {chunk.page_from > 0 && <span>p.{chunk.page_from}</span>}
                <span style={{ marginLeft: 'auto' }}>{chunk.section_path || 'body'}</span>
              </div>
              {chunk.context_header && <div className="chunk-ctx">{chunk.context_header}</div>}
              <div className="chunk-text">{chunk.text}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div style={{ fontSize: 13, marginTop: 5, textTransform: 'capitalize' }}>{value}</div>
    </div>
  )
}
