import { useRef, useState } from 'react'
import type { DocumentRecord } from '../types'

interface Props {
  documents: DocumentRecord[]
  selected: number[]
  onToggleSelect: (id: number) => void
  onUpload: (files: FileList) => void
  onDelete: (id: number) => void
  onInspect: (doc: DocumentRecord) => void
}

const money = (value: number) =>
  value >= 1_000_000 ? `$${(value / 1_000_000).toFixed(1)}M` : `$${Math.round(value / 1000)}k`

const STATUS_LABEL: Record<string, string> = {
  pending: 'Queued',
  parsing: 'Parsing document',
  embedding: 'Generating embeddings',
  ready: '',
  failed: 'Failed',
}

export function DocumentLibrary({ documents, selected, onToggleSelect, onUpload, onDelete, onInspect }: Props) {
  const [dragOver, setDragOver] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  return (
    <aside className="sidebar">
      <div className="section-head">
        Library
        <span className="count">{documents.length ? `${documents.length} docs` : ''}</span>
      </div>

      <div
        className="dropzone"
        data-over={dragOver}
        onClick={() => fileInput.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          if (e.dataTransfer.files.length) onUpload(e.dataTransfer.files)
        }}
      >
        <strong>Add documents</strong> or drop them here
        <span className="dropzone-hint">PDF, DOCX, MD, TXT</span>
        <input
          ref={fileInput}
          type="file"
          multiple
          accept=".pdf,.docx,.md,.markdown,.txt"
          style={{ display: 'none' }}
          onChange={(e) => {
            if (e.target.files?.length) onUpload(e.target.files)
            e.target.value = ''
          }}
        />
      </div>

      <div className="scroll">
        {documents.length === 0 ? (
          <div className="empty">
            No documents yet.
            <br />
            Upload past proposals, case studies or SOWs — or run <code>seed.py</code> to load the samples.
          </div>
        ) : (
          documents.map((doc) => {
            const busy = ['pending', 'parsing', 'embedding'].includes(doc.status)
            return (
              <div
                key={doc.id}
                className="doc"
                data-selected={selected.includes(doc.id)}
                onClick={() => doc.status === 'ready' && onToggleSelect(doc.id)}
                title={doc.status === 'ready' ? 'Click to scope questions to this document' : doc.error ?? ''}
              >
                <div className="doc-title">
                  <span>{doc.title}</span>
                  <div className="doc-actions" onClick={(e) => e.stopPropagation()}>
                    <button className="icon-btn" onClick={() => onInspect(doc)} title="Inspect chunks">
                      ⓘ
                    </button>
                    <button className="icon-btn" onClick={() => onDelete(doc.id)} title="Remove">
                      ✕
                    </button>
                  </div>
                </div>

                {doc.status === 'ready' && (
                  <div className="tags">
                    {doc.industry && doc.industry !== 'other' && (
                      <span className="tag" data-kind="industry">
                        {doc.industry.replace('_', ' ')}
                      </span>
                    )}
                    {doc.doc_type && doc.doc_type !== 'other' && (
                      <span className="tag">{doc.doc_type.replace('_', ' ')}</span>
                    )}
                    {doc.engagement_model && doc.engagement_model !== 'other' && (
                      <span className="tag" data-kind="model">
                        {doc.engagement_model.replace(/_/g, ' ')}
                      </span>
                    )}
                    {doc.value_usd > 0 && (
                      <span className="tag" data-kind="money">
                        {money(doc.value_usd)}
                      </span>
                    )}
                    {doc.year > 0 && <span className="tag">{doc.year}</span>}
                    {doc.tech_stack.slice(0, 3).map((tech) => (
                      <span key={tech} className="tag" data-kind="tech">
                        {tech}
                      </span>
                    ))}
                    {doc.tech_stack.length > 3 && (
                      <span className="tag" data-kind="tech">
                        +{doc.tech_stack.length - 3}
                      </span>
                    )}
                  </div>
                )}

                {doc.status !== 'ready' && (
                  <div className="status-line" data-status={doc.status}>
                    {busy && <span className="spinner" />}
                    {STATUS_LABEL[doc.status] ?? doc.status}
                    {doc.status === 'failed' && doc.error && `: ${doc.error.slice(0, 90)}`}
                  </div>
                )}

                {doc.status === 'ready' && (
                  <div className="status-line" data-status="ready">
                    {doc.chunk_count} chunks
                    {doc.page_count > 1 && ` · ${doc.page_count} pages`}
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>
    </aside>
  )
}
