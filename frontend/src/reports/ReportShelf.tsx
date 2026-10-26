import { useRef, useState } from 'react'
import { SHAPES, STAGES, TYPE_ORDER, TYPES } from './meta'
import type { Report } from './types'

interface Props {
  reports: Report[]
  activeId: number | null
  onSelect: (id: number) => void
  onUpload: (files: FileList) => void
  onDelete: (id: number) => void
}

export function ReportShelf({ reports, activeId, onSelect, onUpload, onDelete }: Props) {
  const [dragOver, setDragOver] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  return (
    <aside className="rl-shelf">
      <div className="section-head">
        Reports
        <span className="count">{reports.length ? `${reports.length} loaded` : ''}</span>
      </div>

      <button
        className="rl-drop"
        data-over={dragOver}
        onClick={() => input.current?.click()}
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
        <span className="rl-drop-icon" aria-hidden>
          ⤓
        </span>
        <span>
          <strong>Upload a report</strong>
          <small>PDF · the pipeline adapts to its shape</small>
        </span>
        <input
          ref={input}
          type="file"
          accept=".pdf"
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files?.length) onUpload(e.target.files)
            e.target.value = ''
          }}
        />
      </button>

      <div className="scroll">
        {reports.length === 0 && (
          <div className="empty">
            No reports yet. Upload an annual report, a long economic outlook or a product
            fact sheet — each is indexed with a recipe chosen for its structure.
          </div>
        )}
        {reports.map((report) => (
          <ReportCard
            key={report.id}
            report={report}
            active={report.id === activeId}
            onSelect={() => onSelect(report.id)}
            onDelete={() => onDelete(report.id)}
          />
        ))}
      </div>
    </aside>
  )
}

function ReportCard({ report, active, onSelect, onDelete }: { report: Report; active: boolean; onSelect: () => void; onDelete: () => void }) {
  const busy = report.status === 'pending' || report.status === 'processing'
  const types = report.profile.chunk_types ?? {}
  const total = TYPE_ORDER.reduce((sum, t) => sum + (types[t] ?? 0), 0)

  return (
    <div className="rl-card" data-active={active} data-status={report.status} onClick={onSelect} role="button" tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && onSelect()}>
      <div className="rl-card-top">
        {report.shape && <span className="rl-shape" data-shape={report.shape}>{SHAPES[report.shape].label}</span>}
        <button
          className="icon-btn rl-card-del"
          title="Remove report"
          onClick={(e) => {
            e.stopPropagation()
            onDelete()
          }}
        >
          ✕
        </button>
      </div>
      <div className="rl-card-title">{report.title}</div>

      {busy && (
        <div className="rl-stages">
          {STAGES.map((stage) => {
            const index = STAGES.indexOf(report.stage as (typeof STAGES)[number])
            const mine = STAGES.indexOf(stage)
            const state = index === -1 ? 'todo' : mine < index ? 'done' : mine === index ? 'now' : 'todo'
            return <span key={stage} className="rl-stage" data-state={state}>{stage}</span>
          })}
          <div className="rl-stage-detail">
            <span className="spinner" /> {report.stage_detail || 'Queued'}
          </div>
        </div>
      )}

      {report.status === 'failed' && <div className="rl-card-error">{report.error}</div>}

      {report.status === 'ready' && (
        <>
          <div className="rl-card-meta">
            {report.page_count} pages · {report.profile.sections ?? 0} sections · {report.chunk_count} chunks
          </div>
          {total > 0 && (
            <div className="rl-typebar" title={TYPE_ORDER.map((t) => `${TYPES[t].label}: ${types[t] ?? 0}`).join('\n')}>
              {TYPE_ORDER.map((t) =>
                types[t] ? <span key={t} data-type={t} style={{ flexGrow: types[t] }} /> : null,
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
