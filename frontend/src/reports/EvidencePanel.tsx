import { useEffect, useMemo, useState } from 'react'
import { reportApi } from './api'
import { pages, TYPES } from './meta'
import type { Evidence, Report, SectionRow } from './types'

interface Props {
  report: Report | null
  evidence: Evidence[]
  cited: Set<number>
  pending: boolean
  highlighted: number | null
  preview: { page: number; reportId: number } | null
  onPreview: (preview: { page: number; reportId: number } | null) => void
}

type Tab = 'evidence' | 'map'

export function EvidencePanel({ report, evidence, cited, pending, highlighted, preview, onPreview }: Props) {
  const [tab, setTab] = useState<Tab>('evidence')

  return (
    <aside className="rl-evidence">
      <div className="rl-panel-tabs">
        <button data-active={tab === 'evidence'} onClick={() => setTab('evidence')}>
          Evidence {evidence.length > 0 && <span>{evidence.length}</span>}
        </button>
        <button data-active={tab === 'map'} onClick={() => setTab('map')}>
          Section map
        </button>
      </div>

      {preview && (
        <PagePreview
          key={`${preview.reportId}-${preview.page}`}
          reportId={preview.reportId}
          page={preview.page}
          pageCount={report?.page_count ?? preview.page}
          onChange={(page) => onPreview({ ...preview, page })}
          onClose={() => onPreview(null)}
        />
      )}

      <div className="scroll">
        {tab === 'evidence' ? (
          evidence.length === 0 ? (
            <div className="empty">
              Ask a question to see which passages, tables, charts and footnotes were retrieved —
              and which of them the answer actually cites.
            </div>
          ) : (
            evidence.map((item) => (
              <EvidenceCard
                key={item.chunk_id}
                item={item}
                cited={pending ? null : cited.has(item.index)}
                highlighted={highlighted === item.index}
                onOpenPage={() => onPreview({ page: item.page_from, reportId: item.doc_id })}
              />
            ))
          )
        ) : report ? (
          <SectionMap report={report} usedPaths={new Set(evidence.filter((e) => cited.has(e.index)).map((e) => e.section_path))}
            onOpenPage={(page) => onPreview({ page, reportId: report.id })} />
        ) : (
          <div className="empty">Select a report to see its structure.</div>
        )}
      </div>
    </aside>
  )
}

function EvidenceCard({ item, cited, highlighted, onOpenPage }: { item: Evidence; cited: boolean | null; highlighted: boolean; onOpenPage: () => void }) {
  const crumbs = item.section_path.split(' > ')
  return (
    <div id={`rl-ev-${item.index}`} className="rl-ev" data-cited={cited ?? 'pending'} data-highlight={highlighted}>
      <div className="rl-ev-head">
        <span className="rl-ev-index">{item.index}</span>
        <span className="rl-type" data-type={item.content_type}>{TYPES[item.content_type]?.short ?? item.content_type}</span>
        {item.routed && <span className="rl-flag" title="Inside the section the question was routed to">in scope</span>}
        <button className="rl-ev-page" onClick={onOpenPage} title="Show this page">
          {pages(item.page_from, item.page_to)} ↗
        </button>
      </div>
      <div className="rl-ev-path" title={item.section_path}>
        {crumbs.length > 2 ? `… › ${crumbs.slice(-2).join(' › ')}` : crumbs.join(' › ')}
      </div>
      <div className="rl-ev-snippet" data-type={item.content_type}>{item.snippet}</div>
      {item.note_text && <div className="rl-ev-note">{item.note_text}</div>}
      <div className="rl-ev-foot">
        {cited === null ? 'retrieved' : cited ? 'cited in answer' : 'retrieved, not cited'}
        {item.rerank !== null && <span>rerank {item.rerank.toFixed(1)}</span>}
      </div>
    </div>
  )
}

function PagePreview({ reportId, page, pageCount, onChange, onClose }: { reportId: number; page: number; pageCount: number; onChange: (page: number) => void; onClose: () => void }) {
  const [loaded, setLoaded] = useState(false)
  return (
    <div className="rl-preview">
      <div className="rl-preview-bar">
        <button className="icon-btn" disabled={page <= 1} onClick={() => onChange(page - 1)} aria-label="Previous page">‹</button>
        <span>Page {page} / {pageCount}</span>
        <button className="icon-btn" disabled={page >= pageCount} onClick={() => onChange(page + 1)} aria-label="Next page">›</button>
        <button className="icon-btn" onClick={onClose} aria-label="Close preview" style={{ marginLeft: 'auto' }}>✕</button>
      </div>
      <div className="rl-preview-img" data-loaded={loaded}>
        <img src={reportApi.pageUrl(reportId, page)} alt={`Page ${page}`} onLoad={() => setLoaded(true)} />
      </div>
    </div>
  )
}

function SectionMap({ report, usedPaths, onOpenPage }: { report: Report; usedPaths: Set<string>; onOpenPage: (page: number) => void }) {
  const [rows, setRows] = useState<SectionRow[]>([])
  const [filter, setFilter] = useState('')

  useEffect(() => {
    let live = true
    reportApi.sections(report.id).then((r) => live && setRows(r)).catch(() => live && setRows([]))
    return () => {
      live = false
    }
  }, [report.id, report.status])

  // Depth comes from the tree, not the raw heading level: levels are
  // sparse (1, 2, 6) because outline, font and layout headings are numbered
  // on different scales.
  const depth = useMemo(() => {
    const byOrdinal = new Map(rows.map((r) => [r.ordinal, r]))
    const memo = new Map<number, number>()
    const of = (row: SectionRow): number => {
      if (memo.has(row.ordinal)) return memo.get(row.ordinal)!
      const parent = row.parent !== null ? byOrdinal.get(row.parent) : undefined
      const value = !parent || parent.ordinal === 0 ? 0 : of(parent) + 1
      memo.set(row.ordinal, value)
      return value
    }
    return new Map(rows.map((r) => [r.ordinal, of(r)]))
  }, [rows])

  const visible = rows.filter((r) => r.ordinal !== 0 && (!filter || r.title.toLowerCase().includes(filter.toLowerCase())))
  const used = (row: SectionRow) => [...usedPaths].some((p) => p === row.path || p.startsWith(`${row.path} > `))

  return (
    <div className="rl-map">
      <input className="rl-map-filter" placeholder={`Filter ${rows.length - 1} sections…`} value={filter} onChange={(e) => setFilter(e.target.value)} />
      {visible.map((row) => (
        <button key={row.ordinal} className="rl-map-row" data-used={used(row)} onClick={() => onOpenPage(row.page_from)}
          style={{ paddingLeft: 12 + Math.min(depth.get(row.ordinal) ?? 0, 4) * 14 }} title={row.path}>
          <span className="rl-map-title">{row.title}</span>
          <span className="rl-map-pages">{pages(row.page_from, row.page_to)}</span>
          {row.chunk_count > 0 && <span className="rl-map-count">{row.chunk_count}</span>}
        </button>
      ))}
    </div>
  )
}
