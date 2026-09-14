import type { Citation } from '../types'

interface Props {
  citations: Citation[]
  highlighted: number | null
}

export function SourcesPanel({ citations, highlighted }: Props) {
  return (
    <aside className="sources">
      <div className="section-head">
        Sources
        <span className="count">{citations.length || ''}</span>
      </div>
      <div className="scroll">
        {citations.length === 0 ? (
          <div className="empty">
            Evidence used to answer appears here, with the document section and page it came from.
          </div>
        ) : (
          citations.map((citation) => (
            <div
              key={citation.index}
              id={`source-${citation.index}`}
              className="source"
              data-highlight={highlighted === citation.index}
            >
              <div className="source-head">
                <div className="source-index">{citation.index}</div>
                <div className="source-title">{citation.title}</div>
              </div>
              <div className="source-path">
                {citation.section_path || 'body'}
                {citation.page_from > 0 && ` · p.${citation.page_from}`}
              </div>
              <div className="source-snippet">{citation.snippet}</div>
            </div>
          ))
        )}
      </div>
    </aside>
  )
}
