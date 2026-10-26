import { seconds, SHAPES, TYPE_ORDER, TYPES } from './meta'
import type { Report } from './types'

/** "How this report was indexed": the shape, the recipe it selected, and what parsing found. */
export function IndexProfile({ report }: { report: Report }) {
  const stats = report.profile.stats ?? {}
  const types = report.profile.chunk_types ?? {}
  const timings = report.profile.timings ?? {}
  const shape = report.shape ? SHAPES[report.shape] : null

  const found: [string, number | undefined, string][] = [
    ['Pages', stats.pages, ''],
    ['Sections', report.profile.sections, 'heading tree'],
    ['Outline headings', stats.outline_headings, 'from PDF bookmarks'],
    ['Font / layout headings', (stats.font_headings ?? 0) + (stats.minor_headings ?? 0) + (stats.run_in_headings ?? 0), 'size, family, run-in'],
    ['Tables rebuilt', stats.tables, 'from page geometry'],
    ['Charts transcribed', stats.figures, stats.vision_pages ? `${stats.vision_pages} pages sent to vision` : 'vision off'],
    ['Footnotes linked', stats.footnotes, 'marker → note'],
    ['Header/footer patterns', stats.furniture_patterns_removed, 'stripped'],
  ]

  return (
    <div className="rl-profile">
      {shape && (
        <div className="rl-profile-hero" data-shape={report.shape}>
          <div className="rl-profile-kicker">Detected shape</div>
          <div className="rl-profile-shape">{shape.label}</div>
          <p>{shape.blurb}</p>
        </div>
      )}

      <div className="rl-profile-grid">
        <section>
          <h3>Recipe applied</h3>
          <ol className="rl-recipe">
            {(report.profile.strategy ?? []).map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
          {stats.th_ligature_repair ? (
            <p className="rl-aside">Repaired a broken “Th” ligature in the source font (“Te global economy” → “The global economy”).</p>
          ) : null}
        </section>

        <section>
          <h3>What parsing found</h3>
          <dl className="rl-found">
            {found.map(([label, value, hint]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>
                  {value ?? 0}
                  {hint && <small>{hint}</small>}
                </dd>
              </div>
            ))}
          </dl>
        </section>

        <section>
          <h3>Chunks by type</h3>
          <div className="rl-type-rows">
            {TYPE_ORDER.map((t) => {
              const n = types[t] ?? 0
              const max = Math.max(1, ...TYPE_ORDER.map((x) => types[x] ?? 0))
              return (
                <div key={t} className="rl-type-row">
                  <span className="rl-type" data-type={t}>{TYPES[t].short}</span>
                  <span className="rl-type-track"><span data-type={t} style={{ width: `${(n / max) * 100}%` }} /></span>
                  <span className="rl-type-n">{n}</span>
                </div>
              )
            })}
          </div>
        </section>

        <section>
          <h3>Ingestion time</h3>
          <dl className="rl-found">
            <div><dt>Parse{stats.vision_pages ? ' + vision' : ''}</dt><dd>{seconds(timings.parse_ms ?? 0)}</dd></div>
            <div><dt>Chunk</dt><dd>{seconds(timings.chunk_ms ?? 0)}</dd></div>
            <div><dt>Embed ({report.chunk_count} chunks)</dt><dd>{seconds(timings.embed_ms ?? 0)}</dd></div>
            <div><dt>Total</dt><dd>{seconds(report.ingest_ms)}</dd></div>
          </dl>
          <p className="rl-aside">Embedder: <code>{report.profile.embedder}</code> (local, CPU)</p>
        </section>
      </div>
    </div>
  )
}
