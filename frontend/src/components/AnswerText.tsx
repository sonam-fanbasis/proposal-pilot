import type { Citation } from '../types'

interface Props {
  text: string
  citations: Citation[]
  onCitationClick: (index: number) => void
}

/**
 * Renders an answer with clickable citation markers.
 *
 * Deliberately a small hand-rolled renderer rather than a markdown library:
 * the only formatting the model is asked to produce is paragraphs, bullets,
 * bold and inline code, and `[n]` markers have to become interactive elements
 * wired to the sources panel -- which a generic renderer would emit as plain
 * text. Keeping it small also means no untrusted HTML is ever injected.
 */
export function AnswerText({ text, citations, onCitationClick }: Props) {
  const valid = new Set(citations.map((c) => c.index))
  const blocks = text.split(/\n{2,}/).filter((b) => b.trim())

  return (
    <div className="bubble">
      {blocks.flatMap((block, blockIndex) =>
        renderBlock(block, blockIndex, valid, onCitationClick),
      )}
    </div>
  )
}


const BULLET = /^\s*[-*]\s+/

/**
 * Render one block, which may mix prose and bullets.
 *
 * Models routinely emit a lead-in line and its bullets with no blank line
 * between them:
 *
 *     Supporting details from the evidence:
 *     - Testing is integrated into development [1].
 *     - Broader regression suites run nightly [1].
 *
 * Treating a block as a list only when *every* line is a bullet sends that
 * through the paragraph path, where HTML collapses the newlines and the
 * dashes run together into one unreadable line. So consecutive bullet lines
 * are grouped into their own list and everything else stays prose.
 */
function renderBlock(
  block: string,
  blockIndex: number,
  valid: Set<number>,
  onClick: (index: number) => void,
) {
  const out: React.ReactNode[] = []
  let bullets: string[] = []
  let prose: string[] = []

  const flushBullets = () => {
    if (!bullets.length) return
    const items = bullets
    bullets = []
    out.push(
      <ul key={`${blockIndex}-ul-${out.length}`}>
        {items.map((item, i) => (
          <li key={i}>{renderInline(item, valid, onClick)}</li>
        ))}
      </ul>,
    )
  }
  const flushProse = () => {
    if (!prose.length) return
    const text = prose.join(' ')
    prose = []
    out.push(<p key={`${blockIndex}-p-${out.length}`}>{renderInline(text, valid, onClick)}</p>)
  }

  for (const line of block.split('\n')) {
    if (BULLET.test(line)) {
      flushProse()
      bullets.push(line.replace(BULLET, ''))
    } else if (line.trim()) {
      flushBullets()
      prose.push(line.trim())
    }
  }
  flushProse()
  flushBullets()
  return out
}

function renderInline(text: string, valid: Set<number>, onClick: (index: number) => void) {
  // One pass over bold, inline code and citation markers together, so a
  // citation inside bold text still renders as a button.
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\[\d+\])/g
  const parts = text.split(pattern)

  return parts.map((part, i) => {
    if (!part) return null
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={i}>{part.slice(2, -2)}</strong>
    if (part.startsWith('`') && part.endsWith('`')) return <code key={i}>{part.slice(1, -1)}</code>

    const match = /^\[(\d+)\]$/.exec(part)
    if (match) {
      const index = Number(match[1])
      // A marker with no matching source was stripped server-side; if one
      // still reaches here it is shown as plain text rather than a dead link.
      if (!valid.has(index)) return <span key={i}>{part}</span>
      return (
        <button
          key={i}
          className="cite"
          onClick={() => onClick(index)}
          title="Jump to source"
          aria-label={`Source ${index}`}
        >
          {index}
        </button>
      )
    }
    return <span key={i}>{part}</span>
  })
}
