import { useCallback, useEffect, useRef, useState } from 'react'
import { api, askStream } from './api'
import { AnswerText } from './components/AnswerText'
import { DocumentLibrary } from './components/DocumentLibrary'
import { DocumentModal } from './components/DocumentModal'
import { EvalPage } from './components/EvalPage'
import { SourcesPanel } from './components/SourcesPanel'
import { StatsPage } from './components/StatsPage'
import { ReportLab } from './reports/ReportLab'
import type { ChatMessage, DocumentRecord, Mode } from './types'

type Tab = 'chat' | 'reports' | 'stats' | 'eval'

const SUGGESTIONS: { label: string; mode: Mode }[] = [
  { label: 'What fixed-bid work have we done under $200k?', mode: 'answer' },
  { label: 'Have we delivered anything in healthcare that met HIPAA?', mode: 'answer' },
  { label: 'Describe your approach to quality assurance and test automation.', mode: 'draft' },
  { label: 'What security certifications does your organisation hold?', mode: 'draft' },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('chat')
  const [documents, setDocuments] = useState<DocumentRecord[]>([])
  const [selected, setSelected] = useState<number[]>([])
  const [inspecting, setInspecting] = useState<DocumentRecord | null>(null)

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [mode, setMode] = useState<Mode>('answer')
  const [busy, setBusy] = useState(false)
  const [conversationId, setConversationId] = useState<number | null>(null)
  const [highlighted, setHighlighted] = useState<number | null>(null)
  const [toast, setToast] = useState<{ text: string; tone?: string } | null>(null)
  const [health, setHealth] = useState<{ llm: string; embeddings: string } | null>(null)

  const endRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  const notify = useCallback((text: string, tone?: string) => {
    setToast({ text, tone })
    window.setTimeout(() => setToast(null), 4500)
  }, [])

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await api.documents())
    } catch {
      /* the poll will retry */
    }
  }, [])

  useEffect(() => {
    refreshDocuments()
    fetch('/api/health').then((r) => r.json()).then(setHealth).catch(() => {})
  }, [refreshDocuments])

  // Ingestion runs in a background thread, so poll while anything is in
  // flight and stop as soon as the library settles.
  useEffect(() => {
    const pending = documents.some((d) => ['pending', 'parsing', 'embedding'].includes(d.status))
    if (!pending) return
    const timer = window.setInterval(refreshDocuments, 1200)
    return () => window.clearInterval(timer)
  }, [documents, refreshDocuments])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleUpload = async (files: FileList) => {
    for (const file of Array.from(files)) {
      try {
        await api.upload(file)
      } catch (error) {
        notify(error instanceof Error ? error.message : `Could not upload ${file.name}`, 'error')
      }
    }
    refreshDocuments()
  }

  const handleDelete = async (id: number) => {
    try {
      await api.deleteDocument(id)
      setSelected((current) => current.filter((x) => x !== id))
      refreshDocuments()
    } catch (error) {
      notify(error instanceof Error ? error.message : 'Delete failed', 'error')
    }
  }

  const jumpToSource = (index: number) => {
    setHighlighted(index)
    document.getElementById(`source-${index}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    window.setTimeout(() => setHighlighted(null), 1800)
  }

  const send = async (question: string, sendMode: Mode = mode) => {
    const trimmed = question.trim()
    if (!trimmed || busy) return

    const ready = documents.filter((d) => d.status === 'ready').length
    if (ready === 0) {
      notify('Upload at least one document before asking a question.', 'error')
      return
    }

    setInput('')
    setBusy(true)
    const assistantId = `a-${Date.now()}`
    setMessages((current) => [
      ...current,
      { id: `u-${Date.now()}`, role: 'user', content: trimmed, mode: sendMode, citations: [], gaps: [], confidence: 0, ungrounded: 0, invalidCitations: [] },
      { id: assistantId, role: 'assistant', content: '', mode: sendMode, citations: [], gaps: [], confidence: 0, ungrounded: 0, invalidCitations: [], streaming: true },
    ])

    const update = (patch: Partial<ChatMessage>) =>
      setMessages((current) => current.map((m) => (m.id === assistantId ? { ...m, ...patch } : m)))

    const controller = new AbortController()
    abortRef.current = controller

    try {
      await askStream(
        { question: trimmed, mode: sendMode, conversation_id: conversationId, document_ids: selected },
        {
          onMeta: (meta) => {
            setConversationId(meta.conversation_id)
            update({ citations: meta.citations, confidence: meta.confidence, retrieval: meta.retrieval })
          },
          onToken: (text) => {
            setMessages((current) =>
              current.map((m) => (m.id === assistantId ? { ...m, content: m.content + text } : m)),
            )
          },
          onDone: (done) => {
            // `final_text` is the verified answer, with any invented citation
            // stripped from what streamed in raw. The offline and
            // low-confidence paths do not send it, and spreading an undefined
            // `content` would blank the message that already streamed in, so
            // the key is only added when the server actually sent one.
            const patch: Partial<ChatMessage> = {
              citations: done.citations,
              gaps: done.gaps,
              ungrounded: done.ungrounded_sentences,
              invalidCitations: done.invalid_citations,
              totalMs: done.total_ms,
              streaming: false,
            }
            if (done.final_text !== undefined) patch.content = done.final_text
            update(patch)
          },
          onError: (message) => {
            update({ streaming: false })
            notify(message, 'error')
          },
        },
        controller.signal,
      )
    } catch (error) {
      if (!controller.signal.aborted) {
        notify(error instanceof Error ? error.message : 'Request failed', 'error')
      }
      update({ streaming: false })
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant')
  const offline = health?.llm === 'offline-extractive'

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">P</div>
          <div>
            ProposalPilot <small>RFP response assistant</small>
          </div>
        </div>

        <nav className="tabs">
          {(['chat', 'reports', 'stats', 'eval'] as Tab[]).map((name) => (
            <button key={name} className="tab" data-active={tab === name} onClick={() => setTab(name)}>
              {{ chat: 'Proposal library', reports: 'Report Lab', stats: 'Corpus', eval: 'Evaluation' }[name]}
            </button>
          ))}
        </nav>

        <div className="topbar-right">
          {health && (
            <span className="pill" data-tone={offline ? 'warn' : 'good'} title={offline ? 'No GEMINI_API_KEY set — answers are extractive quotes' : 'Gemini connected'}>
              <span className="dot" />
              {offline ? 'Offline mode' : health.llm}
            </span>
          )}
        </div>
      </header>

      {tab === 'reports' ? (
        <ReportLab notify={notify} />
      ) : (
      <div className="layout">
        {tab === 'chat' && (
          <DocumentLibrary
            documents={documents}
            selected={selected}
            onToggleSelect={(id) =>
              setSelected((current) => (current.includes(id) ? current.filter((x) => x !== id) : [...current, id]))
            }
            onUpload={handleUpload}
            onDelete={handleDelete}
            onInspect={setInspecting}
          />
        )}

        {tab === 'chat' ? (
          <main className="panel">
            <div className="scroll messages">
              {messages.length === 0 && (
                <div className="empty" style={{ paddingTop: 60 }}>
                  <div style={{ fontSize: 15, color: 'var(--text)', marginBottom: 8 }}>
                    Ask your proposal library a question
                  </div>
                  Look up what you quoted, what you delivered and what it cost — or switch to
                  <strong style={{ color: 'var(--accent)' }}> Draft </strong>
                  and turn an RFP question into a cited response you can paste into a proposal.
                </div>
              )}

              {messages.map((message) =>
                message.role === 'user' ? (
                  <div key={message.id} className="msg msg-user">
                    <div className="msg-role">
                      You
                      {message.mode === 'draft' && <span className="tag">draft mode</span>}
                    </div>
                    <div className="bubble">{message.content}</div>
                  </div>
                ) : (
                  <div key={message.id} className="msg msg-assistant">
                    <div className="msg-role">
                      ProposalPilot
                      {message.streaming && <span className="spinner" />}
                    </div>

                    {message.content ? (
                      <AnswerText text={message.content} citations={message.citations} onCitationClick={jumpToSource} />
                    ) : (
                      <div className="bubble" style={{ color: 'var(--text-faint)' }}>Searching the library…</div>
                    )}

                    {message.gaps.length > 0 && (
                      <div className="gaps">
                        <div className="gaps-title">Not supported by the library</div>
                        <ul>
                          {message.gaps.map((gap, i) => (
                            <li key={i}>{gap}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {message.invalidCitations.length > 0 && (
                      <div className="warn-box">
                        The model cited {message.invalidCitations.length} source
                        {message.invalidCitations.length > 1 ? 's' : ''} that were not retrieved. Those
                        citations were removed automatically — treat the surrounding claims with caution.
                      </div>
                    )}

                    {!message.streaming && message.content && (
                      <>
                        <div className="answer-meta">
                          <span className="pill" data-tone={message.confidence >= 0.35 ? 'good' : 'warn'}>
                            <span className="dot" />
                            confidence {message.confidence.toFixed(2)}
                          </span>
                          <span className="pill">{message.citations.length} sources</span>
                          {message.ungrounded > 0 && (
                            <span className="pill" data-tone="warn">
                              <span className="dot" />
                              {message.ungrounded} uncited claim{message.ungrounded > 1 ? 's' : ''}
                            </span>
                          )}
                          {message.totalMs !== undefined && <span className="pill">{message.totalMs}ms</span>}
                        </div>

                        {message.retrieval && (
                          <details className="inspector">
                            <summary>Retrieval trace</summary>
                            <div className="inspector-body">
                              <dl className="kv">
                                <dt>rewritten as</dt>
                                <dd>{message.retrieval.standalone_question}</dd>
                                <dt>search query</dt>
                                <dd>{message.retrieval.search_query}</dd>
                                <dt>intent</dt>
                                <dd>{message.retrieval.intent}</dd>
                                <dt>planned by</dt>
                                <dd>{message.retrieval.planned_by}</dd>
                                <dt>filters</dt>
                                <dd>
                                  {Object.keys(message.retrieval.filters).length
                                    ? JSON.stringify(message.retrieval.filters)
                                    : 'none — searched the whole corpus'}
                                </dd>
                                <dt>documents in scope</dt>
                                <dd>
                                  {message.retrieval.filtered_document_count ?? 'all'}
                                  {message.retrieval.filters_relaxed && ' (filters relaxed — no exact match)'}
                                </dd>
                                <dt>candidates fused</dt>
                                <dd>{message.retrieval.candidates}</dd>
                                <dt>retrieval time</dt>
                                <dd>{message.retrieval.retrieval_ms}ms</dd>
                                <dt>embeddings</dt>
                                <dd>{message.retrieval.embedding_backend}</dd>
                              </dl>
                            </div>
                          </details>
                        )}
                      </>
                    )}
                  </div>
                ),
              )}
              <div ref={endRef} />
            </div>

            <div className="composer">
              {messages.length === 0 && (
                <div className="suggestions">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion.label}
                      className="suggestion"
                      onClick={() => {
                        setMode(suggestion.mode)
                        send(suggestion.label, suggestion.mode)
                      }}
                    >
                      {suggestion.label}
                    </button>
                  ))}
                </div>
              )}

              <div className="composer-modes">
                <button className="mode-btn" data-active={mode === 'answer'} onClick={() => setMode('answer')} title="Short, cited, factual answer">
                  Ask
                </button>
                <button className="mode-btn" data-active={mode === 'draft'} onClick={() => setMode('draft')} title="Write a proposal response to an RFP question">
                  Draft RFP response
                </button>
                {selected.length > 0 && (
                  <span className="scope-note">
                    scoped to {selected.length} document{selected.length > 1 ? 's' : ''}
                    <button onClick={() => setSelected([])}>clear</button>
                  </span>
                )}
              </div>

              <div className="composer-row">
                <textarea
                  value={input}
                  placeholder={
                    mode === 'draft'
                      ? 'Paste an RFP question — e.g. "Describe your disaster recovery capability."'
                      : 'Ask about past proposals, pricing, clients, technologies…'
                  }
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      send(input)
                    }
                  }}
                  rows={1}
                  style={{ height: 'auto' }}
                  onInput={(e) => {
                    const el = e.currentTarget
                    el.style.height = 'auto'
                    el.style.height = `${Math.min(el.scrollHeight, 190)}px`
                  }}
                />
                <button className="send" onClick={() => send(input)} disabled={busy || !input.trim()}>
                  {busy ? '…' : 'Send'}
                </button>
              </div>
            </div>
          </main>
        ) : tab === 'stats' ? (
          <main className="panel"><StatsPage /></main>
        ) : (
          <main className="panel"><EvalPage onError={(m) => notify(m, 'error')} /></main>
        )}

        {tab === 'chat' && <SourcesPanel citations={lastAssistant?.citations ?? []} highlighted={highlighted} />}
      </div>
      )}

      {inspecting && <DocumentModal doc={inspecting} onClose={() => setInspecting(null)} />}
      {toast && <div className="toast" data-tone={toast.tone}>{toast.text}</div>}
    </div>
  )
}
