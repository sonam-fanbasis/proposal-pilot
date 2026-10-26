# ProposalPilot

**A RAG assistant over a consulting firm's proposal and case-study library.**

Ask what you quoted a client two years ago, which engagements used Kafka, or what
fixed-bid work you've done under $200k — and get an answer with a citation on every
claim. Or paste a question straight out of an RFP and get a drafted proposal
response, assembled from work you've actually delivered, with the gaps it couldn't
support listed separately.

Built as the hands-on assignment following the RAG training session. It implements
the full pipeline — **upload → parse → chunk → embed → store → retrieve → generate** —
and then addresses the parts that decide whether a RAG system is usable in practice:
knowing when *not* to answer, making retrieval inspectable, and measuring quality
with numbers instead of impressions.

---

## Quick start

```bash
./run.sh
```

That creates the Python environment, installs dependencies, indexes the six sample
documents, builds the frontend, and serves everything at **http://127.0.0.1:8000**.

**No API key is required.** Without one the app runs in *offline mode*: parsing,
cataloguing, chunking, the vector index, hybrid retrieval, filtering and the whole
evaluation harness work exactly as they do online, and answers become ranked
quotations from your documents instead of generated prose. Nothing is stubbed and
nothing is faked.

For generated answers, get a free key at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey) and put it in
`backend/.env`:

```
GEMINI_API_KEY=your_key_here
```

Then re-index so the corpus gets real embeddings: `cd backend && .venv/bin/python seed.py --reset`.

**On model names.** Google retires Gemini models on a rolling basis and closes older
ones to new API keys, so a pinned version goes stale. The defaults are
`gemini-flash-lite-latest` and `gemini-embedding-001`, chosen for free-tier headroom —
`gemini-flash-latest` tracks the newest model and is capped at 20 requests/day, which
one re-index exhausts. If a model ever 404s, the error says exactly how to list what
your key can reach:

```bash
curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY" \
  | grep -o '"name": "models/[^"]*"'
```

Set `GEMINI_MODEL` / `GEMINI_EMBED_MODEL` in `backend/.env` accordingly.

Other ways to run it:

```bash
./run.sh --dev        # Vite dev server on :5173 with hot reload
./run.sh --api-only   # API only, no Node required
docker compose up     # single container, both tiers
```

Tests and evaluation:

```bash
cd backend
.venv/bin/python -m pytest tests/ -q          # 62 tests
.venv/bin/python seed.py --reset --eval       # re-index and print retrieval metrics
```

---

## Report Lab — advanced RAG for three document shapes

The **Report Lab** tab is the advanced-RAG exercise: one pipeline that adapts to
three intentionally different documents instead of chunking everything the same way.

| Use case | Document | Shape detected | What had to change |
|---|---|---|---|
| 1 · Plain text | GPO 2018 Annual Report (82 pp.) | Narrative | Run-in headings ("**History** From the Mayflower…") and font-family headings recovered; running headers stripped; "By the Numbers" label/value panels rebuilt as tables |
| 2 · Long & structured | Global Economic Prospects, June 2026 (200 pp.) | Long & structured | PDF outline re-levelled into chapter → section → subsection; bookmarks moved to the page their title is printed on (otherwise the Middle East forecast table is filed under *South Asia*); chunks never cross a section; **query routing** to chapters, boxes, regions and the executive summary; broken "Th" ligature repaired ("Te global economy") |
| 3 · Visual | iMac Product Environmental Report (12 pp.) | Visual | Charts and icon call-outs transcribed by Gemini vision; borderless tables and label/value panels rebuilt from page geometry; superscript footnote markers linked to their endnotes and embedded with the claim they qualify |

**Pipeline:** layout parsing (PyMuPDF spans: size, family, position) → typed elements
(heading, text, table, figure, footnote, note) → section-aware chunks with full section
path → local embeddings (`snowflake-arctic-embed-xs`, ONNX on CPU — a 200-page report
embeds in ~75 s with no API quota) in `sqlite-vec` with `report_id`/`section` metadata →
**section routing** → dense + BM25 → RRF (routed lists weighted ×2) → **cross-encoder
rerank** (`ms-marco-MiniLM-L-6-v2`) → neighbour expansion + linked footnotes under a
character budget → grounded generation with citation verification.

**Measured:** all nine sample questions from the use-case document pass end to end —
retrieval hit rate 1.0, MRR 1.0, scope precision 1.0 (every evidence chunk for
"South Asia", "Chapter 3" and "the executive summary" came from that section),
0 uncited claims. Full answers: [`docs/report-lab-benchmark.md`](docs/report-lab-benchmark.md).

```bash
cd backend
# ingest the three reports and run the benchmark (writes a markdown report)
.venv/bin/python -m app.report_eval --ingest ../test_upload/*.pdf --out report.md
.venv/bin/python -m app.report_eval --no-answers   # retrieval metrics only, no LLM calls
```

In the UI each answer shows which section it was routed to, which evidence types it
used (TEXT / TABLE / CHART / NOTE / FOOTNOTE), a per-stage timing trace, and every
citation opens the rendered source page. *How it was indexed* shows the detected shape,
the recipe applied and what parsing found; *Benchmark* runs the sample questions.

New modules: `report_parsing.py`, `report_chunking.py`, `report_models.py`,
`report_store.py`, `report_retrieval.py`, `report_generation.py`, `report_eval.py`,
`reports_api.py`; frontend in `frontend/src/reports/`.

---

## Why a proposal library

A generic "chat with your PDFs" demo hides the problems that make RAG hard, because
every question is a similarity lookup and every document is interchangeable. A
proposal library breaks that assumption immediately, and the questions people
actually ask it are the interesting ones:

> *"Have we done anything in healthcare with Django, under $200k, since 2022?"*

Almost none of that is a similarity question. `< $200,000` has no embedding.
"Since 2022" has no embedding. "Healthcare" is a fact about the document, not
its prose. Only "what did we build" is semantic. A pipeline that embeds the whole
question and hopes for the best will answer it confidently and wrongly, because
every proposal in the corpus discusses scope and delivery in near-identical
language.

The second thing a proposal library forces is **refusal**. If the firm has never
done automotive work, the only correct answer is "we haven't" — but the retriever
will always return its six nearest chunks, and a fluent model handed six banking
proposals will happily write a paragraph about automotive experience. Getting that
right is worth more than any amount of answer polish.

---

## What it does

### The pipeline

```
PDF / DOCX / MD / TXT
   │
   ├─ parse ............. structure-aware: headings, sections, pages, tables
   ├─ catalogue ......... client, industry, tech stack, value, duration, model
   ├─ chunk ............. section-bounded, sentence-aligned, contextual headers
   ├─ embed ............. Gemini text-embedding-004 (asymmetric) │ offline hashing
   └─ index ............. sqlite-vec, cosine, metadata-filtered ANN

question
   ├─ resolve follow-up . "and the pricing?" → a standalone question
   ├─ plan .............. split into structured FILTERS + a semantic query
   ├─ filter ............ one indexed SQL pass → candidate document set
   ├─ search ............ filtered vector KNN ∪ BM25 over the same set
   ├─ fuse .............. Reciprocal Rank Fusion (scale-free, no tuning)
   ├─ rerank ............ section prior + metadata agreement + retriever agreement
   ├─ gate .............. confidence too low → refuse, and say what's missing
   └─ generate .......... cited answer │ RFP draft, then citations are verified
```

### Six things worth looking at

**1. Structure-aware chunking.** Section boundaries are chunk boundaries, so a chunk
is always "a piece of one section" and knows which one. `$180,000` under *Pricing*
means something different from `$180,000` under *Savings we delivered*, and fixed-size
chunking erases that distinction. Orphan headings merge *forward* into their content
and tiny subsections merge *backward* into their parent, so nothing gets filed under
the wrong heading. → [`backend/app/chunking.py`](backend/app/chunking.py)

**2. Contextual retrieval.** A chunk reading *"Phase 2 ran for 9 weeks with a team of
six"* names no client, project or technology, so it embeds nowhere near the query that
should find it. Every chunk is prefixed with a generated header before embedding:

```
[Zenith Bank Proposal | Client: Zenith Bank | Industry: banking |
 Tech: react native, node.js | Section: Delivery Plan > Phase 2]
Phase 2 ran for 9 weeks with a team of six...
```

The header is embedded but stripped before the text is shown or quoted as evidence.
It's a deterministic version of Anthropic's contextual retrieval that costs no extra
LLM calls, so indexing a large library stays fast and free. You can see the real
stored headers via the ⓘ icon on any document.

**3. Metadata-filtered ANN.** Documents are catalogued at ingest into indexed columns,
and filters are pushed *into* the vector search:

```sql
SELECT chunk_id, distance FROM vec_chunks
WHERE embedding MATCH ? AND k = ? AND doc_id IN (...)
```

The common alternative — run an unfiltered KNN, discard non-matching hits afterwards —
fails silently whenever the matching documents are a small slice of the corpus: `k`
fills with the wrong industry and the right answer never enters the candidate set.
Technology filters are conjunctive ("Node **and** AWS"), and a filter that matches
nothing is *relaxed* rather than returned empty, with the response labelled as relaxed.

**4. It refuses.** Retrieval confidence blends semantic similarity with IDF-weighted
term coverage, scaled by how many distinct documents corroborate the evidence. Below a
threshold the system declines, shows the closest material it found, and says what
would help. Asked about spacecraft avionics against this corpus it scores 0.05 and
declines; asked what Atlas cost it scores 0.49 and answers.

**5. Citations are verified, not requested.** The prompt demands a `[n]` on every
factual sentence. Afterwards the output is *checked*: citations pointing at evidence
that was never retrieved are stripped, uncited factual claims are counted, and both
numbers are shown in the UI. A model citing `[7]` when six snippets were supplied is
hallucinating, and you're told so rather than handed a confident fabrication.

**6. Draft mode.** The reason the product exists. Paste an RFP line item — *"Describe
your approach to quality assurance"* — and get a proposal-voice response grounded in
real engagements, plus an explicit **gaps** list of everything the question asked for
that the library could not support. That list is what a proposal writer actually needs:
it's the work still to do.

---

## Measured quality

`POST /api/eval/run`, or the **Evaluation** tab. Twelve golden questions with a known
correct source document, including a negative case the system must refuse.

| Metric | With Gemini | Offline (no key) | What it tells you |
|---|---|---|---|
| Recall@6 | **1.00** | **1.00** | The right document was retrieved every time. Without this, no prompt change can fix an answer — this is what separates a retrieval bug from a generation bug. |
| MRR | **0.96** | **0.80** | And it usually ranked first. |
| Keyword coverage | **1.00** | **0.91** | The expected fact actually appeared in the answer. |
| Invalid citations | **0** | **0** | Nothing cited evidence that didn't exist. |
| Ungrounded claims | **2** | **0** | Factual sentences carrying no citation, across 12 answers. |
| Median retrieval | 1651 ms | **2 ms** | Online, this is one embedding API round trip; the index itself is sub-millisecond either way. |

The offline column uses the hashing embedder and no LLM — the weakest configuration
the system can run in, and it still retrieves the right document every time. Real
embeddings mainly improve *ranking* (MRR 0.80 → 0.96), which is what puts the answer
in position 1 instead of position 2.

---

## Known limitations

Stated plainly, because a system that hides these is harder to trust.

- **Offline refusal is imperfect.** The confidence gate correctly refuses most
  out-of-corpus questions, but ones sharing generic vocabulary with the library
  ("veterinary clinic software") can still clear the threshold, because the hashing
  embedder cannot tell topical similarity from word overlap. With real embeddings
  and the model's own grounding instruction, both layers catch it. Offline mode
  returns *visibly irrelevant quotations* rather than fabricated prose, so the
  failure is legible rather than misleading.
- **Scanned PDFs need OCR first.** Text extraction only; an image-only PDF is
  rejected with that message rather than indexed as empty.
- **No authentication.** Single-tenant by design — this is a team's own library,
  not a multi-customer product. Adding auth means a users table and a dependency
  on each route, not a rearchitecture.
- **Heuristic cataloguing is weaker on `client`.** Without an LLM key, industry,
  technology, value, duration and engagement model extract reliably (all six sample
  documents catalogue correctly), but identifying *which* party is the client from
  prose is the one field that genuinely needs a model.
- **`sqlite-vec` scales to a real library, not to a public corpus.** It is fast and
  indexed into the millions of chunks; past that, `vectorstore.py` is the single
  file to swap.

---

## Project layout

```
backend/
  app/
    parsing.py       PDF/DOCX/MD/TXT → blocks with pages and heading levels
    chunking.py      section hierarchy, sentence-aligned windows, context headers
    metadata.py      catalogue extraction (LLM structured output │ vocabulary+regex)
    embeddings.py    Gemini asymmetric embeddings │ offline hashing embedder
    vectorstore.py   sqlite-vec cosine ANN with metadata filters │ NumPy fallback
    retrieval.py     query planning, filter pushdown, hybrid search, RRF, rerank
    generation.py    grounded answering, RFP drafting, citation verification
    ingestion.py     the write path, run on a background thread
    eval.py          recall@k, MRR, keyword coverage, grounding
    main.py          FastAPI routes, SSE streaming, static SPA
  tests/             62 tests
  seed.py            index the sample library, optionally run the eval
frontend/            React + TypeScript + Vite
sample_docs/         six realistic proposals, case studies, an SOW, a capability deck
```

[`ARCHITECTURE.md`](ARCHITECTURE.md) covers the design decisions and the trade-offs
behind each one.

## API

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/documents` | Upload; parse, catalogue and embed in the background |
| `GET` | `/api/documents` | Library with extracted metadata and status |
| `GET` | `/api/documents/{id}/chunks` | Exactly what was indexed, headers included |
| `POST` | `/api/ask` | Answer or draft, with citations and a retrieval trace |
| `POST` | `/api/reports` | Report Lab upload (PDF); layout-parse, chunk, embed in the background |
| `GET` | `/api/reports/{id}/sections` | The section tree with page ranges |
| `GET` | `/api/reports/{id}/chunks?type=table` | Typed chunks (text, table, figure, footnote, note) |
| `GET` | `/api/reports/{id}/pages/{n}.png` | Rendered page, for evidence previews |
| `POST` | `/api/reports/ask/stream` | Routed, reranked, cited answer over SSE with a full trace |
| `POST` | `/api/reports/eval` | Run the use-case sample questions |
| `POST` | `/api/ask/stream` | Same, as SSE: `meta` → `token`… → `done` |
| `GET` | `/api/facets` | Industry / tech / type coverage of the corpus |
| `GET` | `/api/stats` | Corpus size, active backends, real retrieval latency |
| `POST` | `/api/eval/run` | Run the golden set, return metrics |

Interactive docs at `/docs`.
