## ProposalPilot

A RAG assistant over a consulting firm's library of past proposals, case studies and SOWs.

Ask what you quoted a client two years ago, which engagements used Kafka, or what fixed-bid work you've done under $200k — and get an answer with a citation on every claim. Or paste a question straight out of an RFP and get a drafted proposal response, assembled from work actually delivered, with the gaps it couldn't support listed separately.

Built for the RAG training assignment. Implements the full pipeline — **upload → parse → chunk → embed → store → retrieve → generate** — and then the parts that decide whether a RAG system is usable in practice: knowing when *not* to answer, making retrieval inspectable, and measuring quality with numbers instead of impressions.

### Why a proposal library

A generic "chat with your PDFs" demo hides the hard problems, because every question is a similarity lookup. A real question here breaks that immediately:

> *"Have we done anything in healthcare with Django, under $200k, since 2022?"*

Almost none of that is semantic. `< $200,000` has no embedding. "Since 2022" has no embedding. "Healthcare" is a fact about the document, not its prose. A pipeline that embeds the whole question answers it confidently and wrongly, because every proposal in the corpus discusses scope and delivery in near-identical language.

### What makes it different

| | |
|---|---|
| **Structure-aware chunking** | Section boundaries are chunk boundaries. `$180,000` under *Pricing* stays distinguishable from `$180,000` under *Savings we delivered*. |
| **Contextual retrieval** | Each chunk is prefixed with a generated header (client, industry, tech, section) before embedding, then stripped before display. Anthropic's technique without the per-chunk LLM call. |
| **Metadata-filtered ANN** | Filters are pushed *into* the vector query via `sqlite-vec` metadata columns, not applied after. Filtering afterwards fails silently when matching documents are a small slice of the corpus. |
| **It refuses** | A confidence gate declines rather than generating from weak evidence, and says what's missing. |
| **Citations verified** | Citations pointing at evidence that was never retrieved are stripped post-generation; uncited factual claims are counted and surfaced. |
| **Draft mode** | Turns an RFP line item into proposal-voice prose grounded in real engagements, plus an explicit list of what the library could *not* support. |

### Measured quality

Twelve golden questions with a known correct source document, including a negative case the system must refuse.

| Metric | With Gemini | Offline (no API key) |
|---|---|---|
| Recall@6 | **1.00** | **1.00** |
| MRR | **0.96** | 0.80 |
| Keyword coverage | **1.00** | 0.91 |
| Invalid citations | **0** | **0** |

Recall@6 is the metric that matters most: it separates a *retrieval* bug from a *generation* bug. If the right document was never retrieved, no prompt change can fix the answer.

### Stack

Python 3.10 · FastAPI · React + TypeScript + Vite · Gemini (`gemini-flash-lite-latest`, `gemini-embedding-001`) · `sqlite-vec` for indexed cosine ANN · BM25 + Reciprocal Rank Fusion · 53 tests

Runs **without an API key** in offline mode — retrieval, filtering, the vector index and the whole evaluation harness work unchanged; answers become extractive quotations instead of generated prose.

### Run it

```bash
./run.sh          # → http://127.0.0.1:8000
```

See `README.md` for setup and `ARCHITECTURE.md` for the design decisions and the trade-offs behind each one.
