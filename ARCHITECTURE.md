# ProposalPilot — Architecture & Design Decisions

This document explains what was built and, more usefully, *why each alternative was
rejected*. Where a decision has a known cost, the cost is stated.

---

## 1. The requirement behind the requirement

The assignment asks for the RAG pipeline: upload → parse → chunk → embed → store →
retrieve → generate. Every one of those stages has an obvious implementation that
works in a demo and fails in use. The design problem is not assembling the stages —
it is choosing, at each one, between what looks right in a notebook and what survives
contact with a real corpus and real questions.

The use case was picked to force that choice. A consulting firm's proposal library
is asked questions like:

> *"Have we done anything in healthcare with Django, under $200k, since 2022?"*

Decompose it:

| Fragment | Kind of question |
|---|---|
| "healthcare" | fact about the document |
| "with Django" | fact about the document |
| "under $200k" | numeric predicate — **has no embedding** |
| "since 2022" | numeric predicate — **has no embedding** |
| "anything we've done" | semantic |

Embed the whole string and you get a vector dominated by the semantic fragment,
searched against a corpus where every document discusses scope and delivery in
almost identical language. The system returns six confident, wrong chunks. The
architecture below is organised around that observation: **answer the structured
part structurally, and leave only the rest to similarity.**

---

## 2. High-level shape

```
React + TypeScript SPA  ──fetch / SSE──►  FastAPI (one process)
                                            │
     ┌──────────────────────────────────────┼──────────────────────────────┐
     │ WRITE PATH (background thread)        │ READ PATH                    │
     │                                       │                              │
     │ parsing.py    structure + pages       │ retrieval.py                 │
     │ metadata.py   catalogue the document  │   resolve_followup           │
     │ chunking.py   sections + ctx headers  │   plan_query  → filters      │
     │ embeddings.py 768-d, asymmetric       │   resolve_filters → doc_ids  │
     │ vectorstore.py sqlite-vec upsert      │   vector KNN ∪ BM25          │
     │                                       │   RRF fuse → rerank → gate   │
     │                                       │ generation.py → verify       │
     └───────────────────────┬───────────────┴──────────────────────────────┘
                             ▼
                    ONE SQLite file
        documents · doc_tech · chunks · vec_chunks (ANN)
        conversations · messages · query_log · eval_*
```

Everything is one Python process and one SQLite file. That is a decision, not a
shortcut — see §4.

---

## 3. Chunking: the stage that decides everything downstream

**Rejected: fixed-size windows over the flattened document.** The default in most
tutorials. It is fast and it destroys the one thing a proposal library depends on:
*which section a passage came from*. `$180,000` under **Pricing** is what we charged.
`$180,000` under **Savings we delivered** is what the client saved. Flatten the
document and those two chunks are indistinguishable — and a pricing question will
cheerfully retrieve the second.

**Built: section-bounded, sentence-aligned windows.**

1. Parsing emits `Block`s tagged with page and heading level. DOCX gives real outline
   levels from paragraph styles; PDF and text fall back to heuristics tuned on the
   conventions that dominate proposals — numbered headings (`3.2 Pricing`), all-caps
   lines, and a vocabulary of known section names.
2. The heading stack is rebuilt into a `section_path` (`Commercials > Payment Milestones`).
3. Chunks never straddle a heading. Within a section, windows are packed from whole
   sentences up to a character target, carrying an overlap tail forward so a fact
   landing on a seam is retrievable from both sides.

Two merge rules, which exist because getting them wrong mislabels content:

- An **orphan heading** (`Commercials`, with nothing under it before the next
  heading) merges *forward* into the subsection that follows, which keeps its own
  deeper path. Merging it backward would file the pricing tables under whatever
  section happened to precede them.
- A **tiny subsection** merges *backward*, but only into its own parent by path
  prefix — so two unrelated short sections are never blended just for being short.

**Cost:** a document with no detectable heading structure (a scanned-and-OCR'd wall of
text) degrades to plain sentence-packed windows. The section prior in reranking then
contributes nothing. It degrades rather than breaks.

### Contextual headers

A chunk reading *"Phase 2 ran for 9 weeks with a team of six"* is nearly
unretrievable: it names no client, project or technology, so it embeds far from the
query that should find it. Before embedding, each chunk is prefixed:

```
[Zenith Bank Proposal | Client: Zenith Bank | Industry: banking |
 Tech: react native, node.js | Section: Delivery Plan > Phase 2]
```

This is a deterministic take on Anthropic's *contextual retrieval*. The published
technique asks an LLM to write a context sentence per chunk, which is better but
costs one call per chunk — a 200-page library becomes slow and expensive to index,
and impossible to index at all without a key. Composing the header from metadata
already extracted captures most of the benefit for zero marginal cost.

The header is embedded but **stripped before the text is displayed or sent as
evidence**, so it never leaks into an answer. `GET /api/documents/{id}/chunks`
exposes the real stored headers.

---

## 4. Storage: one SQLite file

**Rejected: Chroma / Qdrant / pgvector.** All are good. All add a service to run,
and — with the exception of pgvector — put the vector index in a different system
from the metadata that has to filter it, which makes filtered search a two-round-trip
join across a network boundary.

**Built: `sqlite-vec` in the same file as the relational data.** The capability that
decided it:

```sql
SELECT chunk_id, distance FROM vec_chunks
WHERE embedding MATCH ? AND k = ? AND doc_id IN (...)
```

`doc_id` is a `vec0` metadata column, so **the filter is evaluated inside the ANN
search**. Combined with `distance_metric=cosine` over L2-normalised vectors,
`similarity = 1 - distance`.

Why that matters more than it sounds: the common pattern is to run an unfiltered
KNN and drop non-matching hits afterwards. That works when the filter matches most
of the corpus and **fails silently** when it doesn't — ask for the one healthcare
project in a corpus of forty and `k=20` fills with banking proposals, leaving zero
healthy candidates. The bug looks like "the model didn't know", which sends you
debugging the prompt. Pushing the filter down means `k` is spent entirely on
documents that already qualify.

`db.VEC_AVAILABLE` gates a NumPy brute-force fallback if the extension can't load.
Correctness is preserved; only latency changes.

**Cost:** single-writer. Fine for a team-scale library; a high-write multi-tenant
deployment wants Postgres + pgvector, which is a change to `vectorstore.py` and
`db.py` alone.

---

## 5. Cataloguing at ingest

Structured metadata — client, industry, doc type, engagement model, year, value,
duration, tech stack — is extracted once at ingest into indexed columns. Doing it at
query time would mean re-reading every document per question.

Two paths produce the same shape: Gemini structured output (schema-constrained, so
there is no "parse JSON out of a markdown fence" failure mode), or a vocabulary +
regex pass. **All six sample documents catalogue correctly with no LLM at all**, which
took three non-obvious fixes worth recording:

| Problem | Naive result | Fix |
|---|---|---|
| *"a dedicated team engagement **rather than** a fixed bid"* | `fixed_bid` — a first-match substring scan reads the negation as the answer | Score each model by surrounding context; declaration phrases add, negation phrases subtract, and the window is clipped to the sentence so a negation can't leak across a full stop |
| *"totalled **$320,000**… saved **€340,000** per year"* | `340000` — taking the largest figure | Prefer amounts preceded by an explicit total-value phrase; plain maximum only as fallback |
| *"Built with **React Native**"* | `["react native", "react"]` — and since tech filters are ANDed, a React Native query then demands a technology the document never claimed | Match longest name first and blank out the matched span |

A capability deck that describes all three commercial models is catalogued as
`other` rather than guessing, and its quoted averages ("our average engagement is 11
months") are not recorded as engagement facts — otherwise a duration filter would
match a document that has no duration.

**Cost:** `client` is the one field the heuristic path does not do well. Identifying
which party is the customer from prose needs a model.

---

## 6. Retrieval

### Query planning

Every question is split into **filters** (checkable against metadata) and a
**semantic query** (everything else). The LLM path uses a constrained schema; the
heuristic path uses vocabulary matching and regexes for money and year ranges. On the
sample corpus the heuristic path alone produces correct filters for all twelve golden
questions.

Filters resolve to a document id list in one indexed SQL pass. Two rules that matter:

- Technology filters are **conjunctive** — "Node **and** AWS" — via a normalised
  `doc_tech` table. A `LIKE '%node%'` over a JSON blob gets this wrong twice: it
  matches `nodemon`, and it cannot express AND without N scans.
- A **price filter must not exclude documents with no recorded value.** "Unknown" is
  not "too expensive", and silently dropping uncatalogued documents from a price
  query is exactly the kind of invisible recall loss that makes users stop trusting
  a search tool.

When a strict filter matches nothing, filters are **relaxed** rather than returning
empty — asked for React Native *and* AWS *and* fintech with two of three present, the
near-misses are recovered and the response is labelled `filters_relaxed` so the user
is not misled.

### Hybrid search and fusion

Vector KNN and BM25 run over the same filtered candidate set and merge by
**Reciprocal Rank Fusion**.

**Rejected: a weighted score blend** (`0.7 * cosine + 0.3 * bm25`). Cosine similarity
and BM25 live on different, corpus-dependent scales, so those weights need retuning
for every corpus and silently drift as documents are added. RRF reads only *ranks*,
so it is scale-free and has nothing to tune.

Reranking is deliberately cheap and explainable — no second model, no added latency:
a section prior (a pricing question is probably answered in a pricing section),
metadata agreement with the requested filters, and a bonus when both retrievers
independently surfaced the same chunk.

### The confidence gate

The most important component, and the one that took the most iteration.

```python
relevance   = 0.6 * best_cosine + 0.4 * idf_weighted_term_coverage
relevance  *= 1 - 0.5 * unknown_term_fraction
confidence  = relevance * (0.8 + 0.2 * corroboration)
```

Three findings are baked into that expression, each of which was a real bug:

1. **Corroboration must multiply, not add.** As an additive term it carried the score
   alone: a question about spacecraft avionics retrieved three unrelated banking
   documents, scored 0.25 on corroboration, cleared the threshold, and produced a
   confident answer from irrelevant evidence. Evidence that is not relevant does not
   become trustworthy by being plentiful.
2. **Term coverage must be IDF-weighted.** Counting terms equally let filler carry
   the score — *"have we done automotive work"* looked two-thirds covered because
   "done" and "work" appear everywhere, while "automotive" appeared in no document.
3. **Unknown terms must be measured separately from coverage.** Counting a corpus-absent
   word in the coverage denominator *and* in the unknown penalty punishes it twice.
   That made *"describe your QA process"* score 0.05 — despite correctly retrieving
   the Quality Assurance section — purely because the corpus spells that concept
   "approach". Coverage is now computed over known terms only; the unknown fraction
   is a separate multiplier, halved because an unseen word is ambiguous evidence: it
   may mean the subject is absent, or merely that the user chose a synonym.

The stopword list includes question framing ("describe", "tell", "outline") and
generic business filler ("work", "project", "experience") for the same reason —
their absence from the corpus means nothing, and counting it would make every
politely-phrased question look unanswerable.

Below the threshold the system declines, shows the closest material, and says what
would help. **Cost, stated honestly:** in offline mode this still leaks on questions
sharing generic vocabulary with the corpus (`"veterinary clinic software"` scores
0.29). The hashing embedder cannot separate topical similarity from word overlap.
With real embeddings the semantic term does the work, and the model's own grounding
instruction is a second layer. Offline, the failure surfaces as *visibly irrelevant
quotations* rather than fabricated prose — legible rather than misleading.

---

## 7. Generation and citation verification

Grounding is **enforced, not requested**. The prompt demands `[n]` on every factual
sentence; afterwards `verify_citations()`:

1. Normalises `"claim. [2]"` → `"claim [2]."` — without this the sentence splitter
   separates every trailing citation from its claim and scores the sentence as
   ungrounded, putting a constant invisible offset on the grounding metric.
2. **Strips citations pointing at evidence that was never retrieved.** A model citing
   `[7]` when six snippets were supplied is hallucinating.
3. Counts uncited factual sentences. Numeric claims are never exempted for being
   short — *"It cost $450,000."* is the most dangerous thing an ungrounded generator
   can emit.

Both counts surface in the UI. The streamed answer is replaced by the verified text
on completion, so what gets stored and displayed has had invented citations removed.

**Offline generation is extractive by design.** With no model available the honest
move is to quote the corpus, not to write prose. A RAG system that invents text when
its model is unreachable is worse than one that says it cannot answer.

---

## 8. Evaluation

"The answers look good" is not a quality claim, and it is the only claim most RAG
demos can make. Twelve golden questions with a known correct source document,
including a negative case the system must refuse.

**Recall@k is the metric that matters most**, because it separates a *retrieval* bug
from a *generation* bug. If the right document was never retrieved, no prompt change
can fix the answer — and without measuring it, you cannot tell those two failures
apart from the outside. Every eval run is stored, so runs are comparable over time.

Offline: Recall@6 1.00, MRR 0.80, keyword coverage 0.91, zero invalid citations,
zero ungrounded claims, 2 ms median retrieval.

---

## 9. Things deliberately not built

- **Multi-tenancy and auth.** This is a team's own library. Adding auth is a users
  table and a route dependency, not a rearchitecture.
- **A task queue.** Ingestion runs on a background thread with status persisted to
  the document row. Celery plus a broker would add two services to solve a problem
  this workload does not have; the status column already gives progress and failure
  reporting.
- **Agentic tool-calling.** It is genuinely useful for open-ended research, and it is
  the wrong shape here: the questions this product answers are a lookup plus a
  filter, and a planning loop would add latency and failure modes to a path where
  filter pushdown already gives an exact answer.
- **Re-ranking with a cross-encoder.** Real quality gain, but it means downloading
  `torch` — turning a 40 MB install into 2 GB and making the offline path
  impractical. The cheap explainable reranker covers the section-prior case, which is
  most of the benefit on this corpus. `_rerank()` is where a cross-encoder would slot
  in.
