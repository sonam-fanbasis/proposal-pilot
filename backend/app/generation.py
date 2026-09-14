"""Answer generation with enforced grounding.

Two modes, because a proposal team asks two different things of its library:

* `answer` -- "what did we quote Zenith Bank?" A short, cited, factual answer.
* `draft`  -- "Describe your QA process." Not a question at all: a line item
  from an RFP that needs a *proposal response* written in proposal voice,
  assembled from what the firm has actually delivered before. This is the mode
  the product exists for; plain Q&A is the fallback.

Grounding is enforced rather than requested. The prompt demands a `[n]`
citation on every factual sentence, and `verify_citations()` then checks the
output, drops references to evidence that was never retrieved, and reports an
`ungrounded` count that the UI surfaces. A model that cites [7] when six
snippets were supplied is hallucinating, and the user gets told so instead of
being quietly handed a confident-looking fabrication.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from . import llm
from .retrieval import Hit, RetrievalResult

_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")

ANSWER_SYSTEM = """You answer questions about a consulting firm's library of past proposals, \
case studies and statements of work. You are speaking to that firm's own sales and delivery team.

Absolute rules:
- Use ONLY the numbered evidence provided. You have no other knowledge of this firm.
- Put a citation like [2] at the end of every sentence that states a fact. Cite several as [1][3] when they agree.
- Never cite a number that does not appear in the evidence.
- If the evidence does not answer the question, say so plainly in one sentence and state what is missing. \
Do not pad the answer with what the evidence does happen to say.
- Numbers, prices, dates, durations and client names must be copied exactly. Never estimate, round or interpolate.
- Be concise and specific. Lead with the direct answer, then the supporting detail.
- When several past projects are relevant, compare them briefly rather than describing only the first."""

DRAFT_SYSTEM = """You draft responses to RFP questions for a consulting firm, using only \
evidence from the firm's library of past proposals, case studies and statements of work.

Write the response as it would appear in the submitted proposal:
- Confident, specific, professional prose. Second person for the client ("your team"), first person plural for the firm ("we").
- Lead with the direct answer to what was asked, then the evidence for it.
- Ground every claim in the evidence and cite it as [2]. Concrete proof points -- named clients, real numbers, real outcomes -- are the whole value here.
- Never invent a capability, metric, certification, client name or number that is not in the evidence.
- 150-300 words unless the question clearly needs more. No headings unless the response has genuinely distinct parts.
- Do not open with "We are pleased to" or similar filler. Start with substance.

After the response, output a line containing exactly `---GAPS---`, then a short bullet list of \
anything the question asked for that the evidence could NOT support. This list is for the \
internal proposal writer, not the client. Write `- none` if the evidence covered everything."""


@dataclass
class Citation:
    index: int
    chunk_id: int
    doc_id: int
    title: str
    filename: str
    section_path: str
    page_from: int
    page_to: int
    snippet: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class GeneratedAnswer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    ungrounded_sentences: int = 0
    invalid_citations: list[int] = field(default_factory=list)
    mode: str = "answer"


def build_evidence(hits: list[Hit]) -> str:
    """Render retrieved chunks as numbered evidence.

    Each block leads with its provenance. Telling the model *where* a snippet
    came from is what lets it answer "which project was that?" and lets it
    weigh a signed SOW above a marketing capability deck.
    """
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        descriptor = [hit.title]
        if hit.client:
            descriptor.append(f"client: {hit.client}")
        if hit.industry:
            descriptor.append(hit.industry)
        if hit.year:
            descriptor.append(str(hit.year))
        if hit.doc_type:
            descriptor.append(hit.doc_type.replace("_", " "))
        location = hit.section_path or "body"
        if hit.page_from:
            location += f", p.{hit.page_from}"
        blocks.append(
            f"[{index}] {' | '.join(descriptor)}\n"
            f"    Section: {location}\n"
            f"    {hit.text}"
        )
    return "\n\n".join(blocks)


def _citations_from_hits(hits: list[Hit]) -> list[Citation]:
    return [
        Citation(
            index=index,
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            title=hit.title,
            filename=hit.filename,
            section_path=hit.section_path,
            page_from=hit.page_from,
            page_to=hit.page_to,
            snippet=hit.text[:400],
        )
        for index, hit in enumerate(hits, start=1)
    ]


def verify_citations(text: str, hits: list[Hit]) -> tuple[str, list[Citation], int, list[int]]:
    """Strip impossible citations and measure how much of the answer is grounded.

    Returns the cleaned text, only the citations actually used, the count of
    factual sentences carrying no citation, and any citation numbers the model
    invented.
    """
    # Models group citations as "[1, 2]" or "[1,2]" no matter how the prompt
    # asks for them. Left alone, the `[n]` pattern matches neither, so those
    # sources vanish from the panel and every sentence carrying one is counted
    # as an uncited claim -- the grounding metric reports failures that are
    # really just formatting. Split groups into "[1][2]" first.
    text = re.sub(
        r"\[\s*(\d+(?:\s*,\s*\d+)+)\s*\]",
        lambda m: "".join(f"[{n.strip()}]" for n in m.group(1).split(",")),
        text,
    )

    # Pull a citation that trails its sentence back inside it: "we quoted
    # $450,000. [2]" becomes "we quoted $450,000 [2]." Without this the
    # sentence splitter separates the claim from its citation and every such
    # sentence is scored as ungrounded -- a constant, invisible offset on the
    # grounding metric. It also reads better.
    text = re.sub(r"([.!?])\s*((?:\[\d+\])+)(?=\s|$)", r" \2\1", text)

    valid = set(range(1, len(hits) + 1))
    invalid = sorted({int(n) for n in _CITATION.findall(text)} - valid)

    if invalid:
        def drop_invalid(match: re.Match) -> str:
            return "" if int(match.group(1)) not in valid else match.group(0)

        text = _CITATION.sub(drop_invalid, text)
        text = re.sub(r" +([.,;:])", r"\1", text)

    used = sorted({int(n) for n in _CITATION.findall(text)} & valid)

    ungrounded = 0
    for sentence in _SENTENCE.split(text):
        stripped = sentence.strip().lstrip("-*# ").strip()
        if len(stripped) < 15:
            continue
        # A number is the strongest signal of a checkable claim, and short
        # numeric sentences ("It cost $450,000.") are the most dangerous thing
        # an ungrounded generator can emit -- so they are never skipped for
        # being brief. Sentences with neither figures nor proper nouns are
        # usually connective prose and are not counted.
        has_number = bool(re.search(r"\d", stripped))
        has_proper_noun = bool(re.search(r"\b[A-Z][a-z]{2,}", stripped[1:]))
        if not has_number and (not has_proper_noun or len(stripped) < 40):
            continue
        if not _CITATION.search(stripped):
            ungrounded += 1

    all_citations = _citations_from_hits(hits)
    return text.strip(), [c for c in all_citations if c.index in used], ungrounded, invalid


def _split_gaps(raw: str) -> tuple[str, list[str]]:
    if "---GAPS---" not in raw:
        return raw.strip(), []
    body, _, tail = raw.partition("---GAPS---")
    gaps = []
    for line in tail.splitlines():
        cleaned = line.strip().lstrip("-*• ").strip()
        if cleaned and cleaned.lower() not in {"none", "none.", "n/a"}:
            gaps.append(cleaned)
    return body.strip(), gaps


def build_prompt(question: str, hits: list[Hit], mode: str, history: list[dict] | None = None) -> str:
    parts: list[str] = []
    if history:
        transcript = "\n".join(
            f"{m.get('role')}: {m.get('content', '')[:300]}" for m in history[-4:]
        )
        parts.append(f"Conversation so far (for context only):\n{transcript}\n")
    parts.append(f"EVIDENCE\n{build_evidence(hits)}\n")
    if mode == "draft":
        parts.append(f"RFP QUESTION TO RESPOND TO:\n{question}\n\nWrite the proposal response.")
    else:
        parts.append(f"QUESTION: {question}\n\nAnswer using only the evidence above.")
    return "\n".join(parts)


# --- offline extractive path -------------------------------------------------

def _extractive(question: str, hits: list[Hit], mode: str, reason: str = "offline") -> GeneratedAnswer:
    """Quote the corpus instead of writing prose.

    With no model available the honest move is extraction, not generation:
    pick the sentences that best match the question and present them as
    quotations with their citations. The user gets real, verifiable content
    and is never shown invented text.
    """
    terms = {w for w in _WORD.findall(question.lower()) if len(w) > 2}
    scored: list[tuple[float, int, str]] = []
    for index, hit in enumerate(hits, start=1):
        for sentence in _SENTENCE.split(hit.text):
            sentence = sentence.strip()
            if len(sentence) < 40:
                continue
            words = {w for w in _WORD.findall(sentence.lower()) if len(w) > 2}
            if not words:
                continue
            overlap = len(terms & words) / (len(terms) or 1)
            # Favour sentences with concrete figures; they answer most of the
            # questions this product is asked.
            density = 0.15 if re.search(r"\d", sentence) else 0.0
            rank_weight = 1.0 / index
            score = overlap + density + 0.25 * rank_weight
            if overlap > 0 or density > 0:
                scored.append((score, index, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return GeneratedAnswer(
            text=(
                "I could not find anything in the indexed documents that answers this. "
                "Try rephrasing, or upload the proposal or case study that covers it."
            ),
            mode=mode,
        )

    seen: set[str] = set()
    lines: list[str] = []
    for _score, index, sentence in scored[:6]:
        key = sentence[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {sentence.rstrip(' .')} [{index}].")

    # The banner must state the real cause. Reporting a rate-limited or
    # erroring API as "no language model is configured" sends the reader to
    # check a key that is already correct.
    if reason == "offline":
        header = (
            "**Offline mode** — no language model is configured, so these are the most "
            "relevant passages quoted directly from your documents rather than a written "
            "answer. Set `GEMINI_API_KEY` for generated responses.\n\n"
        )
    elif reason == "rate_limit":
        header = (
            "**Rate limited** — the Gemini free tier quota is exhausted, so these are the "
            "most relevant passages quoted directly from your documents. Retrieval and "
            "citations are unaffected. Wait for the quota to reset, or set `GEMINI_MODEL` "
            "in `backend/.env` to a model with more free headroom.\n\n"
        )
    else:
        header = (
            f"**Language model unavailable** ({reason}) — these are the most relevant "
            "passages quoted directly from your documents. Retrieval and citations are "
            "unaffected.\n\n"
        )
    # Grounding is measured on the quoted passages only. The disclaimer is our
    # own prose, and counting it as an uncited factual claim would put a
    # constant offset on every eval run -- exactly the kind of noise that hides
    # a real regression.
    body = "\n".join(lines)
    cleaned, citations, ungrounded, invalid = verify_citations(body, hits)
    return GeneratedAnswer(
        text=header + cleaned, citations=citations, gaps=[], ungrounded_sentences=ungrounded,
        invalid_citations=invalid, mode=mode,
    )


# --- public API --------------------------------------------------------------

def generate(
    question: str,
    result: RetrievalResult,
    mode: str = "answer",
    history: list[dict] | None = None,
) -> GeneratedAnswer:
    from .config import get_settings

    hits = result.hits
    if not hits:
        return GeneratedAnswer(
            text=(
                "Nothing in the indexed library matches that. "
                + (
                    "The filters in your question matched no documents, so I widened the search and still found nothing. "
                    if result.relaxed
                    else ""
                )
                + "Upload the relevant proposal or case study and ask again."
            ),
            mode=mode,
        )

    if result.confidence < get_settings().min_confidence:
        # Refusing on weak evidence is a feature. The alternative is a fluent
        # answer built from the nearest six chunks, which is precisely how RAG
        # systems mislead people.
        preview = "\n".join(
            f"- {h.title} — {h.section_path or 'body'}" for h in hits[:3]
        )
        return GeneratedAnswer(
            text=(
                "I do not have strong enough evidence to answer that reliably "
                f"(retrieval confidence {result.confidence:.2f}). The closest material is:\n\n{preview}\n\n"
                "Try naming the client, technology or project, or upload the document that covers this."
            ),
            citations=_citations_from_hits(hits[:3]),
            mode=mode,
        )

    if llm.is_offline():
        return _extractive(question, hits, mode)

    system = DRAFT_SYSTEM if mode == "draft" else ANSWER_SYSTEM
    prompt = build_prompt(question, hits, mode, history)
    try:
        raw = llm.complete(prompt, system=system, temperature=0.25, max_output_tokens=2048)
    except Exception as exc:
        message = str(exc)
        if "429" in message or "RESOURCE_EXHAUSTED" in message:
            reason = "rate_limit"
        else:
            reason = message.split("{")[0].strip()[:80] or "unknown error"
        print(f"[generation] LLM call failed ({reason}); falling back to extractive. {message[:200]}")
        return _extractive(question, hits, mode, reason)

    body, gaps = _split_gaps(raw)
    cleaned, citations, ungrounded, invalid = verify_citations(body, hits)
    return GeneratedAnswer(
        text=cleaned, citations=citations, gaps=gaps,
        ungrounded_sentences=ungrounded, invalid_citations=invalid, mode=mode,
    )


def generate_stream(
    question: str,
    result: RetrievalResult,
    mode: str = "answer",
    history: list[dict] | None = None,
) -> Iterator[str]:
    """Token stream for the SSE endpoint. Verification happens after the stream."""
    system = DRAFT_SYSTEM if mode == "draft" else ANSWER_SYSTEM
    yield from llm.stream(
        build_prompt(question, result.hits, mode, history),
        system=system,
        temperature=0.25,
        max_output_tokens=2048,
    )
