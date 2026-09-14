"""Structure-aware chunking with contextual headers.

Two ideas do the heavy lifting here.

1. **Section boundaries are chunk boundaries.** We rebuild the heading
   hierarchy from the block stream and never let a chunk straddle a heading.
   A chunk is therefore always "a piece of one section", and it knows which
   section that is (`section_path`, e.g. `Commercials > Payment Milestones`).

2. **Contextual retrieval.** A raw chunk like "Phase 2 ran for 9 weeks with
   a team of six" is nearly unretrievable -- it names no client, no project,
   no technology, so it embeds far away from the query that should find it.
   Before embedding, each chunk is prefixed with a generated context header:

       [Zenith Bank mobile banking proposal | Industry: fintech |
        Tech: React Native, Node.js | Section: Delivery Plan > Phase 2]
       Phase 2 ran for 9 weeks with a team of six...

   The header is embedded with the chunk but stripped before the text is
   shown to the user or sent to the LLM as evidence. This is a cheap,
   deterministic version of Anthropic's "contextual retrieval" -- it needs no
   extra LLM call per chunk, so ingesting a 200-page proposal library stays
   free and fast.

Small sections are merged with their neighbours so a document of many short
headings does not explode into hundreds of 40-character chunks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .config import get_settings
from .parsing import Block

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])")


@dataclass
class Chunk:
    text: str
    section_path: str
    page_from: int
    page_to: int
    ordinal: int = 0
    context_header: str = ""

    @property
    def embedding_text(self) -> str:
        """What actually gets embedded: context header + body."""
        return f"{self.context_header}\n{self.text}".strip() if self.context_header else self.text


@dataclass
class _Section:
    path: str
    blocks: list[Block] = field(default_factory=list)
    has_body: bool = False

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks).strip()

    @property
    def pages(self) -> tuple[int, int]:
        if not self.blocks:
            return (1, 1)
        pages = [b.page for b in self.blocks]
        return (min(pages), max(pages))


def _split_on_sentences(text: str, target: int, overlap: int, min_chars: int) -> list[str]:
    """Pack sentences up to `target` chars, carrying `overlap` chars forward.

    Splitting mid-sentence is the single most common cause of a retrieved
    chunk that reads as nonsense, so windows are built out of whole sentences
    and only fall back to a hard character cut for a pathological run-on.
    """
    if len(text) <= target:
        return [text] if text.strip() else []

    sentences: list[str] = []
    for raw in _SENTENCE_END.split(text):
        piece = raw.strip()
        if not piece:
            continue
        while len(piece) > target:
            cut = piece.rfind(" ", 0, target)
            cut = cut if cut > target // 2 else target
            sentences.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            sentences.append(piece)

    windows: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        if size + len(sentence) + 1 > target and current:
            windows.append(" ".join(current))
            # Carry the tail of this window into the next one so a fact that
            # lands on the seam is retrievable from both sides.
            tail: list[str] = []
            tail_size = 0
            for previous in reversed(current):
                if tail_size + len(previous) > overlap:
                    break
                tail.insert(0, previous)
                tail_size += len(previous) + 1
            current = tail
            size = tail_size
        current.append(sentence)
        size += len(sentence) + 1
    if current:
        windows.append(" ".join(current))

    # A trailing sliver is folded back into the previous window.
    if len(windows) > 1 and len(windows[-1]) < min_chars:
        windows[-2] = f"{windows[-2]} {windows[-1]}"
        windows.pop()
    return windows


def _sections_from_blocks(blocks: list[Block]) -> list[_Section]:
    stack: list[str] = []
    sections: list[_Section] = []
    current = _Section(path="")

    for block in blocks:
        if block.is_heading:
            if current.blocks:
                sections.append(current)
            level = max(1, block.level)
            del stack[level - 1 :]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(block.text.strip())
            path = " > ".join(p for p in stack if p)
            current = _Section(path=path)
            # Keep the heading text inside the body too: "Pricing" is often
            # the only place the word appears, and it should be searchable.
            current.blocks.append(Block(text=block.text, page=block.page))
        else:
            current.blocks.append(block)
            current.has_body = True
    if current.blocks:
        sections.append(current)
    return [s for s in sections if s.text]


def _merge_small_sections(sections: list[_Section], min_chars: int) -> list[_Section]:
    """Fold orphan headings forward and tiny subsections backward.

    Two distinct problems, and merging them into one rule mislabels content:

    * An *orphan heading* ("Commercials" with nothing under it before the next
      heading) must merge **forward**, into the subsection that follows it, and
      the subsection keeps its own deeper path. Merging it backward would file
      the pricing tables under whatever section happened to precede them.
    * A *tiny subsection* merges **backward**, but only into its own parent
      (`path` prefix match), so we never blend two unrelated sections just
      because they were both short.
    """
    merged: list[_Section] = []
    pending: list[_Section] = []   # heading-only sections awaiting content

    for section in sections:
        if not section.has_body:
            pending.append(section)
            continue
        if pending:
            lead = [b for p in pending for b in p.blocks]
            section.blocks = lead + section.blocks
            pending = []
        parent = merged[-1] if merged else None
        if (
            parent is not None
            and parent.path
            and len(section.text) < min_chars
            and section.path.startswith(parent.path)
        ):
            parent.blocks.extend(section.blocks)
        else:
            merged.append(section)

    if pending:
        if merged:
            merged[-1].blocks.extend(b for p in pending for b in p.blocks)
        else:
            merged.extend(pending)
    return merged


def build_context_header(doc_meta: dict, section_path: str) -> str:
    """The retrieval context prepended to every chunk before embedding."""
    parts: list[str] = []
    title = doc_meta.get("title") or doc_meta.get("filename")
    if title:
        parts.append(str(title))
    if doc_meta.get("client"):
        parts.append(f"Client: {doc_meta['client']}")
    if doc_meta.get("industry"):
        parts.append(f"Industry: {doc_meta['industry']}")
    if doc_meta.get("doc_type"):
        parts.append(f"Type: {str(doc_meta['doc_type']).replace('_', ' ')}")
    if doc_meta.get("year"):
        parts.append(f"Year: {doc_meta['year']}")

    tech = doc_meta.get("tech_stack") or []
    if isinstance(tech, str):
        try:
            tech = json.loads(tech)
        except Exception:
            tech = [t.strip() for t in tech.split(",") if t.strip()]
    if tech:
        parts.append("Tech: " + ", ".join(list(tech)[:8]))
    if section_path:
        parts.append(f"Section: {section_path}")
    return "[" + " | ".join(parts) + "]" if parts else ""


def chunk_document(blocks: list[Block], doc_meta: dict | None = None) -> list[Chunk]:
    settings = get_settings()
    doc_meta = doc_meta or {}

    sections = _merge_small_sections(
        _sections_from_blocks(blocks), settings.chunk_min_chars
    )

    chunks: list[Chunk] = []
    for section in sections:
        page_from, page_to = section.pages
        windows = _split_on_sentences(
            section.text,
            settings.chunk_target_chars,
            settings.chunk_overlap_chars,
            settings.chunk_min_chars,
        )
        for window in windows:
            if not window.strip():
                continue
            chunks.append(
                Chunk(
                    text=window.strip(),
                    section_path=section.path,
                    page_from=page_from,
                    page_to=page_to,
                    ordinal=len(chunks),
                    context_header=build_context_header(doc_meta, section.path),
                )
            )
    return chunks
