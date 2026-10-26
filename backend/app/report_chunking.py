"""Hierarchical, type-aware chunking for reports.

Input: the typed `Element` stream from `report_parsing.py`.
Output: a section tree plus chunks that each belong to exactly one section.

Rules, and the problem each one solves:

* **A chunk never crosses a section boundary.** "Growth is forecast to slow to
  3.9 percent" means nothing without knowing *which region* it is about. In a
  report with six regional sections in identical prose, a chunk that
  straddles "Europe and Central Asia" and "Latin America" is a guaranteed
  wrong answer.
* **Every chunk carries its section path** (`Ch. 2 Regional Outlooks > South
  Asia > Outlook`) in the text that is embedded and keyword-indexed, so the
  retriever can match "South Asia" even when the paragraph only says "SAR".
* **Tables, figures and footnotes are their own chunk types.** A table split
  mid-row is noise; a footnote merged into body text is lost. Tables are
  split by rows with the caption repeated, so every piece is self-describing.
* **Footnotes are linked, not just stored.** A body chunk containing the
  marker `[^18]` has note 18's text appended to its *embedding* text, so a
  claim and the caveat that limits it are retrieved together.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import get_settings
from .report_parsing import Element

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'“])")
_MARKER = re.compile(r"\[\^(\d+)\]")


@dataclass
class Section:
    ordinal: int
    parent: int | None
    level: int
    title: str
    path: str
    page_from: int
    page_to: int = 0


@dataclass
class ReportChunk:
    text: str
    content_type: str          # text | table | figure | footnote | note
    section_ordinal: int
    section_path: str
    page_from: int
    page_to: int
    ordinal: int = 0
    header: str = ""
    note_ids: list[str] = field(default_factory=list)
    note_text: str = ""        # linked footnote text, embedded with the chunk

    @property
    def embedding_text(self) -> str:
        body = f"{self.header}\n{self.text}"
        if self.note_text:
            body += f"\n{self.note_text}"
        return body


def build_sections(elements: list[Element]) -> tuple[list[Section], list[int]]:
    """Rebuild the heading tree. Returns sections and, per element, its section."""
    root = Section(ordinal=0, parent=None, level=0, title="(front matter)", path="Front matter", page_from=1)
    sections = [root]
    stack: list[Section] = []
    owner: list[int] = []
    for element in elements:
        if element.kind == "heading":
            while stack and stack[-1].level >= element.level:
                stack.pop()
            parent = stack[-1] if stack else None
            title = element.text.strip()
            path = f"{parent.path} > {title}" if parent else title
            section = Section(
                ordinal=len(sections), parent=parent.ordinal if parent else 0,
                level=element.level, title=title, path=path, page_from=element.page,
            )
            sections.append(section)
            stack.append(section)
        owner.append(stack[-1].ordinal if stack else 0)

    for section in sections:
        section.page_to = section.page_from
    for element, ordinal in zip(elements, owner):
        sections[ordinal].page_to = max(sections[ordinal].page_to, element.page)
    # A parent spans its children.
    for section in reversed(sections):
        if section.parent is not None:
            parent = sections[section.parent]
            parent.page_to = max(parent.page_to, section.page_to)
    return sections, owner


def _pack(paragraphs: list[tuple[str, int]], target: int, overlap: int) -> list[tuple[str, int, int]]:
    """Pack (text, page) paragraphs into windows of ~target chars.

    Paragraph boundaries are preferred; a paragraph longer than the target is
    split on sentences. The tail of each window is carried into the next so a
    fact on the seam is retrievable from both sides.
    """
    units: list[tuple[str, int]] = []
    for text, page in paragraphs:
        if len(text) <= target:
            units.append((text, page))
            continue
        buffer = ""
        for sentence in _SENTENCE.split(text):
            if buffer and len(buffer) + len(sentence) + 1 > target:
                units.append((buffer, page))
                buffer = ""
            buffer = f"{buffer} {sentence}".strip()
        if buffer:
            units.append((buffer, page))

    windows: list[tuple[str, int, int]] = []
    current: list[tuple[str, int]] = []
    size = 0
    for unit in units:
        if current and size + len(unit[0]) > target:
            windows.append(("\n".join(u[0] for u in current), current[0][1], current[-1][1]))
            current = []
            size = 0
            if overlap:
                carried = _tail(windows[-1][0], overlap)
                current = [(carried, windows[-1][2])]
                size = len(carried)
        current.append(unit)
        size += len(unit[0]) + 1
    if current:
        windows.append(("\n".join(u[0] for u in current), current[0][1], current[-1][1]))
    # Fold a trailing sliver into its predecessor.
    if len(windows) > 1 and len(windows[-1][0]) < 200:
        text, page_from, _ = windows[-2]
        windows[-2] = (text + "\n" + windows[-1][0], page_from, windows[-1][2])
        windows.pop()
    return windows


def _tail(text: str, overlap: int) -> str:
    """The last ~overlap chars of a window, starting on a word boundary."""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    space = tail.find(" ")
    return "…" + (tail[space + 1 :] if space != -1 else tail)


def _split_table(text: str, limit: int = 1800) -> list[str]:
    lines = text.split("\n")
    caption, rows = lines[0], lines[1:]
    pieces: list[str] = []
    current: list[str] = []
    size = len(caption)
    for row in rows:
        if current and size + len(row) > limit:
            pieces.append("\n".join([caption + " (continued)" if pieces else caption, *current]))
            current, size = [], len(caption)
        current.append(row)
        size += len(row) + 1
    if current or not pieces:
        pieces.append("\n".join([caption + (" (continued)" if pieces else ""), *current]))
    return pieces


def chunk_report(elements: list[Element], title: str, shape: str) -> tuple[list[Section], list[ReportChunk]]:
    settings = get_settings()
    # Chunk size follows the document's shape: dense, fact-per-line visual
    # reports get small chunks so one number is not diluted by its
    # neighbours; long analytical prose gets larger ones so an argument stays
    # in one piece.
    target = {"visual": 700, "long_structured": settings.report_chunk_chars + 200}.get(shape, settings.report_chunk_chars)
    overlap = settings.report_chunk_overlap

    sections, owner = build_sections(elements)
    footnotes: dict[str, list[Element]] = {}
    for element in elements:
        if element.kind == "footnote":
            footnotes.setdefault(element.note_id, []).append(element)

    def resolve_note(note_id: str, page: int) -> Element | None:
        """Footnote numbers restart per page or chapter; pick the right one."""
        candidates = footnotes.get(note_id) or []
        if not candidates:
            return None
        same_page = [c for c in candidates if c.page == page]
        if same_page:
            return same_page[0]
        endnotes = [c for c in candidates if c.source == "endnotes"]
        if endnotes:
            return endnotes[0]
        return min(candidates, key=lambda c: abs(c.page - page))

    chunks: list[ReportChunk] = []

    def add(text: str, kind: str, section: Section, page_from: int, page_to: int) -> None:
        text = text.strip()
        if len(text) < 25:
            return
        ids = list(dict.fromkeys(_MARKER.findall(text)))
        notes = []
        for note_id in ids[:6]:
            note = resolve_note(note_id, page_from)
            if note is not None:
                notes.append(f"Footnote {note_id}: {note.text}")
        location = f"p.{page_from}" if page_from == page_to else f"pp.{page_from}-{page_to}"
        label = {"table": "Table", "figure": "Chart/figure (transcribed)", "footnote": "Footnote", "note": "Figure note/caption"}.get(kind)
        header = f"[{title} | {section.path} | {location}" + (f" | {label}" if label else "") + "]"
        chunks.append(ReportChunk(
            text=text, content_type=kind, section_ordinal=section.ordinal, section_path=section.path,
            page_from=page_from, page_to=page_to, ordinal=len(chunks), header=header,
            note_ids=ids, note_text="\n".join(notes),
        ))

    # Walk the stream section by section, keeping element order.
    groups: dict[int, list[Element]] = {}
    order: list[int] = []
    for element, ordinal in zip(elements, owner):
        if element.kind == "heading":
            continue
        if ordinal not in groups:
            groups[ordinal] = []
            order.append(ordinal)
        groups[ordinal].append(element)

    for ordinal in order:
        section = sections[ordinal]
        items = groups[ordinal]
        prose = [(e.text, e.page) for e in items if e.kind == "text"]
        notes = [(e.text, e.page) for e in items if e.kind == "note"]
        # A section's heading is part of its first chunk too: it is often the
        # only place the topic word appears.
        if prose and section.ordinal:
            prose[0] = (f"{section.title}\n{prose[0][0]}", prose[0][1])
        for text, page_from, page_to in _pack(prose, target, overlap):
            add(text, "text", section, page_from, page_to)
        for text, page_from, page_to in _pack(notes, target, 0):
            add(text, "note", section, page_from, page_to)
        for element in items:
            if element.kind == "table":
                for piece in _split_table(element.text):
                    add(piece, "table", section, element.page, element.page)
            elif element.kind == "figure":
                add(element.text, "figure", section, element.page, element.page)
            elif element.kind == "footnote":
                add(f"Footnote {element.note_id}: {element.text}", "footnote", section, element.page, element.page)

    for index, chunk in enumerate(chunks):
        chunk.ordinal = index
    return sections, chunks


def classify_shape(stats: dict) -> str:
    """Name the document's shape, which selects the chunking/retrieval recipe.

    * long_structured -- many pages with a real chapter tree: needs routing.
    * visual          -- short, with facts in graphics: needs vision + tables.
    * narrative       -- everything else: the baseline recipe.
    """
    pages = stats.get("pages", 0)
    visual_ratio = stats.get("drawing_heavy_pages", 0) / max(1, pages)
    if pages >= 120:
        return "long_structured"
    if pages <= 30 and (visual_ratio >= 0.3 or stats.get("figures", 0) >= 2):
        return "visual"
    return "narrative"


STRATEGIES = {
    "narrative": [
        "Paragraph-packed chunks (~1,200 chars) with sentence-safe overlap",
        "Run-in and layout headings recovered so chunks stay inside one topic",
        "Running headers/footers stripped before chunking",
        "Hybrid dense + BM25 retrieval, cross-encoder rerank, neighbour expansion",
    ],
    "long_structured": [
        "PDF outline re-levelled into a chapter → section → subsection tree",
        "Chunks never cross a section; each carries its full section path",
        "Query routing to chapters, boxes and regions before vector search",
        "Larger chunks (~1,400 chars) + neighbour expansion for context",
        "Borderless forecast tables rebuilt as labelled records",
    ],
    "visual": [
        "Charts, icon call-outs and infographics transcribed by a vision model",
        "Tables rebuilt from page geometry as labelled records",
        "Footnote markers linked to their notes and embedded together",
        "Small chunks (~700 chars) so each number keeps its own vector",
        "Quantitative questions guarantee a table/figure in context",
    ],
}
