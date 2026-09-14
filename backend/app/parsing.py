"""Document parsing: files in, structured blocks out.

Most RAG tutorials flatten a document to one long string and then cut it
every N characters. That destroys exactly the information a proposal library
depends on -- which *section* a paragraph came from. A sentence about
"$180,000" means something completely different under "Pricing" than under
"Risks we helped the client avoid".

So parsing here emits a list of `Block`s, each tagged with its page and
whether it is a heading. `chunking.py` then rebuilds the section hierarchy
from that stream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}


@dataclass
class Block:
    text: str
    page: int = 1
    is_heading: bool = False
    level: int = 0


@dataclass
class ParsedDocument:
    blocks: list[Block] = field(default_factory=list)
    page_count: int = 0

    @property
    def full_text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


# --- heading detection ------------------------------------------------------

# "3.", "3.1", "3.1.2" followed by a title. The dominant convention in RFPs,
# SOWs and proposal templates, and the highest-precision signal we have.
_NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+){0,3})[.)]?\s+(\S.{0,90})$")
_MD_HEADING = re.compile(r"^\s*(#{1,6})\s+(\S.*)$")
_BULLET = re.compile(r"^\s*([-*•●▪·]|\d+[.)])\s+")
# Words that make a short line a heading even without numbering or caps.
_SECTION_WORDS = (
    "executive summary", "scope of work", "scope", "pricing", "commercials",
    "cost", "timeline", "milestones", "deliverables", "team", "team structure",
    "technology stack", "tech stack", "architecture", "approach",
    "methodology", "assumptions", "risks", "case study", "about us",
    "why us", "engagement model", "sla", "support", "maintenance",
    "testing", "qa", "security", "compliance", "references", "appendix",
    "problem statement", "solution", "results", "outcomes", "challenges",
)


def _looks_like_heading(line: str) -> tuple[bool, int]:
    """Return (is_heading, level). Level 0 means 'unknown depth'."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False, 0
    if _BULLET.match(line):
        return False, 0

    m = _MD_HEADING.match(stripped)
    if m:
        return True, len(m.group(1))

    m = _NUMBERED.match(stripped)
    if m:
        # "1. We will deliver the following" is a sentence, not a heading.
        title = m.group(2)
        if title.endswith((".", ";", ",")) and len(title.split()) > 8:
            return False, 0
        return True, m.group(1).count(".") + 1

    words = stripped.split()
    if stripped.endswith((".", ",", ";", ":")) and len(words) > 10:
        return False, 0

    letters = [c for c in stripped if c.isalpha()]
    if letters and len(words) <= 12:
        upper_ratio = sum(c.isupper() for c in letters) / len(letters)
        if upper_ratio > 0.85:
            return True, 1
        low = stripped.rstrip(":").lower()
        if low in _SECTION_WORDS:
            return True, 1
        # A *prefix* match is much weaker evidence than an exact one, so it only
        # counts for a short line that reads like a label rather than a
        # sentence. "Scope of Work" is a heading; "Case study. Client: Vanguard
        # Motors AG. Delivered 2024." merely starts with one, and promoting it
        # would make the cover-page blurb the section heading for everything
        # that follows it.
        if (
            len(words) <= 6
            and not re.search(r"[.:;]\s+\S", stripped)
            and any(low.startswith(w) for w in _SECTION_WORDS)
        ):
            return True, 1

        # Title-cased short lines. PDF text extraction throws away the styling
        # that marks a heading, so a section titled "The Problem" arrives
        # looking exactly like body text. Three constraints keep this from
        # eating real content:
        #   * no terminal punctuation and at most six words (already checked);
        #   * at least 60% of words capitalised, which a wrapped prose line
        #     fails ("the client asked us to rebuild" scores 1/6);
        #   * no digits -- this is the one that matters. Without it, every row
        #     of a pricing table ("Months 1-3 USD 210,000") becomes its own
        #     heading, fragmenting the table into one-line sections and
        #     scattering the commercial terms that most need to stay together.
        if not any(c.isdigit() for c in stripped):
            capitalised = [w for w in words if w[:1].isalpha()]
            if capitalised and sum(1 for w in capitalised if w[0].isupper()) / len(capitalised) >= 0.6:
                return True, 1
    return False, 0


def _lines_to_blocks(text: str, page: int) -> list[Block]:
    blocks: list[Block] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            body = " ".join(buffer).strip()
            if body:
                blocks.append(Block(text=body, page=page))
            buffer.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        heading, level = _looks_like_heading(line)
        if heading:
            flush()
            clean = _MD_HEADING.sub(r"\2", line.strip())
            blocks.append(Block(text=clean, page=page, is_heading=True, level=level or 1))
        else:
            buffer.append(line.strip())
    flush()
    return blocks


# --- per-format parsers -----------------------------------------------------

def _parse_pdf(path: Path) -> ParsedDocument:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    doc = ParsedDocument(page_count=len(reader.pages))
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        doc.blocks.extend(_lines_to_blocks(text, page=index))
    return doc


def _parse_docx(path: Path) -> ParsedDocument:
    import docx

    source = docx.Document(str(path))
    doc = ParsedDocument(page_count=1)
    for para in source.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style else ""
        if style.startswith("heading"):
            # Word tells us the real outline level -- trust it over heuristics.
            level = int(style.replace("heading", "").strip() or 1)
            doc.blocks.append(Block(text=text, page=1, is_heading=True, level=level))
        elif style == "title":
            doc.blocks.append(Block(text=text, page=1, is_heading=True, level=1))
        else:
            doc.blocks.extend(_lines_to_blocks(text, page=1))

    # Tables carry the pricing and timeline grids in most proposals, so they
    # are flattened row-wise rather than dropped.
    for table in source.tables:
        rows = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            doc.blocks.append(Block(text="\n".join(rows), page=1))
    return doc


def _parse_text(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    doc = ParsedDocument(page_count=1)
    doc.blocks = _lines_to_blocks(text, page=1)
    return doc


def parse(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix == ".docx":
        return _parse_docx(path)
    if suffix in {".md", ".markdown", ".txt"}:
        return _parse_text(path)
    raise ValueError(
        f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )
