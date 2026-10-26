"""Layout-aware PDF parsing for long and visual reports.

`parsing.py` treats a PDF as a stream of text lines, which is fine for a
three-page proposal and wrong for the documents the Report Lab is built for:

* a 200-page economic report whose chapter structure lives in the PDF outline
  and in font sizes, not in "3.1"-style numbering;
* an annual report whose running header ("2 0 1 8 A N N U A L R E P O R T")
  appears on every page and would otherwise land in every chunk;
* an environmental report whose key numbers sit in charts, icon call-outs,
  borderless tables and footnotes rather than in sentences.

So this parser reads the page *geometry* (PyMuPDF spans with font size,
weight and position) and emits typed `Element`s -- heading, text, table,
footnote, figure -- that `report_chunking.py` turns into section-aware chunks.

Nothing here is specific to the three sample documents. Every rule is driven
by a property of the page (font statistics, repetition across pages, numeric
density of a row) so that it transfers to any report of the same shape.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_settings

# --- data model -------------------------------------------------------------


@dataclass
class Element:
    kind: str            # heading | text | table | footnote | figure
    text: str
    page: int
    level: int = 0       # headings only
    note_id: str = ""    # footnotes only: "18"
    source: str = ""     # how it was found: outline | font | layout | vision | ...


@dataclass
class ParsedReport:
    elements: list[Element] = field(default_factory=list)
    page_count: int = 0
    title: str = ""
    stats: dict = field(default_factory=dict)


# --- text repair ------------------------------------------------------------

# Some typesetting pipelines encode "Th" as a single ligature glyph with no
# Unicode mapping, so extraction drops it: "Te global economy", "Tis box".
# Only repaired when the document demonstrably has the defect, because "Te"
# and "Tis" are not English words but "Ten" and "Tree" are.
_TH_LIGATURE_WORDS = {
    "Te": "The", "Tis": "This", "Tey": "They", "Tese": "These", "Tose": "Those",
    "Tere": "There", "Teir": "Their", "Tus": "Thus", "Tird": "Third",
    "Trough": "Through", "Tat": "That", "Ten": None,  # ambiguous: never repaired
}
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
_SOFT_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")


def _needs_th_repair(sample: str) -> bool:
    return len(re.findall(r"\bTe [a-z]", sample)) >= 3


def _repair(text: str, th_repair: bool) -> str:
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    if th_repair:
        for bad, good in _TH_LIGATURE_WORDS.items():
            if good:
                text = re.sub(rf"\b{bad}\b(?=\s+[a-z0-9])", good, text)
    return text


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines, healing words hyphenated across the break."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        elif out:
            out += " " + line
        else:
            out = line
    return re.sub(r"\s+", " ", out).strip()


# --- page model -------------------------------------------------------------


@dataclass
class _Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool
    page: int
    block: int
    cells: list[tuple[float, float, str]] = field(default_factory=list)  # (x0, x1, text)
    font: str = ""
    run_in: str = ""     # "History" in "History  From the Mayflower Compact..."

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


_NUMERIC_CELL = re.compile(r"^[\(\[]?[-–−+]?\$?\d[\d,]*(\.\d+)?%?[\)\]]?[a-z]?$|^\.\.$|^–$|^-$|^n/?a$", re.I)
_NOTE_MARK = re.compile(r"^\d{1,2}$")


def _read_page_lines(page, page_no: int, th_repair: bool) -> list[_Line]:
    """Every visual line on the page, with superscript note markers tagged."""
    lines: list[_Line] = []
    data = page.get_text("dict", flags=0)
    for b_index, block in enumerate(data.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for raw in block.get("lines", []):
            spans = [s for s in raw.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            body_size = max(s["size"] for s in spans)
            parts: list[str] = []
            cells: list[tuple[float, float, str]] = []
            for s in spans:
                txt = s["text"]
                # A short digit run set noticeably smaller than its line and
                # raised above it is a footnote marker ("STAR.18"). Tag it so
                # the chunker can attach note 18 to this sentence.
                is_marker = (
                    _NOTE_MARK.match(txt.strip())
                    and parts
                    and (
                        (s["flags"] & 1 and s["size"] < body_size * 0.9)  # flagged superscript
                        or (s["size"] < body_size * 0.8 and s["bbox"][3] < raw["bbox"][3] - 1)  # small and raised
                    )
                )
                if is_marker:
                    parts.append(f"[^{txt.strip()}]")
                    continue
                parts.append(txt)
                # Cells: spans separated by a visible horizontal gap.
                if cells and s["bbox"][0] - cells[-1][1] < 6:
                    x0, _x1, prev = cells[-1]
                    cells[-1] = (x0, s["bbox"][2], prev + txt)
                else:
                    cells.append((s["bbox"][0], s["bbox"][2], txt))
            text = _repair("".join(parts), th_repair).strip()
            text = re.sub(r"\s+(\[\^\d+\])", r"\1", text)
            if not text:
                continue
            # A superscript is sometimes extracted as its own "line": a lone
            # small number hugging the end of the previous line. Re-attach it
            # as a marker instead of letting it read as the value "18".
            prev = lines[-1] if lines and lines[-1].block == b_index else None
            if (
                prev is not None
                and _NOTE_MARK.match(text)
                and body_size < prev.size * 0.8
                and 0 <= raw["bbox"][0] - prev.x1 < 4
                and raw["bbox"][1] < prev.y1 and raw["bbox"][3] > prev.y0
            ):
                prev.text += f"[^{text}]"
                prev.x1 = raw["bbox"][2]
                continue
            first = spans[0]
            run_in = _run_in_heading(spans, th_repair)
            if run_in and text.startswith(run_in):
                text = text[len(run_in):].strip()
            x0, y0, x1, y1 = raw["bbox"]
            lines.append(
                _Line(
                    text=text, x0=x0, y0=y0, x1=x1, y1=y1,
                    size=round(first["size"], 1),
                    bold=bool(first["flags"] & 16) or "bold" in first.get("font", "").lower(),
                    page=page_no, block=b_index,
                    cells=[(a, b, _repair(c, th_repair).strip()) for a, b, c in cells if c.strip()],
                    font=first.get("font", ""), run_in=run_in,
                )
            )
    return lines


_HEAVY_FONT = re.compile(r"(bold|black|heavy|semi|demi|medium|-md\b|-bd\b|-sb\b)", re.I)


def _run_in_heading(spans: list[dict], th_repair: bool) -> str:
    """A heading set inline at the start of a paragraph, in a heavier face.

    Annual reports love this layout: "History  From the Mayflower Compact..."
    where "History" is Helvetica Medium and the paragraph is Garamond. It has
    the same size as body text, so size-based detection never sees it.
    """
    if len(spans) < 2:
        return ""
    lead_font = spans[0].get("font", "")
    if lead_font == spans[-1].get("font", "") or not (_HEAVY_FONT.search(lead_font) or spans[0]["flags"] & 16):
        return ""
    lead = ""
    for span in spans:
        if span.get("font", "") != lead_font:
            break
        lead += span["text"]
    lead = _repair(lead, th_repair).strip()
    if 2 < len(lead) <= 60 and _looks_like_title(lead) and not lead.endswith((".", ",")):
        return lead
    return ""


# --- running headers and footers -------------------------------------------

def _furniture_key(text: str) -> str:
    """Normalise a margin line so page 4's and page 40's header compare equal."""
    key = re.sub(r"\d+", "#", text.lower())
    return re.sub(r"[^a-z#|]+", "", key)


def _find_furniture(pages: list[list[_Line]], heights: list[float]) -> set[str]:
    """Lines repeated in the top/bottom margin of many pages are page furniture."""
    counts: Counter[str] = Counter()
    for lines, height in zip(pages, heights):
        seen: set[str] = set()
        for line in lines:
            if line.y1 < height * 0.09 or line.y0 > height * 0.91:
                key = _furniture_key(line.text)
                if key and key not in seen:
                    seen.add(key)
                    counts[key] += 1
    threshold = max(3, int(len(pages) * 0.2))
    return {key for key, n in counts.items() if n >= threshold}


def _is_furniture(line: _Line, height: float, furniture: set[str]) -> bool:
    in_margin = line.y1 < height * 0.09 or line.y0 > height * 0.91
    if not in_margin:
        return False
    if re.fullmatch(r"[ivxlc\d]{1,5}", line.text.strip().lower()):
        return True  # bare page number (arabic or roman)
    return _furniture_key(line.text) in furniture


# --- tables rebuilt from geometry -----------------------------------------

def _rows_by_baseline(lines: list[_Line]) -> list[list[_Line]]:
    rows: list[list[_Line]] = []
    for line in sorted(lines, key=lambda l: (round(l.yc, 0), l.x0)):
        if rows and abs(rows[-1][0].yc - line.yc) <= 2.5:
            rows[-1].append(line)
        else:
            rows.append([line])
    return [sorted(r, key=lambda l: l.x0) for r in rows]


def _row_cells(row: list[_Line]) -> list[tuple[float, float, str]]:
    cells: list[tuple[float, float, str]] = []
    for line in row:
        cells.extend(line.cells or [(line.x0, line.x1, line.text)])
    return sorted(cells, key=lambda c: c[0])


def _is_numeric_row(cells: list[tuple[float, float, str]]) -> bool:
    if len(cells) < 3:
        return False
    values = [c[2] for c in cells[1:]]
    numeric = sum(1 for v in values if _NUMERIC_CELL.match(v.replace(" ", "")))
    return numeric >= max(2, int(len(values) * 0.6))


_YEAR_CELL = re.compile(r"^(19|20)\d\d([/–-]\d{2,4})?[a-z]?$|^q[1-4]\b", re.I)


def _is_year_header(cells: list[tuple[float, float, str]]) -> bool:
    """A column-header row: several period labels ("2023", "2025e", "2026/27f")."""
    years = sum(1 for c in cells if _YEAR_CELL.match(c[2].replace(" ", "")))
    return years >= 3 and years >= (len(cells) - 1) * 0.6


def _table_caption(rows: list[list[_Line]], first_row: int) -> tuple[str, set[int]]:
    """The caption above a table: "TABLE 2.9 South Asia country forecasts"."""
    used: set[int] = set()
    top = rows[first_row][0].y0
    for k in range(first_row - 1, max(-1, first_row - 8), -1):
        row = rows[k]
        if top - row[0].y1 > 90:
            break
        text = row[0].text
        if re.match(r"^(table|figure|exhibit|chart)\s*[\dA-Z]", text, re.I):
            caption = text
            used.update(id(l) for l in row)
            # A units line directly under the caption belongs to it.
            if k + 1 < first_row:
                nxt = " ".join(l.text for l in rows[k + 1])
                if nxt.startswith("("):
                    caption += " " + nxt
                    used.update(id(l) for l in rows[k + 1])
            return caption, used
    return "", used


def _extract_tables(lines: list[_Line], page_no: int) -> tuple[list[Element], set[int]]:
    """Find borderless numeric tables by row geometry.

    A run of three or more baseline-aligned rows whose cells are mostly
    numbers is a table, whatever the PDF's drawing commands say. Each data row
    is serialised as a *record* -- "Nepal | 2025e: 4.6 | 2026f: 2.3" -- so a
    row carries its own column labels into whatever chunk it lands in. A bare
    "4.6 2.3 4.2" is unanswerable once separated from its header; a record is
    not.

    Column headers are period rows ("2023 2024 2025e 2026f"), which may recur
    mid-table when a table switches from calendar to fiscal years; each data
    row is matched to the nearest header cell by horizontal position.
    """
    rows = _rows_by_baseline(lines)
    used: set[int] = set()
    tables: list[Element] = []

    def flush(state: dict) -> None:
        if state["numeric"] >= 3:
            caption, cap_used = _table_caption(rows, state["first"])
            used.update(cap_used)
            used.update(state["ids"])
            tables.append(Element(
                kind="table",
                text=(caption or f"Table on page {page_no}") + "\n" + "\n".join(state["records"]),
                page=page_no, source="geometry",
            ))

    state: dict | None = None
    for index, row in enumerate(rows):
        cells = _row_cells(row)
        ids = {id(l) for l in row}
        if _is_year_header(cells):
            columns = [((x0 + x1) / 2, text) for x0, x1, text in cells if _YEAR_CELL.match(text.replace(" ", ""))]
            label = cells[0][2] if not _YEAR_CELL.match(cells[0][2].replace(" ", "")) else ""
            if state is None or (state["records"] and row[0].y0 - state["last_y"] > 30):
                if state:
                    flush(state)
                state = {"first": index, "columns": columns, "records": [], "numeric": 0, "ids": set(), "last_y": row[0].y1}
            state["columns"] = columns
            state["ids"].update(ids)
            if label:
                state["records"].append(f"{label}:")
            state["last_y"] = row[0].y1
            continue

        if _is_numeric_row(cells):
            if state is None:
                state = {"first": index, "columns": [], "records": [], "numeric": 0, "ids": set(), "last_y": row[0].y1}
            label = cells[0][2]
            values = []
            for x0, x1, value in cells[1:]:
                header = ""
                if state["columns"]:
                    centre = (x0 + x1) / 2
                    header = min(state["columns"], key=lambda c: abs(c[0] - centre))[1]
                values.append(f"{header}: {value}" if header else value)
            state["records"].append(f"{label} | " + " | ".join(values))
            state["numeric"] += 1
            state["ids"].update(ids)
            state["last_y"] = row[0].y1
            continue

        if state is not None:
            text = " ".join(c[2] for c in cells)
            gap = row[0].y0 - state["last_y"]
            # A short label row inside a table is a group heading
            # ("Calendar-year basis", "EMDEs"); anything else ends the table.
            is_label = len(cells) <= 2 and (len(text) <= 60 or text.startswith("("))
            if gap < 24 and is_label and not re.match(r"^(source|note)s?\b", text, re.I):
                state["records"].append(f"{text}:")
                state["ids"].update(ids)
                state["last_y"] = row[0].y1
                continue
            flush(state)
            state = None
    if state is not None:
        flush(state)
    tables.extend(_key_value_tables(rows, used, page_no))
    return tables, used


def _is_kv_row(row: list[_Line]) -> bool:
    cells = _row_cells(row)
    if len(cells) != 2:
        return False
    # Same type size on both sides: a prose line beside a chart's axis tick
    # is a coincidence of baselines, not a label and its value.
    if max(l.size for l in row) - min(l.size for l in row) > 0.6:
        return False
    (_lx0, lx1, label), (vx0, _vx1, value) = cells
    return (
        vx0 - lx1 > 15
        and len(label.split()) <= 8
        and not label.rstrip().endswith((",", ";"))
        and len(value) <= 24
        and any(ch.isdigit() for ch in value)
        and sum(ch.isalpha() for ch in label) >= 3
    )


def _key_value_tables(rows: list[list[_Line]], used: set[int], page_no: int) -> list[Element]:
    """Two-column fact panels: "Total product footprint ........ 346 kg CO2e".

    Spec sheets and environmental reports present their headline numbers as
    label/value pairs set far apart on one baseline. Each pair alone is a
    16-character fragment that looks like chart noise; together they are the
    most important table on the page.
    """
    tables: list[Element] = []
    run: list[int] = []

    def flush() -> None:
        if len(run) >= 2:
            first = rows[run[0]][0]
            caption_parts: list[str] = []
            for k in range(run[0] - 1, max(-1, run[0] - 4), -1):
                row = rows[k]
                if any(id(l) in used for l in row) or first.y0 - row[0].y1 > 35:
                    break
                text = " ".join(c[2] for c in _row_cells(row))
                if len(text) > 70:
                    break
                caption_parts.insert(0, text)
                used.update(id(l) for l in row)
            records = []
            for index in run:
                label, value = (c[2] for c in _row_cells(rows[index]))
                records.append(f"{label.lstrip('•·- ').strip()}: {value}")
                used.update(id(l) for l in rows[index])
            caption = " — ".join(caption_parts) or f"Key figures on page {page_no}"
            tables.append(Element("table", caption + "\n" + "\n".join(records), page_no, source="key-value"))
        run.clear()

    for index, row in enumerate(rows):
        if any(id(l) in used for l in row):
            flush()
            continue
        if _is_kv_row(row):
            if run and row[0].y0 - rows[run[-1]][0].y1 > 30:
                flush()
            run.append(index)
        elif run and len(_row_cells(row)) == 1 and len(row[0].text) < 60 and row[0].y0 - rows[run[-1]][0].y1 < 12:
            continue  # a stray label line inside the panel
        else:
            flush()
    flush()
    return tables


# --- headings ---------------------------------------------------------------

_TOC_JUNK = re.compile(r"\.(pdf|docx?)$|^\s*(contents|summary of contents|table of contents)\s*$", re.I)


def _clean_title(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"(^|\s)-{2,}\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _outline_level(title: str, raw_level: int) -> int:
    """Re-level outline entries by what they say, not how they were nested.

    Report outlines are often flat (every entry level 1) or inconsistent. The
    *title* is the more reliable signal: "Ch. 2" is a chapter, "Box 2.1" and a
    regional sub-entry live inside one.
    """
    if re.match(r"^(ch\.?|chapter|part)\s*\d", title, re.I):
        return 1
    if re.match(r"^(box|annex|appendix)\s*[\d.a-z]", title, re.I):
        return 2
    return raw_level


def _outline(doc) -> list[tuple[int, str, int, bool]]:
    """(level, title, page, indented) from the PDF bookmarks, junk removed."""
    entries = []
    for level, title, page, *_ in doc.get_toc(simple=False):
        clean = _clean_title(title)
        if page < 1 or not clean or _TOC_JUNK.search(clean):
            continue
        # Bookmarks that are whole sentences were generated from body text.
        if len(clean) > 90 or clean.endswith((":", ".")):
            continue
        indented = "--" in title
        entries.append((level, clean, page, indented))
    repeats = Counter(e[1].lower() for e in entries)
    return [e for e in entries if repeats[e[1].lower()] < 3]


def _norm_title(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _resolve_outline_pages(outline: list[tuple[int, str, int, bool]], pages: list[list[_Line]]) -> list[tuple[int, str, int, bool]]:
    """Move each bookmark to the page where its title is actually printed.

    Bookmarks frequently point one page early (to the facing page, or to the
    page the previous section's trailing table sits on). Taken literally,
    "South Asia" would start on the page holding the Middle East forecast
    table, and that table would be filed -- and retrieved -- as South Asia.
    """
    def block_texts(page_no: int) -> list[str]:
        if not 1 <= page_no <= len(pages):
            return []
        blocks: dict[int, list[str]] = defaultdict(list)
        for line in pages[page_no - 1]:
            blocks[line.block].append(line.text)
        return [_norm_title(_join_lines(v)) for v in blocks.values()]

    resolved = []
    for level, title, page, indented in outline:
        target = _norm_title(title)
        found = page
        for candidate in (page, page + 1, page - 1):
            if any(t == target or (len(target) > 6 and t.startswith(target)) for t in block_texts(candidate)):
                found = candidate
                break
        resolved.append((level, title, found, indented))
    return resolved


def _body_size(all_lines: list[_Line]) -> float:
    weights: Counter[float] = Counter()
    for line in all_lines:
        weights[line.size] += len(line.text)
    return weights.most_common(1)[0][0] if weights else 10.0


def _family(font: str) -> str:
    """'HelveticaNeueLTStd-Md' -> 'helveticaneueltstd'."""
    return re.split(r"[-,+]", font.split("+")[-1])[0].lower()


def _body_family(lines: list[_Line]) -> str:
    weights: Counter[str] = Counter()
    for line in lines:
        weights[_family(line.font)] += len(line.text)
    return weights.most_common(1)[0][0] if weights else ""


def _looks_like_title(text: str) -> bool:
    words = text.split()
    if not words or len(words) > 14 or len(text) > 110:
        return False
    if text.endswith((".", ",", ";")) and len(words) > 4:
        return False
    if sum(c.isdigit() for c in text) > len(text) * 0.3:
        return False
    return text[0].isupper() or text[0].isdigit()


# --- vision ---------------------------------------------------------------

VISION_PROMPT = """This is page {page} of the document "{title}".
Extract every fact that is shown VISUALLY on this page: values in charts and graphs, \
numbers in tables, icon call-outs and infographic statistics, and labelled diagrams.

Rules:
- One fact per line, written as a complete standalone sentence that names what is being measured, \
e.g. "iMac total product carbon footprint: 346 kg CO2e."
- Copy numbers, units and percentages exactly as printed. Never estimate a value that is not printed.
- For a chart, state its title, then each labelled bar/segment with its value.
- Keep footnote markers as [^n] when a superscript number follows a value.
- Skip ordinary paragraph text, page headers and navigation.
- If the page has no visual facts, answer exactly: NONE"""


def _vision_cache_path(digest: str) -> Path:
    path = get_settings().data_dir / "vision_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{digest}.json"


def _vision_page(doc_path: Path, page_index: int, title: str) -> str:
    """Ask the multimodal model to transcribe a page's charts and call-outs."""
    import pymupdf
    from google import genai
    from google.genai import types

    settings = get_settings()
    with pymupdf.open(str(doc_path)) as doc:
        png = doc[page_index].get_pixmap(dpi=settings.vision_dpi).tobytes("png")
    digest = hashlib.sha256(png + settings.gemini_model.encode()).hexdigest()[:24]
    cache = _vision_cache_path(digest)
    if cache.exists():
        return json.loads(cache.read_text())["text"]

    client = genai.Client(api_key=settings.gemini_api_key)
    prompt = VISION_PROMPT.format(page=page_index + 1, title=title)
    last: Exception | None = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=[types.Part.from_bytes(data=png, mime_type="image/png"), prompt],
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=2048),
            )
            text = (response.text or "").strip()
            cache.write_text(json.dumps({"page": page_index + 1, "text": text}))
            return text
        except Exception as exc:  # 429/503 on the free tier
            last = exc
            import time

            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"vision failed on page {page_index + 1}: {last}")


def _visual_pages(doc, text_chars: list[int]) -> list[int]:
    """Pages where facts live in graphics: many vector drawings or images."""
    candidates = []
    for index, page in enumerate(doc):
        drawings = len(page.get_drawings())
        images = len(page.get_images())
        if drawings >= 10 or images >= 1:
            candidates.append(index)
    return candidates


# --- main entry point -------------------------------------------------------

def parse_report(path: Path, use_vision: bool | None = None, progress=None) -> ParsedReport:
    import pymupdf

    settings = get_settings()
    progress = progress or (lambda *_: None)
    doc = pymupdf.open(str(path))
    report = ParsedReport(page_count=doc.page_count)
    meta_title = (doc.metadata or {}).get("title") or ""
    report.title = meta_title if 4 < len(meta_title) < 120 and not meta_title.lower().endswith(".pdf") else ""

    sample = "".join(doc[i].get_text() for i in range(min(doc.page_count, 25)))
    th_repair = _needs_th_repair(sample)

    progress("parsing", f"Reading layout of {doc.page_count} pages")
    pages = [_read_page_lines(page, index + 1, th_repair) for index, page in enumerate(doc)]
    heights = [page.rect.height for page in doc]
    furniture = _find_furniture(pages, heights)
    pages = [
        [l for l in lines if not _is_furniture(l, h, furniture)] for lines, h in zip(pages, heights)
    ]
    all_lines = [l for lines in pages for l in lines]
    body = _body_size(all_lines)
    body_family = _body_family(all_lines)

    # Heading font tiers: sizes clearly above body text, largest first.
    heading_sizes = sorted(
        {l.size for l in all_lines if l.size >= body + 1.5 and _looks_like_title(l.text)}, reverse=True
    )[:4]

    outline = _resolve_outline_pages(_outline(doc), pages)
    outline_by_page: dict[int, list[tuple[int, str, bool]]] = defaultdict(list)
    for level, title, page, indented in outline:
        outline_by_page[page].append((level, title, indented))

    stats = Counter()
    elements: list[Element] = []
    footnote_zone = 0.80

    for page_no, lines in enumerate(pages, start=1):
        height = heights[page_no - 1]
        page_body = _body_size(lines) if sum(len(l.text) for l in lines) > 600 else body
        page_family = _body_family(lines) or body_family
        tables, used = _extract_tables(lines, page_no)
        stats["tables"] += len(tables)

        # Outline headings for this page (allowing the off-by-one that
        # printed-page vs PDF-page numbering causes).
        pending_outline = list(outline_by_page.get(page_no, []))
        matched_outline: set[int] = set()

        blocks: dict[int, list[_Line]] = defaultdict(list)
        for line in lines:
            if id(line) not in used:
                blocks[line.block].append(line)

        page_elements: list[Element] = []
        for _b, block_lines in sorted(blocks.items(), key=lambda kv: kv[0]):
            block_text = _join_lines([l.text for l in block_lines])
            if not block_text:
                continue
            first = block_lines[0]
            size = max(l.size for l in block_lines)

            # 1. outline match: a bookmark whose title is this block.
            outline_hit = None
            norm = re.sub(r"\W+", " ", block_text.lower()).strip()
            for idx, (level, title, indented) in enumerate(pending_outline):
                t = re.sub(r"\W+", " ", title.lower()).strip()
                if t and (norm == t or (len(t) > 6 and norm.startswith(t)) or (len(norm) > 6 and t.endswith(norm))):
                    outline_hit = (idx, level, title, indented)
                    break
            # 0. a run-in heading opens the block: split it off.
            if first.run_in:
                page_elements.append(Element("heading", first.run_in, page_no, level=6, source="run-in"))
                stats["run_in_headings"] += 1

            if outline_hit and len(block_text) < 120:
                idx, level, title, indented = outline_hit
                matched_outline.add(idx)
                lvl = 2 if indented else _outline_level(title, level)
                page_elements.append(Element("heading", title, page_no, level=lvl, source="outline"))
                stats["outline_headings"] += 1
                continue

            # 2. font-tier heading: larger than body text, short, title-like.
            if size in heading_sizes and _looks_like_title(block_text) and len(block_lines) <= 3:
                tier = heading_sizes.index(size)
                page_elements.append(Element("heading", block_text, page_no, level=2 + tier, source="font"))
                stats["font_headings"] += 1
                continue

            # 3. footnotes: small text in the bottom band starting with a number.
            small = size < page_body * 0.85
            if small and first.y0 > height * footnote_zone and re.match(r"^\d{1,3}[.\s]", block_text):
                for note in re.split(r"(?:(?<=\s)|^)(?=\d{1,3}\.?\s+[A-Z])", block_text):
                    m = re.match(r"^(\d{1,3})\.?\s+(.*)", note.strip())
                    if m:
                        page_elements.append(
                            Element("footnote", m.group(2).strip(), page_no, note_id=m.group(1), source="page-foot")
                        )
                        stats["footnotes"] += 1
                continue

            # 4. a minor heading: one short title-like line standing alone,
            # either in body size ("Recent developments", "Outlook") or in a
            # different type family from the body ("A Message from the
            # Acting Deputy Director" in Helvetica over Garamond text).
            other_family = _family(first.font) != page_family
            if (
                len(block_lines) == 1
                and (abs(size - page_body) < 0.6 or (other_family and size >= page_body * 0.95))
                and size >= body * 0.9  # chart-heavy pages pull page_body down to label size
                and _looks_like_title(block_text)
                and len(block_text.split()) <= (10 if other_family else 6)
                and not block_text.endswith((".", ":", ","))
                and not re.search(r"\d", block_text)
                and block_text[:1].isupper()
            ):
                page_elements.append(Element("heading", block_text.rstrip(":").strip(), page_no, level=6, source="layout"))
                stats["minor_headings"] += 1
                continue

            kind = "note" if small else "text"
            if kind == "note":
                letters = sum(c.isalpha() for c in block_text)
                if len(block_text) < 25 or letters < len(block_text) * 0.5:
                    stats["axis_labels_dropped"] += 1
                    continue
            page_elements.append(Element(kind, block_text, page_no, source="layout"))

        # Bookmarks that matched no block still mark where a section starts.
        unmatched = [
            Element("heading", title, page_no, level=2 if indented else _outline_level(title, level), source="outline")
            for idx, (level, title, indented) in enumerate(pending_outline)
            if idx not in matched_outline
        ]
        stats["outline_headings"] += len(unmatched)
        page_elements = unmatched + page_elements

        elements.extend(page_elements)
        elements.extend(tables)

    # Endnote pages: a heading like "Endnotes"/"Notes" followed by numbered
    # paragraphs. Each note becomes its own footnote element.
    elements = _merge_fragments(elements, body)
    elements = _split_endnotes(elements, stats)

    # Vision pass for short, visual documents. A 12-page environmental report
    # costs ~8 calls; a 200-page economic report would cost 100+ and its
    # charts are already described by captions and notes, so it is skipped.
    if use_vision is None:
        use_vision = settings.vision_enabled and doc.page_count <= settings.vision_max_pages
    if use_vision and not settings.gemini_api_key:
        use_vision = False
    if use_vision:
        visual = _visual_pages(doc, [])
        progress("vision", f"Reading charts and call-outs on {len(visual)} visual pages")
        title = report.title or path.stem
        figures: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=settings.vision_workers) as pool:
            futures = {pool.submit(_vision_page, path, index, title): index for index in visual}
            for future, index in futures.items():
                try:
                    figures[index] = future.result()
                except Exception as exc:
                    print(f"[report_parsing] {exc}")
                    stats["vision_failures"] += 1
        for index in sorted(figures):
            text = figures[index].strip()
            if not text or text.upper().startswith("NONE"):
                continue
            facts = [ln.strip(" -•*") for ln in text.splitlines() if ln.strip(" -•*")]
            elements.append(Element("figure", "\n".join(facts), index + 1, source="vision"))
            stats["figures"] += 1
        stats["vision_pages"] = len(visual)

    elements.sort(key=lambda e: e.page)  # stable: in-page order is preserved
    if not report.title:
        report.title = _guess_title(elements, path)
    stats["pages"] = doc.page_count
    stats["outline_entries"] = len(outline)
    stats["th_ligature_repair"] = int(th_repair)
    stats["furniture_patterns_removed"] = len(furniture)
    stats["drawing_heavy_pages"] = sum(1 for p in doc if len(p.get_drawings()) >= 25)
    report.elements = elements
    report.stats = dict(stats)
    doc.close()
    return report


def _merge_fragments(elements: list[Element], body: float) -> list[Element]:
    """Repair two layout artefacts that split one thing into several elements.

    * A drop cap -- the giant "T" opening a chapter -- is extracted as its own
      one-letter block and would otherwise become a heading.
    * A display title set over several lines ("CONSOLIDATED FINANCIAL" /
      "STATEMENTS") arrives as consecutive headings of the same tier.
    """
    out: list[Element] = []
    carry = ""
    for element in elements:
        if element.kind == "heading" and len(element.text.strip()) <= 2 and element.text.strip().isalpha():
            carry = element.text.strip()
            continue
        if carry and element.kind == "text":
            element.text = carry + element.text
            carry = ""
        if (
            out
            and element.kind == "heading"
            and out[-1].kind == "heading"
            and out[-1].page == element.page
            and out[-1].level == element.level
            and out[-1].source == element.source == "font"
        ):
            out[-1].text = f"{out[-1].text} {element.text}"
            continue
        out.append(element)

    # A "heading" found only by layout that is immediately followed by
    # another heading is not introducing anything -- it is a label in an org
    # chart, a list of names, a signature block. Demote it to text.
    for i, element in enumerate(out[:-1]):
        nxt = out[i + 1]
        if element.kind == "heading" and element.source == "layout" and nxt.kind == "heading" and nxt.source in {"layout", "run-in"}:
            element.kind = "text"
    return out


def _split_endnotes(elements: list[Element], stats: Counter) -> list[Element]:
    out: list[Element] = []
    in_notes = False
    for element in elements:
        if element.kind == "heading":
            in_notes = bool(re.match(r"^(end\s*notes|notes|footnotes)$", element.text.strip(), re.I))
            out.append(element)
            continue
        if in_notes and element.kind in {"text", "note"}:
            parts = re.split(r"(?:^|\s)(?=\d{1,3}\s+[A-Z])", element.text)
            found = False
            for part in parts:
                m = re.match(r"^(\d{1,3})\s+(.*)", part.strip(), re.S)
                if m:
                    out.append(Element("footnote", m.group(2).strip(), element.page, note_id=m.group(1), source="endnotes"))
                    stats["footnotes"] += 1
                    found = True
                elif part.strip() and out and out[-1].kind == "footnote":
                    out[-1].text += " " + part.strip()  # a note wrapped into the next block
            if found or not parts:
                continue
        out.append(element)
    return out


def _guess_title(elements: list[Element], path: Path) -> str:
    for element in elements[:20]:
        if element.kind == "heading" and 4 < len(element.text) < 90:
            return element.text
    return path.stem.replace("_", " ").replace("-", " ")

