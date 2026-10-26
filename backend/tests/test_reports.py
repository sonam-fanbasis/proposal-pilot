"""Report Lab: layout parsing, chunking, routing and the end-to-end path.

The PDF is generated here with PyMuPDF so every structural feature under test
-- running headers, outline, font-tier headings, borderless numeric tables,
label/value panels, footnote markers and endnotes -- is present by
construction, and the tests never depend on the sample reports being on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")

from app.report_chunking import build_sections, chunk_report, classify_shape  # noqa: E402
from app.report_parsing import _repair, parse_report  # noqa: E402

BODY = (
    "Growth in the region is expected to soften to 6.3 percent in 2026, mainly reflecting "
    "higher energy prices and weaker remittances. Activity is projected to recover as "
    "disruptions wane, with growth averaging 7 percent a year in 2027-28. "
)


def _report_pdf(path: Path) -> Path:
    doc = pymupdf.open()

    def page_with_header(number: int):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 40), "WORLD OUTLOOK | JUNE 2026", fontsize=7)
        page.insert_text((300, 770), str(number), fontsize=8)
        return page

    # p1: executive summary
    page = page_with_header(1)
    page.insert_text((72, 90), "Executive Summary", fontsize=22, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 120, 540, 400), "Risks to the outlook remain tilted to the downside. " * 4, fontsize=11)

    # p2: a region, with a borderless forecast table
    page = page_with_header(2)
    page.insert_text((72, 90), "South Asia", fontsize=16, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 110, 540, 260), BODY * 2, fontsize=11)
    page.insert_text((72, 300), "TABLE 2.9 South Asia forecast summary", fontsize=10, fontname="hebo")
    years = ["2024", "2025e", "2026f", "2027f"]
    for i, year in enumerate(years):
        page.insert_text((250 + i * 70, 325), year, fontsize=8, fontname="hebo")
    rows = [("EMDE SAR, GDP", ["6.8", "7.0", "6.3", "6.9"]), ("India", ["7.1", "7.7", "6.6", "7.2"]),
            ("Bangladesh", ["4.2", "3.5", "3.8", "4.6"]), ("Nepal", ["3.7", "4.6", "2.3", "4.2"])]
    for r, (label, values) in enumerate(rows):
        y = 340 + r * 12
        page.insert_text((80, y), label, fontsize=8)
        for i, value in enumerate(values):
            page.insert_text((255 + i * 70, y), value, fontsize=8)

    # p3: another region, so routing has something to tell apart
    page = page_with_header(3)
    page.insert_text((72, 90), "Sub-Saharan Africa", fontsize=16, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 110, 540, 300), BODY.replace("6.3", "4.1") * 2, fontsize=11)

    # p4: a label/value fact panel and a footnote marker
    page = page_with_header(4)
    page.insert_text((72, 90), "Carbon Footprint", fontsize=16, fontname="hebo")
    for i, (label, value) in enumerate([("Total product footprint", "346 kg CO2e"), ("Production", "50%"), ("Product use", "45%")]):
        page.insert_text((220, 140 + i * 15), label, fontsize=7)
        page.insert_text((440, 140 + i * 15), value, fontsize=7)
    page.insert_htmlbox(
        pymupdf.Rect(72, 250, 540, 290),
        '<p style="font-size:11px">The device uses 58 percent less energy than the requirement.<sup>18</sup></p>',
    )

    # p5: endnotes
    page = page_with_header(5)
    page.insert_text((72, 90), "Endnotes", fontsize=16, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 110, 540, 300),
                        "18 Energy values are based on the ENERGY STAR Program Requirements for Computers.",
                        fontsize=8)

    doc.set_toc([[1, "Executive Summary", 1], [1, "Ch. 2 Regional Outlooks", 2], [2, "South Asia", 2],
                 [2, "Sub-Saharan Africa", 3]])
    out = path / "outlook.pdf"
    doc.save(str(out))
    return out


@pytest.fixture
def parsed(tmp_path):
    return parse_report(_report_pdf(tmp_path), use_vision=False)


def test_running_headers_and_page_numbers_are_removed(parsed):
    text = "\n".join(e.text for e in parsed.elements)
    assert "WORLD OUTLOOK" not in text
    assert parsed.stats["furniture_patterns_removed"] >= 1


def test_outline_and_font_headings_build_the_tree(parsed):
    headings = [e.text for e in parsed.elements if e.kind == "heading"]
    assert "Executive Summary" in headings
    assert "South Asia" in headings and "Sub-Saharan Africa" in headings
    sections, _ = build_sections(parsed.elements)
    paths = {s.path for s in sections}
    assert any(p.endswith("South Asia") for p in paths)


def test_borderless_table_rows_carry_their_column_labels(parsed):
    tables = [e for e in parsed.elements if e.kind == "table" and "TABLE 2.9" in e.text]
    assert tables, "the forecast table should be rebuilt with its caption"
    assert "India | 2024: 7.1 | 2025e: 7.7 | 2026f: 6.6 | 2027f: 7.2" in tables[0].text


def test_label_value_panel_becomes_a_table(parsed):
    panel = [e for e in parsed.elements if e.kind == "table" and "Production: 50%" in e.text]
    assert panel and "Product use: 45%" in panel[0].text


def test_footnote_marker_is_linked_to_its_endnote(parsed):
    _sections, chunks = chunk_report(parsed.elements, "Outlook", "visual")
    body = [c for c in chunks if "58 percent" in c.text]
    assert body and "[^18]" in body[0].text
    assert "ENERGY STAR Program Requirements" in body[0].note_text
    assert any(c.content_type == "footnote" for c in chunks)


def test_chunks_never_cross_sections(parsed):
    _sections, chunks = chunk_report(parsed.elements, "Outlook", "long_structured")
    for chunk in chunks:
        if "6.3 percent" in chunk.text:
            assert "Sub-Saharan" not in chunk.section_path
        if "4.1 percent" in chunk.text:
            assert chunk.section_path.endswith("Sub-Saharan Africa")


def test_th_ligature_repair_only_fixes_non_words():
    assert _repair("Te global economy", True) == "The global economy"
    assert _repair("Ten countries", True) == "Ten countries"
    assert _repair("Te global economy", False) == "Te global economy"


def test_shape_classification():
    assert classify_shape({"pages": 200}) == "long_structured"
    assert classify_shape({"pages": 12, "drawing_heavy_pages": 6}) == "visual"
    assert classify_shape({"pages": 80, "drawing_heavy_pages": 5}) == "narrative"


def test_end_to_end_routing_keeps_regions_apart(app_db, tmp_path):
    from app import report_store
    from app.report_generation import generate
    from app.report_retrieval import retrieve, route

    report_store.init()
    out = report_store.ingest_file(_report_pdf(tmp_path), use_vision=False)
    report_id = out["report_id"]

    plan = route("What is the growth outlook for South Asia?", [report_id])
    assert plan.kind == "title" and "South Asia" in plan.labels

    result = retrieve("What is the growth outlook for South Asia?", [report_id])
    assert result.hits
    # The South Asia section holds two chunks (prose + table); both must
    # outrank the near-identical Sub-Saharan Africa prose.
    assert all(h.section_path.endswith("South Asia") for h in result.hits[:2])

    summary = retrieve("According to the executive summary, what are the main risks?", [report_id])
    assert summary.route.kind == "explicit"
    assert summary.hits[0].section_path.startswith("Executive Summary")

    answer = generate(result)  # offline: extractive, still cited
    assert answer["citations"]
