"""Chunking is where most RAG quality is decided, so it is tested directly."""

from app.chunking import build_context_header, chunk_document
from app.parsing import _lines_to_blocks, _looks_like_heading

SAMPLE = """# Vendor Proposal

## Executive Summary
We rebuilt the payments hub. It now settles in real time instead of overnight batches.

## Commercials
### Payment Milestones
Twenty percent on signature. Thirty percent on user acceptance testing. Fifty percent on launch.
"""


def test_detects_common_heading_conventions():
    assert _looks_like_heading("3.2 Pricing")[0]
    assert _looks_like_heading("EXECUTIVE SUMMARY")[0]
    assert _looks_like_heading("## Scope of Work")[0]
    assert _looks_like_heading("Pricing")[0]


def test_prose_and_bullets_are_not_headings():
    assert not _looks_like_heading("- we delivered the thing")[0]
    assert not _looks_like_heading("The client asked us to rebuild their payment gateway.")[0]
    # A numbered list item that reads as a sentence is not a section heading.
    assert not _looks_like_heading(
        "1. We will deliver the following items across the engagement lifecycle."
    )[0]


def test_section_path_reflects_heading_hierarchy():
    chunks = chunk_document(_lines_to_blocks(SAMPLE, 1), {"title": "Vendor Proposal"})
    paths = [c.section_path for c in chunks]
    assert any("Executive Summary" in p for p in paths)
    assert any(p.endswith("Commercials > Payment Milestones") for p in paths)


def test_orphan_heading_merges_forward_not_backward():
    """"Commercials" has no body of its own; its content lives in the subsection.

    Merging it backward would file the payment milestones under the executive
    summary -- the exact mislabelling that makes a citation point at the wrong
    part of a document.
    """
    chunks = chunk_document(_lines_to_blocks(SAMPLE, 1), {})
    milestone_chunks = [c for c in chunks if "signature" in c.text]
    assert milestone_chunks, "milestone text should survive chunking"
    assert "Commercials" in milestone_chunks[0].section_path
    assert "Executive Summary" not in milestone_chunks[0].section_path


def test_context_header_is_embedded_but_not_part_of_the_body():
    chunks = chunk_document(
        _lines_to_blocks(SAMPLE, 1),
        {"title": "Vendor Proposal", "client": "Alpha Bank", "industry": "fintech",
         "tech_stack": ["node.js"]},
    )
    chunk = chunks[0]
    assert "Alpha Bank" in chunk.embedding_text
    assert "fintech" in chunk.embedding_text
    # The header must never leak into the text shown to the user or quoted as
    # evidence -- only into what gets embedded.
    assert "Alpha Bank" not in chunk.text
    assert chunk.embedding_text.endswith(chunk.text)


def test_context_header_handles_missing_metadata():
    assert build_context_header({}, "") == ""
    assert "Section: Pricing" in build_context_header({"title": "X"}, "Pricing")


def test_pdf_style_title_case_headings_are_detected():
    """PDF extraction discards styling, so "The Problem" arrives as body text."""
    assert _looks_like_heading("The Problem")[0]
    assert _looks_like_heading("What We Built")[0]


def test_table_rows_are_never_promoted_to_headings():
    """A pricing table must survive as one section, not fragment into rows.

    Regression guard: the title-case rule that recovers PDF headings also
    matched every row of a pricing table ("Months 1-3 USD 210,000"), turning
    each into its own section and scattering the commercial terms that most
    need to be retrieved together. Excluding lines containing digits is what
    keeps the table intact.
    """
    assert not _looks_like_heading("Months 1-3 USD 210,000")[0]
    assert not _looks_like_heading("Ingestion platform Months 3-9 USD 465,000")[0]
    assert not _looks_like_heading("Phase 1, Discovery and Foundations, weeks 1 to 6")[0]


def test_a_sentence_starting_with_a_section_word_is_not_a_heading():
    """"Case study. Client: X. Delivered 2024." merely starts with one."""
    assert not _looks_like_heading("Case Study. Client: Vanguard Motors AG. Delivered 2024.")[0]
    assert _looks_like_heading("Case Study")[0]
