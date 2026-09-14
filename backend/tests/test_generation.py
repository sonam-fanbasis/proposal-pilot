"""Grounding guarantees: what the system does when the model misbehaves."""

from app.generation import _split_gaps, generate, verify_citations
from app.retrieval import Hit, retrieve


def _hits(count: int = 3) -> list[Hit]:
    return [
        Hit(chunk_id=i, doc_id=1, text=f"evidence {i}", section_path="Commercials",
            page_from=1, page_to=1, title="Proposal", filename="p.md", client="",
            industry="", doc_type="", year=0)
        for i in range(1, count + 1)
    ]


def test_citations_pointing_at_nonexistent_evidence_are_removed():
    text, citations, _ungrounded, invalid = verify_citations(
        "We quoted 450,000 USD [1]. Delivery took nine months [7].", _hits(3)
    )
    assert invalid == [7]
    assert "[7]" not in text
    assert [c.index for c in citations] == [1]


def test_only_cited_evidence_is_returned_as_a_source():
    """The sources panel must show what was used, not everything retrieved."""
    _text, citations, _u, _i = verify_citations("Only the second source matters [2].", _hits(3))
    assert [c.index for c in citations] == [2]


def test_a_citation_after_the_full_stop_still_counts_as_grounded():
    """"claim. [2]" must not be scored as an uncited claim.

    Splitting on sentence boundaries first would separate every such citation
    from its claim and put a constant false offset on the grounding metric.
    """
    text, _c, ungrounded, _i = verify_citations(
        "We quoted 450,000 USD for the rebuild. [1]", _hits(3)
    )
    assert ungrounded == 0
    assert text == "We quoted 450,000 USD for the rebuild [1]."


def test_short_numeric_claims_are_not_exempt_from_grounding():
    _t, _c, ungrounded, _i = verify_citations("It cost 450,000 dollars.", _hits(3))
    assert ungrounded == 1


def test_connective_prose_is_not_counted_as_an_uncited_claim():
    _t, _c, ungrounded, _i = verify_citations(
        "This approach keeps risk low and the team engaged.", _hits(3)
    )
    assert ungrounded == 0


def test_gaps_section_is_split_out_of_the_draft():
    body, gaps = _split_gaps("The response body.\n---GAPS---\n- no SOC 2 evidence\n- none")
    assert body == "The response body."
    assert gaps == ["no SOC 2 evidence"]


def test_refuses_rather_than_answering_from_nothing(seeded):
    """An empty corpus match must produce a refusal, never invented prose."""
    result = retrieve("what is our experience building spacecraft avionics?")
    answer = generate("what is our experience building spacecraft avionics?", result)
    lowered = answer.text.lower()
    assert any(
        phrase in lowered
        for phrase in ("do not have", "nothing in the indexed", "could not find")
    ), answer.text


def test_answers_are_extractive_and_cited_when_offline(seeded):
    result = retrieve("what was the fixed bid price?")
    answer = generate("what was the fixed bid price?", result)
    assert answer.citations, "an offline answer must still carry citations"
    assert answer.invalid_citations == []
    # Every quoted line must trace back to a retrieved chunk.
    retrieved = {hit.text for hit in result.hits}
    for citation in answer.citations:
        assert any(citation.snippet[:60] in text for text in retrieved)


def test_grouped_citations_are_split_and_counted():
    """Models write "[1, 2]" regardless of what the prompt asks for.

    Regression guard: the `[n]` pattern matches neither "[1, 2]" nor "[1,2]",
    so before normalising, those sources dropped out of the sources panel and
    every sentence carrying one was scored as an uncited claim.
    """
    text, citations, ungrounded, invalid = verify_citations(
        "Every pull request passes static analysis [1, 2]. We target 80% coverage [1,3].", _hits(3)
    )
    assert "[1][2]" in text and "[1][3]" in text
    assert [c.index for c in citations] == [1, 2, 3]
    assert ungrounded == 0
    assert invalid == []
