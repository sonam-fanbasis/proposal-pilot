"""Retrieval: filter pushdown, fusion, relaxation and the refusal guardrail."""

from app.retrieval import _heuristic_plan, _rrf_fuse, resolve_filters, retrieve


def test_plan_splits_hard_constraints_from_the_semantic_query():
    plan = _heuristic_plan("fixed-bid healthcare projects under $200k")
    assert plan.filters.engagement_model == ["fixed_bid"]
    assert plan.filters.industry == ["healthcare"]
    assert plan.filters.value_max == 200_000
    assert plan.intent == "price"


def test_plan_reads_year_and_technology_constraints():
    plan = _heuristic_plan("have we done fintech with React Native since 2022")
    assert plan.filters.year_min == 2022
    assert plan.filters.tech == ["react native"]
    assert plan.filters.industry == ["fintech"]


def test_rrf_rewards_agreement_between_retrievers():
    """A chunk both retrievers rank must beat one only a single retriever found."""
    fused = _rrf_fuse([(1, 0.9), (2, 0.8)], [(2, 5.0), (3, 4.0)], rrf_k=60)
    assert fused[2]["score"] > fused[1]["score"]
    assert fused[2]["vector_rank"] == 2 and fused[2]["keyword_rank"] == 1
    assert fused[3]["keyword_rank"] == 2 and fused[3]["vector_rank"] is None


def test_filters_resolve_to_documents(seeded):
    from app.retrieval import Filters

    assert resolve_filters(Filters()) is None  # no filter means search everything

    fixed = resolve_filters(Filters(engagement_model=["fixed_bid"]))
    dedicated = resolve_filters(Filters(engagement_model=["dedicated_team"]))
    assert len(fixed) == 1 and len(dedicated) == 1
    assert fixed != dedicated


def test_technology_filters_are_conjunctive(seeded):
    """"Node AND Django" must match nothing when no document uses both."""
    from app.retrieval import Filters

    assert len(resolve_filters(Filters(tech=["node.js"]))) == 1
    assert resolve_filters(Filters(tech=["node.js", "django"])) == []


def test_documents_without_a_recorded_value_survive_a_price_filter(seeded):
    """Unknown is not the same as too expensive."""
    from app import db
    from app.retrieval import Filters

    db.execute("UPDATE documents SET value_usd = 0 WHERE title LIKE '%Alpha%'")
    matched = resolve_filters(Filters(value_max=100_000))
    titles = [
        db.query_one("SELECT title FROM documents WHERE id = ?", (i,))["title"] for i in matched
    ]
    assert any("Alpha" in t for t in titles)


def test_impossible_filter_combination_is_relaxed_rather_than_returning_nothing(seeded):
    result = retrieve("fixed bid Django work in logistics under $10k")
    assert result.relaxed is True
    assert result.hits, "relaxation should recover near-misses instead of answering nothing"


def test_filter_pushdown_narrows_the_searched_documents(seeded):
    result = retrieve("what did the healthcare engagement cost?")
    assert result.filtered_doc_count == 1
    assert all("Beta" in hit.title for hit in result.hits)


def test_retrieval_returns_provenance_for_every_hit(seeded):
    result = retrieve("what was the fixed bid price?")
    assert result.hits
    for hit in result.hits:
        assert hit.title and hit.filename
        assert hit.vector_rank is not None or hit.keyword_rank is not None


# --- confidence: the guardrail that decides whether to answer at all --------

def test_confidence_collapses_for_a_subject_the_corpus_never_mentions(seeded):
    """A banking/health corpus asked about spacecraft must score near zero.

    Regression guard. Corroboration was originally *added* to the score, so
    three unrelated documents agreeing scored 0.25 on their own, cleared the
    threshold, and produced a confident answer assembled from irrelevant
    evidence. Corroboration now multiplies relevance instead.
    """
    result = retrieve("what is our experience building spacecraft avionics?")
    assert result.confidence < 0.18


def test_a_synonym_costs_far_less_than_an_absent_subject(seeded):
    """Good retrieval must not be punished for the user's choice of words.

    "Results" is the section heading, so "outcomes" is a synonym the corpus
    never spells -- but the right section is still retrieved, and the score
    must reflect that. An absent *subject* is a different thing entirely.

    Asserted as an ordering rather than a threshold: IDF depends on corpus
    size, so an absolute cutoff calibrated on this two-document fixture would
    mean nothing about the real corpus.
    """
    synonym = retrieve("what were the measured outcomes of the telemedicine work?")
    absent = retrieve("what were the measured outcomes of the spacecraft avionics work?")

    assert synonym.hits
    assert "Results" in synonym.hits[0].section_path
    assert synonym.confidence > absent.confidence * 2


def test_a_plainly_answerable_question_stays_above_the_threshold(seeded):
    assert retrieve("what was the fixed bid price?").confidence >= 0.18
    assert retrieve("which engagement used a dedicated team?").confidence >= 0.18


def test_hyphenation_does_not_change_the_answer(seeded):
    """"fixed-bid" and "fixed bid" must score the same.

    Regression guard. Writers spell compounds inconsistently, and a literal
    substring test scored the correctly-retrieved Commercials section as not
    containing the term the question was about -- refusing a question whose
    filters and retrieval were both right.
    """
    hyphenated = retrieve("what fixed-bid work have we done?")
    spaced = retrieve("what fixed bid work have we done?")

    assert hyphenated.hits and spaced.hits
    assert abs(hyphenated.confidence - spaced.confidence) < 0.08
    assert hyphenated.confidence >= 0.18
