"""Metadata drives the retrieval filters, so wrong metadata is a silent
correctness bug: the document is simply never retrieved for the query it
should answer."""

from app.metadata import (
    _detect_engagement_model, _money_to_usd, extract, find_tech_in_text, normalise_tech,
)


def test_negated_engagement_model_is_not_selected():
    text = (
        "## Engagement Model\n"
        "This was a dedicated team engagement rather than a fixed bid. "
        "A fixed bid would have forced premature specification. "
        "The dedicated team model let us run discovery in parallel."
    )
    assert _detect_engagement_model(text) == "dedicated_team"


def test_declared_model_wins_over_incidental_mentions():
    text = "Northwind engaged us on a time and materials basis. A fixed bid was discussed."
    assert _detect_engagement_model(text) == "time_and_materials"


def test_document_describing_all_models_commits_to_none():
    text = (
        "Fixed bid suits well-specified work. Time and materials suits exploratory work. "
        "Dedicated team suits long-running product development."
    )
    assert _detect_engagement_model(text) == "other"


def test_total_value_wins_over_a_larger_unrelated_figure():
    """The largest number in a case study is often a saving, not the price."""
    text = (
        "The engagement totalled USD 320,000 at a blended rate. "
        "Customer service costs fell by EUR 340,000 per year."
    )
    assert _money_to_usd(text) == 320_000


def test_money_ignores_page_numbers_and_small_line_items():
    assert _money_to_usd("Change requests are charged at USD 85 per hour.") == 0


def test_money_handles_multipliers():
    assert _money_to_usd("a fixed bid of $1.2M") == 1_200_000


def test_longest_tech_match_wins():
    """React Native must not also register plain React.

    Technology filters are ANDed, so a spurious tag makes a query for React
    Native demand a technology the document never claimed.
    """
    assert find_tech_in_text("Built with React Native and Node.js on AWS") == [
        "react native", "node.js", "aws",
    ]


def test_react_is_still_detected_on_its_own():
    found = find_tech_in_text("A React SPA talking to a Node backend")
    assert "react" in found and "node.js" in found


def test_aliases_normalise_to_one_tag():
    assert normalise_tech(["NodeJS", "Node.js", "k8s", "Golang"]) == [
        "node.js", "kubernetes", "go",
    ]


def test_capability_document_reports_no_engagement_value():
    """A firm-level document's quoted averages are not engagement facts."""
    text = (
        "# Capability Statement\n"
        "Our average engagement length is 11 months. We work in healthcare, logistics, "
        "retail and insurance. Fixed bid suits well-specified work."
    )
    meta = extract(text, "capability_statement.md")
    assert meta["doc_type"] == "capability"
    assert meta["value_usd"] == 0
    assert meta["duration_months"] == 0
    # No single industry dominates, so none is claimed.
    assert meta["industry"] == "other"


def test_extract_reads_a_normal_proposal():
    text = (
        "Zenith Bank Proposal\n"
        "We rebuilt the mobile banking app using React Native and Node.js on AWS.\n"
        "This is a fixed bid of USD 450,000 delivered over 9 months in 2023."
    )
    meta = extract(text, "zenith_proposal.md")
    assert meta["value_usd"] == 450_000
    assert meta["duration_months"] == 9
    assert meta["year"] == 2023
    assert meta["engagement_model"] == "fixed_bid"
    assert "react native" in meta["tech_stack"]
