"""Structured metadata extraction at ingest time.

A proposal library is not a flat pile of prose. Every document has facts
attached to it -- who the client was, which industry, what it cost, what it
was built with -- and the most valuable questions a sales or delivery team
asks are *filters over those facts* ("what have we done in healthcare with
AWS, under $200k?"), not similarity searches.

Extracting that structure once at ingest, into real indexed columns, is what
lets `retrieval.py` push filters down into the ANN query. Doing it at query
time instead would mean re-reading every document on every question.

Both paths produce the same shape:
  * with a key, Gemini structured output over the document's opening pages;
  * without one, a vocabulary + regex pass that is weaker on `client` and
    `summary` but solid on tech stack, year, money and engagement model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import llm

INDUSTRIES = [
    "fintech", "banking", "insurance", "healthcare", "pharma", "ecommerce",
    "retail", "logistics", "travel", "hospitality", "edtech", "real_estate",
    "media", "gaming", "manufacturing", "automotive", "energy", "telecom",
    "saas", "government", "nonprofit", "other",
]
DOC_TYPES = ["proposal", "case_study", "sow", "capability", "rfp", "other"]
ENGAGEMENT_MODELS = ["fixed_bid", "time_and_materials", "dedicated_team", "retainer", "other"]

# Canonical tech tags. The alias map exists because "Node", "NodeJS" and
# "Node.js" must become one filterable token -- otherwise a query for one of
# them silently misses two thirds of the corpus.
TECH_ALIASES: dict[str, list[str]] = {
    "node.js": ["node js", "nodejs", "node"],
    "react": ["react.js", "reactjs"],
    "react native": ["react-native"],
    "next.js": ["nextjs", "next js"],
    "angular": ["angularjs"],
    "vue.js": ["vuejs", "vue"],
    "python": [],
    "django": [],
    "fastapi": ["fast api"],
    "flask": [],
    "java": [],
    "spring boot": ["springboot", "spring"],
    "kotlin": [],
    "swift": ["swiftui"],
    "flutter": [],
    "ruby on rails": ["rails", "ror"],
    "php": [],
    "laravel": [],
    ".net": ["dotnet", "asp.net", "c#"],
    "go": ["golang"],
    "rust": [],
    "postgresql": ["postgres", "pgsql"],
    "mysql": [],
    "mongodb": ["mongo"],
    "redis": [],
    "elasticsearch": ["elastic search"],
    "kafka": ["apache kafka"],
    "rabbitmq": [],
    "aws": ["amazon web services"],
    "azure": ["microsoft azure"],
    "gcp": ["google cloud"],
    "kubernetes": ["k8s"],
    "docker": [],
    "terraform": [],
    "graphql": [],
    "rest api": ["restful"],
    "stripe": [],
    "twilio": [],
    "openai": ["gpt-4", "gpt4"],
    "langchain": [],
    "tensorflow": [],
    "pytorch": [],
    "snowflake": [],
    "tableau": [],
    "power bi": ["powerbi"],
}

_MONEY = re.compile(
    r"(?:USD|US\$|\$|EUR|€|INR|₹)\s?([\d][\d,]*(?:\.\d+)?)\s*(k|m|thousand|million|lakh|cr|crore)?",
    re.I,
)
_DURATION = re.compile(r"(\d{1,3})\s*(month|months|week|weeks|sprint|sprints)\b", re.I)
_YEAR = re.compile(r"\b(20[0-2]\d)\b")

_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "client": {"type": "string"},
        "industry": {"type": "string", "enum": INDUSTRIES},
        "doc_type": {"type": "string", "enum": DOC_TYPES},
        "engagement_model": {"type": "string", "enum": ENGAGEMENT_MODELS},
        "year": {"type": "integer"},
        "value_usd": {"type": "integer"},
        "duration_months": {"type": "integer"},
        "tech_stack": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["title", "industry", "doc_type", "tech_stack", "summary"],
}

_SYSTEM = """You catalogue a consulting firm's proposal and case-study library.
Extract only facts that are explicitly stated in the document.
Rules:
- `client` is the customer the work was done FOR, never the vendor writing the document. Use "" if unclear.
- `value_usd` is the total contract value converted to whole US dollars. Use 0 if not stated.
- `duration_months` is the engagement length in months. Use 0 if not stated.
- `tech_stack` lists concrete technologies (languages, frameworks, databases, clouds). Never marketing words like "scalable" or "cloud-native".
- `summary` is two sentences: what the work was, and what outcome it produced.
Do not guess. An empty or zero value is always better than an invented one."""


def normalise_tech(raw_terms: list[str]) -> list[str]:
    """Map free-text technology names onto canonical tags."""
    lookup: dict[str, str] = {}
    for canonical, aliases in TECH_ALIASES.items():
        lookup[canonical] = canonical
        for alias in aliases:
            lookup[alias] = canonical

    found: list[str] = []
    for term in raw_terms:
        key = re.sub(r"\s+", " ", str(term).strip().lower())
        canonical = lookup.get(key)
        if canonical is None:
            # Keep unknown-but-plausible tech names rather than dropping them;
            # the corpus is the source of truth for its own vocabulary.
            if 1 <= len(key) <= 30 and not key.isdigit():
                canonical = key
        if canonical and canonical not in found:
            found.append(canonical)
    return found


def find_tech_in_text(text: str) -> list[str]:
    """Scan free text for known technologies, longest name first.

    Order matters. "React Native" contains "React", so a naive scan tags a
    React Native project with both. That is not a cosmetic issue: the
    retriever ANDs technology filters together, so the spurious `react` tag
    makes a query for React Native demand a technology the document never
    claimed. Matching longest-first and blanking out the span each time means
    every stretch of text is attributed to exactly one technology.
    """
    lowered = f" {re.sub(r'[^a-z0-9+#. ]', ' ', text.lower())} "

    candidates: list[tuple[str, str]] = []
    for canonical, aliases in TECH_ALIASES.items():
        for name in [canonical, *aliases]:
            candidates.append((name, canonical))
    candidates.sort(key=lambda pair: len(pair[0]), reverse=True)

    found: list[str] = []
    for name, canonical in candidates:
        pattern = r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            if canonical not in found:
                found.append(canonical)
            lowered = re.sub(pattern, lambda m: " " * len(m.group(0)), lowered)
    return found


# Phrases that mark an amount as the engagement's total value rather than a
# line item, an hourly rate, or a saving the client made.
_VALUE_CUES = (
    "total contract value", "total engagement value", "engagement value",
    "contract value", "total value", "fixed bid of", "fixed price of",
    "totalled", "totaled", "totalling", "totaling", "total of",
    "engagement is", "value is", "value was", "budget of",
)


def _money_to_usd(text: str) -> int:
    """Find the engagement's total value, not simply the largest figure present.

    Taking the maximum is the obvious implementation and it is wrong. A case
    study that reports "saved the client EUR 340,000 per year" alongside a
    "USD 320,000" engagement value would be catalogued at the saving, and a
    price filter would then quietly exclude it. So amounts preceded by an
    explicit total-value phrase win; the plain maximum is only a fallback when
    the document never states a total.
    """
    multipliers = {
        "k": 1_000, "thousand": 1_000, "m": 1_000_000, "million": 1_000_000,
        "lakh": 100_000, "cr": 10_000_000, "crore": 10_000_000,
    }
    lowered = text.lower()
    anchored = 0
    unanchored = 0

    for match in _MONEY.finditer(text):
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        value *= multipliers.get((match.group(2) or "").lower(), 1)
        # Screen out page numbers, hourly rates and small line items.
        if not (1_000 <= value < 1_000_000_000):
            continue
        window = lowered[max(0, match.start() - 70) : match.start()]
        if any(cue in window for cue in _VALUE_CUES):
            anchored = max(anchored, int(value))
        else:
            unanchored = max(unanchored, int(value))

    return anchored or unanchored


_ENGAGEMENT_WORDS: dict[str, list[str]] = {
    "fixed_bid": ["fixed bid", "fixed-bid", "fixed price", "fixed-price", "lump sum"],
    "time_and_materials": ["time and materials", "time & materials", "t&m"],
    "dedicated_team": ["dedicated team", "dedicated developers", "staff augmentation", "extended team"],
    "retainer": ["retainer", "monthly support fee"],
}
# Preceding phrases that mean "this engagement IS that model".
_DECLARATION_BEFORE = (
    "is a", "was a", "this is a", "delivered as a", "engaged us on a",
    "provided on a", "provided on", "services are provided on a", "priced as a",
    "contracted on a", "runs as a", "billed as a",
)
# Following phrases with the same force ("Fixed bid of USD 180,000").
_DECLARATION_AFTER = ("of usd", "of us$", "of $", "of eur", "of gbp", "of inr", "basis", "contract")
# Phrases that mean the model is being ruled OUT, not chosen.
_NEGATION_BEFORE = (
    "rather than", "instead of", "not a", "no longer", "unwilling",
    "willing to price", "would have", "will not", "ruled out",
)


def _detect_engagement_model(text: str) -> str:
    """Score each commercial model by how it is *used* in the sentence.

    A plain substring search gets this backwards on the most informative
    sentences in the corpus. "This was a dedicated team engagement rather than
    a fixed bid" contains the words "fixed bid", and a first-match-wins scan
    therefore files a dedicated-team engagement as fixed bid -- corrupting
    exactly the filter a user would rely on. Scoring occurrences by their
    surrounding context, and subtracting for negation, reads those sentences
    the way a person does.
    """
    lowered = text.lower()
    scores: dict[str, int] = {}

    for model, words in _ENGAGEMENT_WORDS.items():
        total = 0
        for word in words:
            for match in re.finditer(re.escape(word), lowered):
                before = lowered[max(0, match.start() - 60) : match.start()]
                # Clip to the current sentence. Without this, a negation from
                # the *previous* sentence leaks forward: "...a fixed bid would
                # have forced premature specification. The dedicated team
                # model let us..." would read the second clause as negated.
                before = re.split(r"[.!?\n]", before)[-1]
                after = lowered[match.end() : match.end() + 40]
                if any(cue in before for cue in _NEGATION_BEFORE):
                    total -= 2
                elif any(before.rstrip().endswith(cue) for cue in _DECLARATION_BEFORE) or any(
                    after.lstrip().startswith(cue) for cue in _DECLARATION_AFTER
                ):
                    total += 3
                else:
                    total += 1
        scores[model] = total

    best = max(scores, key=scores.get) if scores else "other"
    # A capability deck describes all three models without being any of them,
    # so a weak winner is reported as "other" rather than guessed.
    return best if scores.get(best, 0) >= 3 else "other"


def _heuristic(text: str, filename: str) -> dict:
    lowered = text.lower()

    industry = "other"
    industry_words = {
        "fintech": ["fintech", "payment", "banking app", "wallet", "kyc", "trading"],
        "banking": ["bank", "core banking", "loan origination"],
        "insurance": ["insurance", "underwriting", "claims processing", "policyholder"],
        "healthcare": ["healthcare", "hipaa", "patient", "ehr", "telemedicine", "clinical"],
        "ecommerce": ["ecommerce", "e-commerce", "checkout", "storefront", "cart"],
        "logistics": ["logistics", "fleet", "shipment", "warehouse", "supply chain"],
        "edtech": ["edtech", "lms", "learner", "course", "student"],
        "travel": ["travel", "booking engine", "itinerary", "hotel"],
        "real_estate": ["real estate", "property listing", "mls"],
        "saas": ["saas", "multi-tenant", "subscription platform"],
        "manufacturing": ["manufacturing", "mes", "shop floor", "plant"],
        "media": ["streaming", "media", "publishing", "cms"],
        "gaming": ["gaming", "game engine", "unity"],
        "energy": ["energy", "solar", "grid", "utility"],
        "telecom": ["telecom", "5g", "network operator"],
        "government": ["government", "public sector", "municipal"],
    }
    scores = {
        name: sum(lowered.count(word) for word in words)
        for name, words in industry_words.items()
    }
    if scores and max(scores.values()) > 0:
        ranked = sorted(scores.values(), reverse=True)
        runner_up = ranked[1] if len(ranked) > 1 else 0
        # A document that mentions four industries in passing (a capability
        # deck listing sector experience) has no industry of its own, so a
        # winner must clearly beat the runner-up rather than merely lead it.
        # Expressed against the runner-up rather than as an absolute floor:
        # a short document naming one industry once, with nothing competing,
        # is unambiguous and should be catalogued, while a long document
        # naming three industries repeatedly is not.
        if ranked[0] > 2 * runner_up:
            industry = max(scores, key=scores.get)

    doc_type = "other"
    name = filename.lower()
    if "case" in name or "case study" in lowered[:2000]:
        doc_type = "case_study"
    elif "sow" in name or "statement of work" in lowered[:2000]:
        doc_type = "sow"
    elif "rfp" in name or "request for proposal" in lowered[:2000]:
        doc_type = "rfp"
    elif "capabilit" in name or "capabilit" in lowered[:1500]:
        doc_type = "capability"
    elif "proposal" in name or "proposal" in lowered[:2000]:
        doc_type = "proposal"

    engagement = _detect_engagement_model(text)

    years = [int(y) for y in _YEAR.findall(text)]
    duration = 0
    match = _DURATION.search(text)
    if match:
        count, unit = int(match.group(1)), match.group(2).lower()
        duration = count if unit.startswith("month") else max(1, round(count / 4.3))
        if unit.startswith("sprint"):
            duration = max(1, round(count * 2 / 4.3))

    # First non-empty line is the title far more often than not.
    title = next(
        (line.strip() for line in text.splitlines() if 3 < len(line.strip()) < 120),
        Path(filename).stem.replace("_", " ").replace("-", " ").title(),
    )
    summary = " ".join(text.split())[:280]

    # A capability statement or RFP describes the firm, not one engagement, so
    # its "value" and "duration" are averages quoted in prose. Recording them
    # as engagement facts would make price and duration filters match a
    # document that has no price or duration.
    if doc_type in {"capability", "rfp"}:
        value_usd = 0
        duration = 0
    else:
        value_usd = _money_to_usd(text)

    return {
        "title": title,
        "client": "",
        "industry": industry,
        "doc_type": doc_type,
        "engagement_model": engagement,
        "year": max(years) if years else 0,
        "value_usd": value_usd,
        "duration_months": duration,
        "tech_stack": find_tech_in_text(text),
        "summary": summary,
        "extracted_by": "heuristic",
    }


def extract(text: str, filename: str) -> dict:
    """Best-effort metadata for one document. Never raises."""
    fallback = _heuristic(text, filename)
    if llm.is_offline():
        return fallback

    # The opening pages carry the identifying facts; sending the whole
    # document would cost far more tokens for no extra signal.
    excerpt = text[:12000]
    try:
        data = llm.complete_json(
            f"Document filename: {filename}\n\n---\n{excerpt}\n---\n\n"
            "Extract the catalogue metadata for this document.",
            system=_SYSTEM,
            json_schema=_METADATA_SCHEMA,
        )
    except Exception as exc:
        fallback["extraction_error"] = str(exc)[:200]
        return fallback

    merged = dict(fallback)
    for key in (
        "title", "client", "industry", "doc_type", "engagement_model",
        "year", "value_usd", "duration_months", "summary",
    ):
        value = data.get(key)
        if value not in (None, "", 0, []):
            merged[key] = value
    # Union the two tech lists: the model catches prose mentions, the
    # vocabulary scan catches names buried in tables the excerpt may have cut.
    merged["tech_stack"] = normalise_tech(
        list(data.get("tech_stack") or []) + fallback["tech_stack"]
    )
    if merged.get("industry") not in INDUSTRIES:
        merged["industry"] = fallback["industry"]
    if merged.get("doc_type") not in DOC_TYPES:
        merged["doc_type"] = fallback["doc_type"]
    if merged.get("engagement_model") not in ENGAGEMENT_MODELS:
        merged["engagement_model"] = fallback["engagement_model"]
    merged["extracted_by"] = "llm"
    return merged
