import pytest

from app.services.verifier import (
    EvidenceAssessment,
    VerificationSignal,
    aggregate_evidence_status,
    aggregate_status,
    is_directly_relevant,
    verify_claim,
)


class StubNLI:
    def __init__(self, labels: dict[str, float]) -> None:
        self.labels = labels
        self.arguments: tuple[str, str] | None = None

    def __call__(self, premise: str, hypothesis: str) -> dict[str, dict[str, float]]:
        self.arguments = (premise, hypothesis)
        return {"labels": self.labels}


def test_uses_evidence_as_premise_and_claim_as_hypothesis() -> None:
    model = StubNLI({"entailment": 0.91, "neutral": 0.06, "contradiction": 0.03})

    signal = verify_claim("The claim", "The evidence", model)

    assert signal.label == "entailment"
    assert signal.score == 0.91
    assert model.arguments == ("The evidence", "The claim")


def test_maps_model_label_indices_explicitly() -> None:
    model = StubNLI({"LABEL_0": 0.10, "LABEL_1": 0.20, "LABEL_2": 0.70})

    signal = verify_claim("The claim", "The evidence", model)

    assert signal.label == "entailment"


@pytest.mark.parametrize(
    ("signals", "expected"),
    [
        ([], "insufficient_evidence"),
        ([VerificationSignal("neutral", 0.95, "p", "h")], "insufficient_evidence"),
        ([VerificationSignal("entailment", 0.90, "p", "h")], "supported"),
        ([VerificationSignal("contradiction", 0.90, "p", "h")], "refuted"),
        (
            [
                VerificationSignal("entailment", 0.90, "p", "h"),
                VerificationSignal("contradiction", 0.85, "p", "h"),
            ],
            "conflicting_evidence",
        ),
    ],
)
def test_aggregate_status_is_cautious(signals, expected) -> None:
    assert aggregate_status(signals) == expected


def test_empty_claim_or_evidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="claim"):
        verify_claim("", "evidence", StubNLI({}))
    with pytest.raises(ValueError, match="evidence"):
        verify_claim("claim", " ", StubNLI({}))


def test_evidence_aggregation_requires_relevance_and_source_quality() -> None:
    supported = VerificationSignal("entailment", 0.95, "p", "h")

    assert aggregate_evidence_status(
        [EvidenceAssessment(supported, 0.2, "official")]
    ) == "insufficient_evidence"
    assert aggregate_evidence_status(
        [EvidenceAssessment(supported, 0.6, "official")]
    ) == "supported"
    assert aggregate_evidence_status(
        [EvidenceAssessment(supported, 0.6, "context_only")]
    ) == "insufficient_evidence"


def test_keyword_overlap_is_not_direct_evidence_for_life_status_claim() -> None:
    assert not is_directly_relevant(
        "Trump is dead",
        "Trump ally Lindsey Graham dies at 71.",
    )
    assert not is_directly_relevant(
        "Trump is dead",
        "Trump says the Anti-Weaponization Fund is dead.",
    )
    assert is_directly_relevant(
        "Trump is dead",
        "Officials confirmed that Trump died on Tuesday.",
    )
    assert is_directly_relevant(
        "Trump is dead",
        "The death of President Trump was confirmed by officials.",
    )
    assert is_directly_relevant(
        "Trump is dead",
        "Officials confirmed Trump's death.",
    )
    assert is_directly_relevant(
        "Trump is dead",
        "Fact checkers say Donald Trump did not die.",
    )
    assert not is_directly_relevant(
        "Trump is dead",
        "Trump ally Lindsey Graham died after a brief illness.",
    )


def test_unrelated_passages_are_not_relevant_even_with_shared_words() -> None:
    claim = "The Python 3.13 release notes are available"
    assert not is_directly_relevant(claim, "HCL Notes. features are not supported in this client")
    assert not is_directly_relevant(claim, "2026.38.5.11 Official Tesla Release Notes - Software Updates")
    assert is_directly_relevant(
        claim, "What's New In Python 3.13. Release notes for Python 3.13 are available."
    )


def _assessment(label: str, score: float, premise: str, similarity: float = 0.9):
    return EvidenceAssessment(
        VerificationSignal(label, score, premise, "claim"), similarity, "reputable"
    )


def test_entailment_from_a_debunking_passage_is_not_support() -> None:
    premise = "Moon landing conspiracy theories claim the 1969 landing was faked."
    assert aggregate_evidence_status([_assessment("entailment", 0.95, premise)]) == "insufficient_evidence"


def test_entailment_from_a_plain_passage_is_support() -> None:
    premise = "Apollo 11 landed on the Moon in July 1969."
    assert aggregate_evidence_status([_assessment("entailment", 0.95, premise)]) == "supported"


def test_confident_contradiction_on_a_close_passage_refutes() -> None:
    premise = "Canberra is the capital of Australia."
    assert aggregate_evidence_status([_assessment("contradiction", 0.92, premise, 0.67)]) == "refuted"


def test_weak_contradiction_on_a_loose_passage_does_not_refute() -> None:
    premise = "Canberra is the capital of Australia."
    assert aggregate_evidence_status([_assessment("contradiction", 0.7, premise, 0.4)]) == "insufficient_evidence"


def test_fiction_sources_are_recognised_by_title_suffix() -> None:
    from app.services.verifier import is_fiction_source

    for title in (
        "Fly Me to the Moon (2024 film)",
        "Apollo 13 (film)",
        "The Martian (novel)",
        "Moon (video game)",
        "Hannibal (TV series)",
        "Dark Side of the Moon (album)",
        "Sherlock Holmes (character)",
    ):
        assert is_fiction_source(title), title
    for title in (
        "Apollo 11",
        "Moon landing conspiracy theories",
        "Paris (disambiguation)",
        "Earth",
        "Python (programming language)",
        "Film (magazine)",
    ):
        assert not is_fiction_source(title), title


def test_life_status_relevance_understands_full_names() -> None:
    claim = "Dua Lipa dead"
    assert is_directly_relevant(claim, "Dua Lipa dead 2026: Singer killed by internet death hoax")
    assert is_directly_relevant(claim, "Dua Lipa's death was a hoax, fact checkers say.")
    assert is_directly_relevant(claim, "Dua Lipa did not die, her team confirmed.")
    assert not is_directly_relevant(claim, "Songwriter for Britney Spears and Dua Lipa dead after stabbing")
    assert not is_directly_relevant(claim, "Dua Lipa announces new album; fan favourite song is dead simple")


CLAIM = "NASA confirms Earth will experience 15 days of darkness"


def _passage_assessment(label: str, score: float, premise: str, similarity: float = 0.6, quality: str = "unknown"):
    return EvidenceAssessment(VerificationSignal(label, score, premise, CLAIM), similarity, quality)


def test_a_passage_that_quotes_the_hoax_is_not_support() -> None:
    quoted = (
        'The article continues with: " [T]his time, NASA confirms what\'s been circling the web recently - '
        'our planet Earth will experience total darkness for 15 days in November 2017'
    )
    assert aggregate_evidence_status([_passage_assessment("entailment", 0.99, quoted, 0.89)]) == "insufficient_evidence"


def test_text_inside_quotation_marks_is_not_assertion() -> None:
    premise = 'Posts shared: "NASA confirms Earth will experience 15 days of darkness this November." Check before sharing.'
    assert aggregate_evidence_status([_passage_assessment("entailment", 0.97, premise, 0.9)]) == "insufficient_evidence"


def test_a_confident_debunk_is_not_lost_to_low_lexical_similarity() -> None:
    debunk = "No, NASA Is Not Predicting 15 Days Of Complete Darkness In November"
    assert aggregate_evidence_status([_passage_assessment("contradiction", 1.0, debunk, 0.56, quality="reputable")]) == "refuted"


def test_quoted_hoax_plus_debunks_resolve_to_refuted_or_conflicting_never_supported() -> None:
    quoted = 'The article continues with: " [T]this time, NASA confirms Earth will experience total darkness for 15 days'
    debunk = "No, NASA Is Not Predicting 15 Days Of Complete Darkness In November"
    second_debunk = "The fake news claimed that NASA had confirmed a surprise event that will plunge the planet into darkness for 15 days."
    status = aggregate_evidence_status(
        [
            _passage_assessment("entailment", 0.99, quoted, 0.89),
            EvidenceAssessment(VerificationSignal("contradiction", 1.0, debunk, CLAIM), 0.56, "unknown", "forbes.com"),
            EvidenceAssessment(VerificationSignal("contradiction", 0.9, second_debunk, CLAIM), 0.78, "unknown", "indiatimes.com"),
        ]
    )
    assert status == "refuted"  # two independent debunks outweigh the quoted hoax, which cannot support


def test_fake_news_reporting_counts_as_a_debunk() -> None:
    premise = "The fake news claimed that NASA had confirmed a surprise event that will plunge the planet into darkness for 15 days."
    assert aggregate_evidence_status([_passage_assessment("contradiction", 0.85, premise, 0.78, quality="reputable")]) == "refuted"


def test_several_authoritative_debunks_outweigh_one_stray_supporting_passage() -> None:
    claim = "Bats are blind"
    debunks = [
        EvidenceAssessment(VerificationSignal("contradiction", 1.0, "No, bats are not blind.", claim), 0.9, "official"),
        EvidenceAssessment(VerificationSignal("contradiction", 0.93, "A myth: bats are blind.", claim), 0.9, "reputable"),
        EvidenceAssessment(VerificationSignal("contradiction", 0.98, "Bats are generally not blind.", claim), 0.9, "unknown"),
    ]
    stray = EvidenceAssessment(VerificationSignal("entailment", 0.9, "Some bats rely little on sight.", claim), 0.9, "unknown")
    assert aggregate_evidence_status(debunks + [stray]) == "refuted"


def test_balanced_support_and_refutation_stays_conflicting() -> None:
    claim = "A contested claim"
    support = EvidenceAssessment(VerificationSignal("entailment", 0.9, "It is so.", claim), 0.9, "official")
    refute = EvidenceAssessment(VerificationSignal("contradiction", 0.9, "It is not so.", claim), 0.9, "official")
    assert aggregate_evidence_status([support, refute]) == "conflicting_evidence"


def test_context_only_sources_can_never_decide_a_claim() -> None:
    for label in ("entailment", "contradiction"):
        signal = VerificationSignal(label, 1.0, "It is not so.", "claim")
        assert aggregate_evidence_status([EvidenceAssessment(signal, 1.0, "context_only")]) == "insufficient_evidence"


def _from(source: str, label: str, quality: str = "unknown", score: float = 0.95):
    signal = VerificationSignal(label, score, "It is not so, according to this page.", "claim")
    return EvidenceAssessment(signal, 0.8, quality, source)


def test_a_single_unknown_source_can_neither_refute_nor_support() -> None:
    assert aggregate_evidence_status([_from("ludwig.guru", "contradiction", score=0.99)]) == "insufficient_evidence"
    assert aggregate_evidence_status([_from("blog.example", "entailment")]) == "insufficient_evidence"


def test_the_same_site_twice_is_still_one_source() -> None:
    twice = [_from("blog.example", "contradiction"), _from("blog.example", "contradiction")]
    assert aggregate_evidence_status(twice) == "insufficient_evidence"


def test_two_independent_sources_corroborate() -> None:
    assert aggregate_evidence_status([_from("a.example", "contradiction"), _from("b.example", "contradiction")]) == "refuted"
    assert aggregate_evidence_status([_from("a.example", "entailment"), _from("b.example", "entailment")]) == "supported"


def test_one_official_or_reputable_source_is_enough() -> None:
    assert aggregate_evidence_status([_from("usgs.gov", "contradiction", quality="official")]) == "refuted"
    assert aggregate_evidence_status([_from("bbc.com", "entailment", quality="reputable")]) == "supported"


def test_an_uncorroborated_winner_of_a_conflict_stays_conflicting() -> None:
    mixed = [_from("a.example", "contradiction", score=0.99), _from("b.example", "entailment", score=0.3)]
    assert aggregate_evidence_status(mixed) in {"conflicting_evidence", "insufficient_evidence"}


def test_weak_entailment_is_not_support_but_strong_entailment_is() -> None:
    def support(score):
        signal = VerificationSignal("entailment", score, "State capital is Sydney, Australia's largest city.", "claim")
        return EvidenceAssessment(signal, 0.9, "official", "en.wikipedia.org")

    assert aggregate_evidence_status([support(0.69)]) == "insufficient_evidence"
    assert aggregate_evidence_status([support(0.95)]) == "supported"


def test_refutation_keeps_the_lower_bar_than_support() -> None:
    signal = VerificationSignal("contradiction", 0.69, "It is not so.", "claim")
    assert aggregate_evidence_status([EvidenceAssessment(signal, 0.9, "official", "usgs.gov")]) == "refuted"


def test_a_fact_checker_restating_a_claim_is_not_support_but_its_debunk_is_refutation() -> None:
    claim = "NASA confirms Earth will experience 15 days of darkness"
    restated = "NASA has confirmed that the Earth will experience 15 days of total darkness between November 15 and 29."
    debunk = "No, NASA is not predicting 15 days of darkness."

    def assess(source, label, premise, quality="reputable"):
        return EvidenceAssessment(VerificationSignal(label, 0.99, premise, claim), 0.8, quality, source)

    assert aggregate_evidence_status([assess("www.snopes.com", "entailment", restated)]) == "insufficient_evidence"
    assert aggregate_evidence_status([assess("www.snopes.com", "contradiction", debunk)]) == "refuted"
    mixed = [
        assess("www.snopes.com", "entailment", restated),
        assess("www.space.com", "contradiction", debunk),
        assess("forbes.com", "contradiction", debunk, quality="unknown"),
    ]
    assert aggregate_evidence_status(mixed) == "refuted"
    # a non-fact-checking reputable source that genuinely asserts the claim still supports it
    assert aggregate_evidence_status([assess("www.bbc.com", "entailment", "The mission launched in 2022.")]) == "supported"
