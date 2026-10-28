import pytest

from app.services.claim_extractor import claims_to_dicts, extract_claims


def test_extracts_multiple_domain_independent_claims_and_entities() -> None:
    claims = extract_claims(
        "The Central Bank announced a new account policy. "
        "Customers must verify accounts at https://example.com/verify."
    )

    assert [claim.claim_id for claim in claims] == ["claim_1", "claim_2"]
    assert claims[0].claim_text == "The Central Bank announced a new account policy"
    assert "Central Bank" in claims[0].entities
    assert claims[1].claim_type == "factual"
    assert "https://example.com/verify" in claims[1].entities


def test_question_is_not_presented_as_factual_claim() -> None:
    claims = extract_claims("Why did the university cancel the examination?")

    assert len(claims) == 1
    assert claims[0].claim_type == "non_factual"


def test_results_are_json_compatible() -> None:
    result = claims_to_dicts(extract_claims("NASA published a climate report."))

    assert result == [
        {
            "claim_id": "claim_1",
            "claim_text": "NASA published a climate report",
            "entities": ["NASA"],
            "claim_type": "factual",
        }
    ]


@pytest.mark.parametrize("value", ["", "   ", None])
def test_empty_input_is_rejected(value: str | None) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        extract_claims(value)  # type: ignore[arg-type]


def test_decimals_and_version_numbers_do_not_split_claims() -> None:
    texts = [c.claim_text for c in extract_claims(
        "The Python 3.13 release notes are available. Pay $2.99 now."
    )]
    assert texts == ["The Python 3.13 release notes are available", "Pay $2.99 now"]


def test_abbreviations_do_not_end_a_sentence() -> None:
    texts = [c.claim_text for c in extract_claims("Dr. Smith said v2.0 is out. Really?")]
    assert texts == ["Dr. Smith said v2.0 is out", "Really"]


def test_personal_and_transactional_statements_are_not_check_worthy() -> None:
    from app.services.claim_service import ClaimSource, extract_factual_claims

    text = (
        "Your Amazon order has shipped and will arrive on Friday. "
        "GitHub: a new sign-in to your account was detected. "
        "I changed my number. Click here to verify. If this was you, no action is needed. "
        "We are moving the meeting to Thursday."
    )
    assert extract_factual_claims(ClaimSource("x", text)) == []


def test_public_world_facts_remain_check_worthy() -> None:
    from app.services.claim_service import ClaimSource, extract_factual_claims

    text = "Paris is the capital of France. The Moon landing in 1969 was faked. Water boils at 100 degrees Celsius."
    assert len(extract_factual_claims(ClaimSource("x", text))) == 3


def test_interjections_and_greetings_are_not_claims() -> None:
    from app.services.claim_service import ClaimSource, extract_factual_claims

    for text in ("Thanks.", "Congratulations!", "Hey, nice to see you", "Look at this photo from yesterday's event."):
        assert extract_factual_claims(ClaimSource("x", text)) == [], text


def test_lure_phrases_are_removed_from_the_claim_text() -> None:
    texts = [c.claim_text for c in extract_claims("dua lipa dead watch video")]
    assert texts == ["dua lipa dead"]
    assert [c.claim_text for c in extract_claims("Breaking: X is dead, click here for the full video")] == ["Breaking: X is dead"]
    assert [c.claim_text for c in extract_claims("Watch video")] == ["Watch video"]  # nothing left to keep


def test_time_relative_claims_without_a_named_entity_are_not_checkable() -> None:
    from app.services.claim_service import ClaimSource, extract_factual_claims

    for text in (
        "The city council announced that all public schools will close tomorrow.",
        "A bus on route 42 was delayed by twenty minutes today.",
        "The bakery near the station sold out of bread by noon this morning.",
    ):
        assert extract_factual_claims(ClaimSource("x", text)) == [], text
    for text in (
        "Keanu Reeves died yesterday.",
        "NASA will launch the Artemis mission next month.",
        "The Moon landing in 1969 was faked.",
        "Bats are blind.",
    ):
        assert extract_factual_claims(ClaimSource("x", text)), text


def test_coordination_statements_are_not_claims() -> None:
    from app.services.claim_service import ClaimSource, extract_factual_claims

    for text in (
        "Just confirming dinner at 7 on Friday.",
        "Reminder: the meeting moved to room 4B.",
        "Quick note, checking in about the invoice.",
        "Hey, running late, see you soon.",
    ):
        assert extract_factual_claims(ClaimSource("x", text)) == [], text
    assert extract_factual_claims(ClaimSource("x", "NASA confirmed the launch of Artemis in 2022."))
