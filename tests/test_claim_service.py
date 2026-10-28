from app.services.claim_service import ClaimSource, extract_factual_claims


def test_claim_service_keeps_only_factual_claims() -> None:
    claims = extract_factual_claims(
        ClaimSource("input-1", "NASA published a report. Why did it matter?")
    )

    assert [claim.claim_text for claim in claims] == ["NASA published a report"]
