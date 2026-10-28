"""Reusable claim extraction and evidence-verification boundary."""

from dataclasses import dataclass

from app.services.claim_extractor import ExtractedClaim, extract_claims


@dataclass(frozen=True)
class ClaimSource:
    input_id: str
    text: str


def extract_factual_claims(source: ClaimSource) -> list[ExtractedClaim]:
    """Extract factual claims while keeping the originating input available."""
    return [
        claim
        for claim in extract_claims(source.text)
        if claim.claim_type == "factual"
    ]
