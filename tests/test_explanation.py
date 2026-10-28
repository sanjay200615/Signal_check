from app.services.claim_extractor import ExtractedClaim
from app.services.explanation import explain_claim
from app.services.pipeline import PipelineClaim, PipelineEvidence


def test_explanation_uses_verdict_and_sources() -> None:
    evidence = PipelineEvidence(
        passage_id="p1",
        title="A report",
        source_name="Example News",
        source_url="https://example.com",
        published_date=None,
        passage="The claim is contradicted.",
        retrieval_similarity=0.9,
        nli_label="contradiction",
        nli_score=0.9,
        source_type="live_news_result",
        source_quality="reputable",
        retrieved_date="2026-10-27",
    )
    result = PipelineClaim(
        ExtractedClaim("claim_1", "Trump is dead", ["Trump"], "factual"),
        "refuted",
        [evidence],
        "Evidence contradicts the claim.",
    )

    explanation = explain_claim(result)

    assert "Trump is dead" in explanation
    assert "likely false" in explanation
    assert "Example News" in explanation
