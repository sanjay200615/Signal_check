from pydantic import BaseModel, Field


class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class UrlAnalysisRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2_000)


class EvidenceResult(BaseModel):
    passage_id: str
    title: str
    source_name: str
    source_url: str | None
    published_date: str | None
    passage: str
    retrieval_similarity: float
    nli_label: str
    nli_score: float
    nli_class_scores: dict[str, float] | None = None
    source_type: str
    source_quality: str
    source_quality_weight: float
    retrieved_date: str
    provider: str
    last_updated_date: str | None = None
    domain: str | None = None


class ClaimResult(BaseModel):
    claim_id: str
    claim_text: str
    entities: list[str]
    claim_type: str
    verification_status: str
    evidence: list[EvidenceResult]
    reason: str
    analysis: str


class TextAnalysisResponse(BaseModel):
    input_type: str
    extracted_text: str
    claims: list[ClaimResult]
    overall_status: str
    limitations: list[str]
