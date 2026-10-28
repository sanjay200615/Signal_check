"""Modality-neutral contracts shared by the multimodal pipeline."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class InputType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    URL = "url"
    AUDIO = "audio"
    VIDEO = "video"


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Verdict(StrEnum):
    GENUINE = "genuine"
    DECEPTIVE = "deceptive"
    NEEDS_REVIEW = "needs_review"


class CaseInputContract(BaseModel):
    input_id: str
    input_type: InputType
    created_at: datetime
    content_hash: str = Field(min_length=64, max_length=64)
    text: str | None = None
    source_url: str | None = None
    file_name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    transcription_status: str | None = None
    transcription_error: str | None = None
    audio_meta: dict[str, Any] | None = None


class ClaimContract(BaseModel):
    claim_id: str
    claim_text: str = Field(min_length=1)
    source_input_ids: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    claim_type: str = "factual"
    verification_status: str | None = None


class ModalitySignal(BaseModel):
    signal_id: str
    input_id: str
    modality: InputType
    signal_type: str
    status: ProcessingStatus
    score: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    details: dict[str, Any] = Field(default_factory=dict)


class EvidenceContract(BaseModel):
    evidence_id: str
    claim_id: str
    passage: str
    source_url: str | None = None
    source_name: str
    retrieval_score: float = Field(ge=0, le=1)
    nli_label: str
    nli_score: float = Field(ge=0, le=1)
    source_type: str | None = None
    source_quality: str | None = None
    source_quality_weight: float | None = Field(default=None, ge=0, le=1)
    published_date: str | None = None
    retrieved_date: str | None = None
    provider: str | None = None
    domain: str | None = None


class ExplanationItem(BaseModel):
    kind: str  # risk | cross_channel | claim | reassurance | info
    text: str
    refs: list[str] = Field(default_factory=list)  # input ids or evidence ids behind the statement


class CaseAnalysisContract(BaseModel):
    case_id: str
    status: ProcessingStatus
    inputs: list[CaseInputContract] = Field(default_factory=list)
    claims: list[ClaimContract] = Field(default_factory=list)
    signals: list[ModalitySignal] = Field(default_factory=list)
    evidence: list[EvidenceContract] = Field(default_factory=list)
    verdict: Verdict | None = None
    risk_score: float | None = Field(default=None, ge=0, le=1)
    ai_investigation: dict[str, Any] | None = None
    decision_basis: str | None = None
    recommended_action: str | None = None
    limitations: list[str] = Field(default_factory=list)
    content_assessment: str | None = None
    deception_risk: str | None = None
    intent_assessment: str = "not_determinable_from_content"
    risk_breakdown: dict[str, float] = Field(default_factory=dict)
    explanation: list[ExplanationItem] = Field(default_factory=list)
