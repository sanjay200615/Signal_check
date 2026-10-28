"""Schemas for the manual multimodal case workflow."""

from pydantic import BaseModel, Field


class CreateCaseResponse(BaseModel):
    case_id: str
    created_at: str
    status: str
    inputs: list["CaseInputResponse"]


class AddTextInputRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class AddUrlInputRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2_000)


class CaseInputResponse(BaseModel):
    input_id: str
    input_type: str
    created_at: str
    content_hash: str
    text: str | None = None
    source_url: str | None = None
    file_name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    transcription_status: str | None = None
    transcription_error: str | None = None
    audio_meta: dict | None = None


class CaseResponse(CreateCaseResponse):
    pass
