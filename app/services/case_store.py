"""Temporary in-process storage for manually submitted verification cases."""

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha256
from threading import Lock
from uuid import uuid4


SUPPORTED_INPUT_TYPES = {"text", "url", "image", "audio"}


@dataclass(frozen=True)
class CaseInput:
    input_id: str
    input_type: str
    created_at: str
    content_hash: str
    text: str | None = None
    source_url: str | None = None
    file_name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    raw_content: bytes | None = field(default=None, repr=False, compare=False)
    transcription_status: str | None = None  # supplied | processing | completed | failed (audio only)
    transcription_error: str | None = None
    audio_meta: dict | None = None


@dataclass
class VerificationCase:
    case_id: str
    created_at: str
    status: str = "collecting_inputs"
    inputs: list[CaseInput] = field(default_factory=list)


class CaseStore:
    """Keep raw inputs only in process memory until analysis is implemented."""

    def __init__(self) -> None:
        self._cases: dict[str, VerificationCase] = {}
        self._lock = Lock()

    def create(self) -> VerificationCase:
        case = VerificationCase(
            case_id=f"case_{uuid4().hex}",
            created_at=_now(),
        )
        with self._lock:
            self._cases[case.case_id] = case
        return case

    def get(self, case_id: str) -> VerificationCase:
        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                raise KeyError(case_id)
            return _copy_case(case)

    def delete(self, case_id: str) -> None:
        with self._lock:
            if case_id not in self._cases:
                raise KeyError(case_id)
            del self._cases[case_id]

    def add_input(self, case_id: str, item: CaseInput) -> VerificationCase:
        if item.input_type not in SUPPORTED_INPUT_TYPES:
            raise ValueError(f"Unsupported input type: {item.input_type}.")
        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                raise KeyError(case_id)
            case.inputs.append(item)
            return _copy_case(case)

    def set_audio_result(
        self,
        case_id: str,
        input_id: str,
        *,
        text: str | None,
        metadata: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Record the outcome of transcribing one audio input and update the case status."""
        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                raise KeyError(case_id)
            for index, item in enumerate(case.inputs):
                if item.input_id == input_id:
                    case.inputs[index] = replace(
                        item,
                        text=text or None,
                        transcription_status="failed" if error else "completed",
                        transcription_error=error,
                        audio_meta=metadata,
                    )
                    break
            else:
                raise KeyError(input_id)
            pending = any(item.transcription_status == "processing" for item in case.inputs)
            case.status = "processing" if pending else "collecting_inputs"

    def set_status(self, case_id: str, status: str) -> None:
        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                raise KeyError(case_id)
            case.status = status


def build_text_input(text: str) -> CaseInput:
    normalized = text.strip()
    if not normalized:
        raise ValueError("text must not be blank.")
    return CaseInput(
        input_id=f"input_{uuid4().hex}",
        input_type="text",
        created_at=_now(),
        content_hash=_hash(normalized.encode("utf-8")),
        text=normalized,
    )


def build_url_input(url: str) -> CaseInput:
    normalized = url.strip()
    if not normalized:
        raise ValueError("url must not be blank.")
    return CaseInput(
        input_id=f"input_{uuid4().hex}",
        input_type="url",
        created_at=_now(),
        content_hash=_hash(normalized.encode("utf-8")),
        source_url=normalized,
    )


def build_image_input(
    content: bytes,
    *,
    file_name: str | None,
    mime_type: str | None,
) -> CaseInput:
    if not content:
        raise ValueError("image must not be empty.")
    return CaseInput(
        input_id=f"input_{uuid4().hex}",
        input_type="image",
        created_at=_now(),
        content_hash=_hash(content),
        file_name=file_name,
        mime_type=mime_type,
        size_bytes=len(content),
        raw_content=content,
    )


def build_audio_input(
    content: bytes,
    *,
    file_name: str | None,
    mime_type: str | None,
    transcript: str | None = None,
) -> CaseInput:
    if not content:
        raise ValueError("audio must not be empty.")
    return CaseInput(
        input_id=f"input_{uuid4().hex}",
        input_type="audio",
        created_at=_now(),
        content_hash=_hash(content),
        text=transcript.strip() if transcript and transcript.strip() else None,
        file_name=file_name,
        mime_type=mime_type,
        size_bytes=len(content),
        raw_content=content,
        transcription_status="supplied" if transcript and transcript.strip() else "processing",
        audio_meta={"source": "supplied"} if transcript and transcript.strip() else None,
    )


def _copy_case(case: VerificationCase) -> VerificationCase:
    return VerificationCase(
        case_id=case.case_id,
        created_at=case.created_at,
        status=case.status,
        inputs=list(case.inputs),
    )


def _hash(content: bytes) -> str:
    return sha256(content).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
