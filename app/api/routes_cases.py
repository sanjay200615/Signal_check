"""API routes for collecting related manual case inputs."""

from dataclasses import asdict

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app.schemas.cases import (
    AddTextInputRequest,
    AddUrlInputRequest,
    CaseInputResponse,
    CaseResponse,
)
from app.schemas.common import CaseAnalysisContract
from app.services.audio import get_transcriber
from app.services.case_analysis import analyze_case
from app.api.routes_text import get_pipeline
from app.services.case_store import (
    CaseInput,
    CaseStore,
    build_image_input,
    build_audio_input,
    build_text_input,
    build_url_input,
)

router = APIRouter(prefix="/api/cases", tags=["cases"])
case_store = CaseStore()


@router.post("", response_model=CaseResponse, status_code=201)
def create_case() -> CaseResponse:
    return _case_response(case_store.create())


@router.get("/{case_id}", response_model=CaseResponse)
def get_case(case_id: str) -> CaseResponse:
    try:
        return _case_response(case_store.get(case_id))
    except KeyError as error:
        raise HTTPException(status_code=404, detail="case not found.") from error


@router.delete("/{case_id}", status_code=204)
def delete_case(case_id: str) -> None:
    try:
        case_store.delete(case_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="case not found.") from error


@router.post("/{case_id}/analyze", response_model=CaseAnalysisContract)
def analyze_manual_case(case_id: str) -> CaseAnalysisContract:
    try:
        case = case_store.get(case_id)
        if not case.inputs:
            raise HTTPException(
                status_code=422,
                detail="case must contain at least one input.",
            )
        if any(item.transcription_status == "processing" for item in case.inputs):
            raise HTTPException(
                status_code=409,
                detail="Audio is still being transcribed; try again in a moment.",
            )
        pipeline = get_pipeline()
        result = analyze_case(
            case,
            text_analyzer=pipeline.analyze,
            nli_model=getattr(pipeline, "nli_model", None),
        )
        case_store.set_status(case_id, "completed")
        return result
    except KeyError as error:
        raise HTTPException(status_code=404, detail="case not found.") from error
    except HTTPException:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        case_store.set_status(case_id, "failed")
        raise HTTPException(
            status_code=503,
            detail="case analysis is unavailable.",
        ) from error


@router.post("/{case_id}/inputs/text", response_model=CaseResponse, status_code=201)
def add_text_input(case_id: str, request: AddTextInputRequest) -> CaseResponse:
    try:
        case = case_store.add_input(case_id, build_text_input(request.text))
    except KeyError as error:
        raise HTTPException(status_code=404, detail="case not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _case_response(case)


@router.post("/{case_id}/inputs/url", response_model=CaseResponse, status_code=201)
def add_url_input(case_id: str, request: AddUrlInputRequest) -> CaseResponse:
    try:
        case = case_store.add_input(case_id, build_url_input(request.url))
    except KeyError as error:
        raise HTTPException(status_code=404, detail="case not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _case_response(case)


@router.post("/{case_id}/inputs/image", response_model=CaseResponse, status_code=201)
async def add_image_input(
    case_id: str,
    file: UploadFile = File(...),
) -> CaseResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="file must be an image.")
    content = await file.read()
    if len(content) > 10_000_000:
        raise HTTPException(status_code=422, detail="image must be 10 MB or smaller.")
    try:
        item = build_image_input(
            content,
            file_name=file.filename,
            mime_type=file.content_type,
        )
        case = case_store.add_input(case_id, item)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="case not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _case_response(case)


@router.post("/{case_id}/inputs/audio", response_model=CaseResponse, status_code=202)
async def add_audio_input(
    case_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    transcript: str = Form(""),
) -> CaseResponse:
    if file.content_type and not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=422, detail="file must be audio.")
    content = await file.read()
    if len(content) > 25_000_000:
        raise HTTPException(status_code=422, detail="audio must be 25 MB or smaller.")
    try:
        item = build_audio_input(
            content,
            file_name=file.filename,
            mime_type=file.content_type,
            transcript=transcript,
        )
        case = case_store.add_input(case_id, item)
        case_store.set_status(case_id, "processing")
        case.status = "processing"
        background_tasks.add_task(_process_audio, case_id, item.input_id, content)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="case not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _case_response(case)


def _process_audio(case_id: str, input_id: str, content: bytes) -> None:
    """Transcribe an uploaded voice note in the background (or accept the supplied transcript)."""
    try:
        item = next(i for i in case_store.get(case_id).inputs if i.input_id == input_id)
        if item.text:  # the user supplied a transcript
            case_store.set_audio_result(case_id, input_id, text=item.text, metadata=item.audio_meta)
            return
        transcript = get_transcriber().transcribe(content)
        case_store.set_audio_result(case_id, input_id, text=transcript.text, metadata=transcript.metadata())
    except (KeyError, StopIteration):
        return
    except (ValueError, RuntimeError) as error:
        case_store.set_audio_result(case_id, input_id, text=None, error=str(error))


def _case_response(case) -> CaseResponse:
    return CaseResponse(
        case_id=case.case_id,
        created_at=case.created_at,
        status=case.status,
        inputs=[
            CaseInputResponse(**{
                key: value
                for key, value in asdict(item).items()
                if key != "raw_content"
            })
            for item in case.inputs
        ],
    )
