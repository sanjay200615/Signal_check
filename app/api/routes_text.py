from functools import lru_cache
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.text_analysis import (
    ClaimResult,
    EvidenceResult,
    TextAnalysisRequest,
    TextAnalysisResponse,
    UrlAnalysisRequest,
)
from app.core.config import Settings
from app.core.readiness import readiness
from app.services.pipeline import TextAnalysisPipeline, build_default_pipeline
from app.services.explanation import explain_claim
from app.services.multimodal import extract_image_text, fetch_url_text

router = APIRouter(prefix="/api/analyze", tags=["analysis"])
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
settings = Settings.from_environment()


@lru_cache(maxsize=1)
def get_pipeline() -> TextAnalysisPipeline:
    pipeline = build_default_pipeline(
        ROOT,
        top_k=settings.top_k,
        minimum_similarity=settings.minimum_similarity,
        evidence_provider=settings.evidence_provider,
    )
    readiness.mark_ready(
        index_loaded=settings.evidence_provider == "local",
        embedding_model_loaded=settings.evidence_provider == "local",
        nli_model_loaded=True,
    )
    return pipeline


@router.post("/text", response_model=TextAnalysisResponse)
def analyze_text(
    request: TextAnalysisRequest,
) -> TextAnalysisResponse:
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text must not be blank.")
    if len(request.text) > settings.max_text_length:
        raise HTTPException(
            status_code=422,
            detail=f"text exceeds the configured {settings.max_text_length}-character limit.",
        )
    try:
        claims, overall_status = get_pipeline().analyze(request.text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (OSError, RuntimeError) as error:
        readiness.mark_error(str(error))
        raise HTTPException(
            status_code=503,
            detail="Text analysis components are unavailable.",
        ) from error

    response = _build_response("text", request.text.strip(), claims, overall_status)
    logger.info(
        "text_response_built final_status=%s claims=%d evidence=%d",
        response.overall_status,
        len(response.claims),
        sum(len(claim.evidence) for claim in response.claims),
    )
    return response


@router.post("/url", response_model=TextAnalysisResponse)
def analyze_url(request: UrlAnalysisRequest) -> TextAnalysisResponse:
    try:
        extracted_text = fetch_url_text(request.url)
        claims, overall_status = get_pipeline().analyze(extracted_text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (OSError, RuntimeError) as error:
        readiness.mark_error(str(error))
        raise HTTPException(status_code=503, detail=str(error)) from error
    return _build_response("url", extracted_text, claims, overall_status)


@router.post("/image", response_model=TextAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)) -> TextAnalysisResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="file must be an image.")
    image_bytes = await file.read()
    if len(image_bytes) > 10_000_000:
        raise HTTPException(status_code=422, detail="image must be 10 MB or smaller.")
    try:
        extracted_text = extract_image_text(image_bytes)
        claims, overall_status = get_pipeline().analyze(extracted_text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (OSError, RuntimeError) as error:
        readiness.mark_error(str(error))
        raise HTTPException(status_code=503, detail="Image analysis is unavailable.") from error
    return _build_response("image", extracted_text, claims, overall_status)


def _build_response(
    input_type: str,
    extracted_text: str,
    claims,
    overall_status: str,
) -> TextAnalysisResponse:
    return TextAnalysisResponse(
        input_type=input_type,
        extracted_text=extracted_text,
        claims=[
            ClaimResult(
                claim_id=result.claim.claim_id,
                claim_text=result.claim.claim_text,
                entities=result.claim.entities,
                claim_type=result.claim.claim_type,
                verification_status=result.status,
                evidence=[
                    EvidenceResult(
                        passage_id=item.passage_id,
                        title=item.title,
                        source_name=item.source_name,
                        source_url=item.source_url,
                        published_date=item.published_date,
                        passage=item.passage,
                        retrieval_similarity=item.retrieval_similarity,
                        nli_label=item.nli_label,
                        nli_score=item.nli_score,
                        nli_class_scores=item.nli_class_scores,
                        source_type=item.source_type,
                        source_quality=item.source_quality,
                        source_quality_weight=item.source_quality_weight,
                        retrieved_date=item.retrieved_date,
                        provider=item.provider,
                        last_updated_date=item.last_updated_date,
                        domain=item.domain,
                    )
                    for item in result.evidence
                ],
                reason=result.reason,
                analysis=explain_claim(result),
            )
            for result in claims
        ],
        overall_status=overall_status,
        limitations=[
            (
                "Evidence was retrieved from the local corpus."
                if settings.evidence_provider == "local"
                else (
                    "Evidence was retrieved from Wikipedia and Google News live search."
                    if settings.evidence_provider == "live"
                    else "Evidence was retrieved from Tavily live web search."
                )
            ),
            "Semantic similarity indicates relevance, not truth.",
            "NLI outputs are model estimates, not proof of truth.",
        ],
    )
