"""Modality adapters: turn one case input into claims, evidence and signals.

Each adapter handles a single input type and returns an ``AdapterOutput``. The
case orchestrator merges the outputs, so adding a modality means adding an
adapter and registering it in ``ADAPTERS``; the orchestrator does not change.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from typing import Protocol
from urllib.parse import urlsplit

from app.schemas.common import (
    ClaimContract,
    EvidenceContract,
    InputType,
    ModalitySignal,
    ProcessingStatus,
)
from app.services.cache import default_cache
from app.services.case_store import CaseInput
from app.services.image_analysis import assess_image
from app.services.ocr import claim_text_from_ocr, domains_in_text, read_image
from app.services.domain_analysis import is_shortener
from app.services.phishing import PhishingAssessment, assess_message, assess_url, named_brands
from app.services.reputation import default_reputation, registered_domain
from app.services.vision import image_text_similarity, vision_enabled
from app.services.url_network import domain_age_days, expand_short_url, network_checks_enabled

TextAnalyzer = Callable[[str], tuple[list[object], str]]
logger = logging.getLogger(__name__)


@dataclass
class AdapterContext:
    text_analyzer: TextAnalyzer
    related_message_text: str = ""
    urls: list[str] = field(default_factory=list)


@dataclass
class AdapterOutput:
    claims: list[ClaimContract] = field(default_factory=list)
    evidence: list[EvidenceContract] = field(default_factory=list)
    signals: list[ModalitySignal] = field(default_factory=list)
    text_items: list[tuple[str, str]] = field(default_factory=list)
    image_context: list[dict[str, object]] = field(default_factory=list)
    contradiction: float = 0.0
    phishing: float = 0.0
    image_anomaly: float = 0.0
    urls_checked: int = 0
    urls_trusted: int = 0
    media_needing_review: int = 0


class ModalityAdapter(Protocol):
    def analyze(self, item: CaseInput, context: AdapterContext) -> AdapterOutput:
        """Analyze one input and return its claims, evidence and signals."""


def verify_text(
    input_id: str,
    text: str,
    text_analyzer: TextAnalyzer,
    output: AdapterOutput,
) -> None:
    """Run claim verification on text and record the results on ``output``."""
    output.text_items.append((input_id, text))
    pipeline_claims, _ = text_analyzer(text)
    for result in pipeline_claims:
        claim_id = f"{input_id}_{result.claim.claim_id}"
        output.claims.append(
            ClaimContract(
                claim_id=claim_id,
                claim_text=result.claim.claim_text,
                source_input_ids=[input_id],
                entities=result.claim.entities,
                claim_type=result.claim.claim_type,
                verification_status=result.status,
            )
        )
        if result.status == "refuted":
            output.contradiction = max(
                output.contradiction,
                max((entry.nli_score for entry in result.evidence), default=0.0),
            )
        output.evidence.extend(
            EvidenceContract(
                evidence_id=f"{input_id}_{entry.passage_id}",
                claim_id=claim_id,
                passage=entry.passage,
                source_url=entry.source_url,
                source_name=entry.source_name,
                retrieval_score=max(0.0, min(1.0, entry.retrieval_similarity)),
                nli_label=entry.nli_label,
                nli_score=entry.nli_score,
                source_type=entry.source_type,
                source_quality=entry.source_quality,
                source_quality_weight=entry.source_quality_weight,
                published_date=entry.published_date,
                retrieved_date=entry.retrieved_date,
                provider=entry.provider,
                domain=entry.domain,
            )
            for entry in result.evidence
        )


class TextAdapter:
    """Verify factual claims in typed text."""

    def analyze(self, item: CaseInput, context: AdapterContext) -> AdapterOutput:
        output = AdapterOutput()
        if item.text:
            verify_text(item.input_id, item.text, context.text_analyzer, output)
            message = assess_message(item.text)
            if message.indicators:
                output.phishing = message.risk_score
                output.signals.append(
                    ModalitySignal(
                        signal_id=f"{item.input_id}_scam_language",
                        input_id=item.input_id,
                        modality=InputType(item.input_type),
                        signal_type="scam_language",
                        status=ProcessingStatus.COMPLETED,
                        score=message.risk_score,
                        confidence=1.0,
                        details={"indicators": list(message.indicators)},
                    )
                )
        return output


class AudioAdapter(TextAdapter):
    """A voice note contributes what was said: the transcript is checked like typed text.

    Audio can only raise risk or add information. Nothing here judges who is speaking, so a
    cloned or synthetic voice is not detected, and a recording with no intelligible speech
    can never be cleared.
    """

    def analyze(self, item: CaseInput, context: AdapterContext) -> AdapterOutput:
        output = super().analyze(item, context)
        transcript = (item.text or "").strip()
        output.signals.append(
            ModalitySignal(
                signal_id=f"{item.input_id}_audio",
                input_id=item.input_id,
                modality=InputType.AUDIO,
                signal_type="audio_transcript",
                status=ProcessingStatus.FAILED if item.transcription_status == "failed" else ProcessingStatus.COMPLETED,
                score=0.0,
                confidence=1.0,
                details={
                    **(item.audio_meta or {}),
                    "transcription_status": item.transcription_status,
                    "transcription_error": item.transcription_error,
                    "transcript_chars": len(transcript),
                },
            )
        )
        if not transcript:
            output.media_needing_review += 1
        return output


class UrlAdapter:
    """Score a URL using structural, reputation, domain-age and message-context signals."""

    def analyze(self, item: CaseInput, context: AdapterContext) -> AdapterOutput:
        output = AdapterOutput()
        if not item.source_url:
            return output
        output.text_items.append((item.input_id, item.source_url))
        reputation = default_reputation()
        checks_enabled = network_checks_enabled()
        cache = default_cache() if checks_enabled else None

        original = assess_url(
            item.source_url, context.related_message_text, reputation=reputation
        )
        final_url = item.source_url
        host = urlsplit(item.source_url).hostname or ""
        if checks_enabled and is_shortener(host):
            expanded = expand_short_url(item.source_url)
            if expanded:
                final_url = expanded
        final_host = urlsplit(final_url).hostname or ""
        age = None
        if checks_enabled:
            age = domain_age_days(registered_domain(final_host), cache=cache)
        assessment = assess_url(
            final_url,
            context.related_message_text,
            reputation=reputation,
            domain_age_days=age,
        )
        assessment = _combine(original, assessment, shortened=final_url != item.source_url)

        output.phishing = assessment.risk_score
        output.urls_checked = 1
        output.urls_trusted = int(assessment.trusted_domain)
        output.signals.append(
            ModalitySignal(
                signal_id=f"{item.input_id}_phishing",
                input_id=item.input_id,
                modality=InputType.URL,
                signal_type="phishing_risk",
                status=ProcessingStatus.COMPLETED,
                score=assessment.risk_score,
                confidence=1.0,
                details={
                    "status": assessment.status,
                    "indicators": list(assessment.indicators),
                    "trusted_domain": assessment.trusted_domain,
                    "final_url": final_url,
                    "domain_age_days": age,
                },
            )
        )
        return output


def _combine(
    original: PhishingAssessment, final: PhishingAssessment, *, shortened: bool
) -> PhishingAssessment:
    """Merge the assessment of a short link with the page it leads to."""
    if not shortened:
        return final
    indicators = tuple(dict.fromkeys(("short_link_expanded", *original.indicators, *final.indicators)))
    risk = max(original.risk_score, final.risk_score)
    status = "likely_phishing" if risk >= 0.6 else ("suspicious" if risk >= 0.3 else "likely_safe")
    return PhishingAssessment(status, risk, indicators, final.trusted_domain and original.risk_score < 0.3)


CLIP_MISMATCH_THRESHOLD = 0.237  # 2nd percentile of matched Flickr30k caption similarities
_MIN_CAPTION_WORDS, _MAX_CAPTION_WORDS = 6, 60


class ImageAdapter:
    """Read an image's text, check what it shows against the rest of the case, and inspect metadata.

    An image can only raise risk, never clear it: a screenshot's address bar, logo and text
    are all trivially forgeable, so nothing here counts as evidence of legitimacy.
    """

    def analyze(self, item: CaseInput, context: AdapterContext) -> AdapterOutput:
        output = AdapterOutput()
        if not item.raw_content:
            return output
        image = assess_image(item.raw_content)
        ocr_text = ""
        try:
            ocr_text = read_image(item.raw_content).text
        except (OSError, RuntimeError, ValueError) as error:
            logger.warning("case_image_ocr_failed input_id=%s error=%s", item.input_id, type(error).__name__)
        if len(ocr_text.strip()) < 3:
            ocr_text = ""

        if ocr_text:
            claim_text = claim_text_from_ocr(ocr_text)
            if claim_text:
                verify_text(item.input_id, claim_text, context.text_analyzer, output)
            risk, indicators, shown_domains = self._read_screenshot(ocr_text, context)
            output.phishing = risk
            if indicators:
                output.signals.append(
                    ModalitySignal(
                        signal_id=f"{item.input_id}_image_risk",
                        input_id=item.input_id,
                        modality=InputType.IMAGE,
                        signal_type="image_text_risk",
                        status=ProcessingStatus.COMPLETED,
                        score=risk,
                        confidence=1.0,
                        details={"indicators": indicators, "shown_domains": shown_domains},
                    )
                )
        else:
            self._check_caption_consistency(item, context, output)

        visual_truth_status = "needs_review" if not ocr_text else "text_extracted_only"
        metadata = {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "has_metadata": image.has_metadata,
            "perceptual_hash": image.perceptual_hash,
            "software": image.software,
            "camera": image.camera,
            "taken_at": image.taken_at,
            "has_gps": image.has_gps,
            "edited_with_software": image.edited_with_software,
            "ocr_text": ocr_text,
            "visual_truth_status": visual_truth_status,
        }
        output.image_context.append({"input_id": item.input_id, **metadata})
        output.signals.append(
            ModalitySignal(
                signal_id=f"{item.input_id}_image",
                input_id=item.input_id,
                modality=InputType.IMAGE,
                signal_type="image_metadata",
                status=ProcessingStatus.COMPLETED,
                score=0.0,
                confidence=1.0,
                details=metadata,
            )
        )
        return output

    @staticmethod
    def _read_screenshot(ocr_text: str, context: AdapterContext) -> tuple[float, list[str], list[str]]:
        reputation = default_reputation()
        message = assess_message(ocr_text)
        risk, indicators = message.risk_score, list(message.indicators)

        shown = domains_in_text(ocr_text)[:3]
        for domain in shown:
            assessment = assess_url("https://" + domain, ocr_text, reputation=reputation)
            risk = max(risk, assessment.risk_score)
            indicators.extend(assessment.indicators)

        brands = named_brands(ocr_text, reputation)
        linked = {registered_domain(urlsplit(url).hostname or "") for url in context.urls}
        linked.discard("")
        if brands and linked:
            owned = {domain for brand in brands for domain in reputation.brands[brand]}
            if not linked & owned:
                risk = max(risk, 0.7)
                indicators.append("image_brand_domain_mismatch")
        shown_registered = {registered_domain(domain) for domain in shown} - {""}
        if shown_registered and linked and not shown_registered & linked:
            risk = max(risk, 0.3)
            indicators.append("screenshot_address_differs_from_link")
        return min(1.0, risk), list(dict.fromkeys(indicators)), shown

    @staticmethod
    def _check_caption_consistency(item: CaseInput, context: AdapterContext, output: AdapterOutput) -> None:
        """For a text-free photo, compare it with a descriptive message using CLIP."""
        message = context.related_message_text.strip()
        word_count = len(message.split())
        if not vision_enabled() or not _MIN_CAPTION_WORDS <= word_count <= _MAX_CAPTION_WORDS:
            return
        try:
            similarity = image_text_similarity(item.raw_content, [message])[0]
        except (OSError, RuntimeError, ValueError) as error:
            logger.warning("clip_unavailable input_id=%s error=%s", item.input_id, type(error).__name__)
            return
        mismatch = similarity < CLIP_MISMATCH_THRESHOLD
        if mismatch:
            output.image_anomaly = 0.6
        output.signals.append(
            ModalitySignal(
                signal_id=f"{item.input_id}_caption_consistency",
                input_id=item.input_id,
                modality=InputType.IMAGE,
                signal_type="image_text_consistency",
                status=ProcessingStatus.COMPLETED,
                score=0.6 if mismatch else 0.0,
                confidence=0.7,
                details={
                    "similarity": round(similarity, 3),
                    "threshold": CLIP_MISMATCH_THRESHOLD,
                    "consistent": not mismatch,
                    "note": "CLIP image-text similarity; reliable for descriptive captions of real photos only.",
                },
            )
        )


ADAPTERS: dict[str, ModalityAdapter] = {
    "text": TextAdapter(),
    "audio": AudioAdapter(),
    "url": UrlAdapter(),
    "image": ImageAdapter(),
}
