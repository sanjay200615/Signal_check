"""End-to-end analysis for a manually collected verification case."""

import logging

from app.schemas.common import (
    CaseAnalysisContract,
    CaseInputContract,
    InputType,
    ModalitySignal,
    ProcessingStatus,
)
from app.services.adapters import ADAPTERS, AdapterContext, AdapterOutput, TextAnalyzer
from app.services.case_store import VerificationCase
from app.services.case_explanation import build_explanation
from app.services.correlation import correlate_texts
from app.services.cross_channel import MAX_CROSS_CHANNEL_SCORE, ChannelInput, analyze_cross_channel
from app.services.reputation import default_reputation
from app.services.fusion import REVIEW_THRESHOLD, fuse_signals
from app.services.investigator import investigate_case

logger = logging.getLogger(__name__)

AI_ESCALATION_CONFIDENCE = 0.6


class _AiNotNeeded(Exception):
    """Internal: skip the AI call when it cannot change the outcome."""


def analyze_case(
    case: VerificationCase,
    *,
    text_analyzer: TextAnalyzer,
    nli_model: object | None = None,
    cross_channel: bool = True,
) -> CaseAnalysisContract:
    """Run each input through its modality adapter, then fuse the signals.

    The verdict comes only from deterministic fusion. The optional LLM
    investigation is attached as an advisory second opinion and never changes it.
    """
    inputs = [
        CaseInputContract(
            input_id=item.input_id,
            input_type=InputType(item.input_type),
            created_at=item.created_at,
            content_hash=item.content_hash,
            text=item.text,
            source_url=item.source_url,
            file_name=item.file_name,
            mime_type=item.mime_type,
            size_bytes=item.size_bytes,
            transcription_status=item.transcription_status,
            transcription_error=item.transcription_error,
            audio_meta=item.audio_meta,
        )
        for item in case.inputs
    ]
    context = AdapterContext(
        text_analyzer=text_analyzer,
        related_message_text=" ".join(item.text for item in case.inputs if item.text),
        urls=[item.source_url for item in case.inputs if item.source_url],
    )
    merged = AdapterOutput()
    for item in case.inputs:
        adapter = ADAPTERS.get(item.input_type)
        if adapter is None:
            logger.warning("no_adapter input_type=%s", item.input_type)
            continue
        output = adapter.analyze(item, context)
        merged.claims.extend(output.claims)
        merged.evidence.extend(output.evidence)
        merged.signals.extend(output.signals)
        merged.text_items.extend(output.text_items)
        merged.image_context.extend(output.image_context)
        merged.contradiction = max(merged.contradiction, output.contradiction)
        merged.phishing = max(merged.phishing, output.phishing)
        merged.image_anomaly = max(merged.image_anomaly, output.image_anomaly)
        merged.urls_checked += output.urls_checked
        merged.urls_trusted += output.urls_trusted
        merged.media_needing_review += output.media_needing_review

    claims, evidence, signals = merged.claims, merged.evidence, merged.signals
    correlation = correlate_texts(merged.text_items)
    if merged.text_items:
        signals.append(
            ModalitySignal(
                signal_id=f"{case.case_id}_correlation",
                input_id=case.case_id,
                modality=InputType.TEXT,
                signal_type="cross_input_coordination",
                status=ProcessingStatus.COMPLETED,
                score=correlation.score,
                confidence=1.0,
                details={
                    "shared_urls": list(correlation.shared_urls),
                    "shared_entities": list(correlation.shared_entities),
                    "similar_pairs": [list(pair) for pair in correlation.similar_pairs],
                },
            )
        )

    cross_findings = []
    if cross_channel and len(case.inputs) > 1:
        ocr_by_input = {image["input_id"]: image.get("ocr_text") or "" for image in merged.image_context}
        channel_inputs = [
            ChannelInput(item.input_id, item.input_type, item.text or ocr_by_input.get(item.input_id, ""))
            for item in case.inputs
        ]
        cross_findings = analyze_cross_channel(channel_inputs, default_reputation(), claims=claims, nli=nli_model)
    cross_score = min(MAX_CROSS_CHANNEL_SCORE, sum(finding.weight for finding in cross_findings))
    if cross_findings:
        signals.append(
            ModalitySignal(
                signal_id=f"{case.case_id}_cross_channel",
                input_id=case.case_id,
                modality=InputType.TEXT,
                signal_type="cross_channel",
                status=ProcessingStatus.COMPLETED,
                score=cross_score,
                confidence=1.0,
                details={
                    "indicators": [finding.indicator for finding in cross_findings],
                    "findings": [
                        {"indicator": f.indicator, "detail": f.detail, "input_ids": list(f.input_ids)}
                        for f in cross_findings
                    ],
                },
            )
        )

    statuses = {claim.verification_status for claim in claims}
    has_support = "supported" in statuses and not statuses & {"refuted", "conflicting_evidence"}
    trusted_domain = merged.urls_checked > 0 and merged.urls_trusted == merged.urls_checked
    fusion = fuse_signals(
        claim_contradiction=merged.contradiction,
        phishing=merged.phishing,
        coordination=correlation.score,
        image_anomaly=merged.image_anomaly,
        cross_channel=cross_score,
        has_evidence=has_support,
        trusted_domain=trusted_domain,
    )
    ai_investigation = None
    ai_investigation_error = None
    try:
        if fusion.verdict == "deceptive":
            raise _AiNotNeeded  # the AI can only add caution, and this verdict already has the most
        if fusion.verdict == "genuine" and not has_support:
            raise _AiNotNeeded  # genuine via a trusted link rests on hard domain evidence the AI cannot improve
        ai_investigation = investigate_case(
            message_text=context.related_message_text,
            urls=[item.source_url for item in case.inputs if item.source_url],
            claims=[claim.model_dump() for claim in claims],
            evidence=[item.model_dump() for item in evidence],
            images=merged.image_context,
        )
    except _AiNotNeeded:
        pass
    except RuntimeError as error:
        logger.warning(
            "ai_investigation_unavailable error=%s",
            type(error).__name__,
        )
        ai_investigation_error = {
            "provider": "groq",
            "status": "unavailable",
            "message": str(error),
        }

    verdict = fusion.verdict
    risk_score = fusion.risk_score
    image_requires_review = any(
        image["visual_truth_status"] == "needs_review" for image in merged.image_context
    )
    # The AI may only make a verdict more cautious, never clear anything or call something deceptive on its own.
    ai_escalated = (
        verdict == "genuine"
        and ai_investigation is not None
        and ai_investigation.verdict == "deceptive"
        and ai_investigation.confidence >= AI_ESCALATION_CONFIDENCE
    )
    if ai_escalated:
        verdict = "needs_review"
        risk_score = max(risk_score, REVIEW_THRESHOLD)
    if verdict == "genuine" and "conflicting_evidence" in statuses:
        verdict = "needs_review"
        risk_score = max(risk_score, REVIEW_THRESHOLD)
    if merged.media_needing_review and verdict == "genuine":
        verdict = "needs_review"  # a recording with no intelligible speech cannot be verified
        risk_score = max(risk_score, 0.5)
    if image_requires_review and verdict == "genuine":
        verdict = "needs_review"
        risk_score = max(risk_score, 0.5)
    if verdict == "deceptive":
        decision_basis = "Strong phishing, contradiction, or coordinated-risk signals crossed the decision threshold."
        recommended_action = "Do not trust the content or submit credentials; verify through an independent official channel."
        deception_risk = "high"
    elif has_support and verdict == "genuine":
        decision_basis = "The claims were supported by the retrieved evidence and no risk indicators were found, but this is not proof of authenticity."
        recommended_action = "Review the linked sources and confirm the claim through an independent trusted source."
        deception_risk = "low"
    elif verdict == "genuine" and trusted_domain:
        decision_basis = "The linked site is a long-established, widely used domain and no phishing indicators were found, but this is not proof that the message itself is authentic."
        recommended_action = "If you were not expecting this message, open the service directly instead of using the link."
        deception_risk = "low"
    elif verdict == "needs_review" and risk_score >= REVIEW_THRESHOLD:
        found = sorted(
            {
                indicator
                for signal in signals
                for indicator in signal.details.get("indicators", [])
            }
            | {
                "image does not match its caption"
                for signal in signals
                if signal.signal_type == "image_text_consistency" and not signal.details.get("consistent", True)
            }
            | ({"the AI investigator judged this likely deceptive"} if ai_escalated else set())
        )
        decision_basis = (
            "Some risk indicators were found, but not enough to call this deceptive"
            + (f": {', '.join(name.replace('_', ' ') for name in found)}." if found else ".")
        )
        recommended_action = "Do not click links or share information until you have verified this through an official channel."
        deception_risk = "elevated"
    else:
        decision_basis = "There was not enough relevant, source-linked evidence to make a reliable decision."
        recommended_action = "Treat the content as unverified and seek confirmation from an independent trusted source."
        deception_risk = "unknown"
    if claims and all(claim.verification_status == "supported" for claim in claims):
        content_assessment = "supported"
    elif any(claim.verification_status == "refuted" for claim in claims):
        content_assessment = "refuted"
    else:
        content_assessment = "insufficient_evidence"
    limitations = [
        "Risk scores are prototype signal scores, not calibrated probabilities.",
        "A missing or inaccessible webpage is not evidence that the content is genuine.",
        "Image analysis currently verifies OCR text and metadata; it does not prove visual authenticity or detect deepfakes.",
    ]
    if any(item.input_type == "audio" for item in case.inputs):
        limitations.append(
            "Audio analysis checks what is said, not who is speaking; cloned or synthetic voices are not detected."
        )
    if any((item.audio_meta or {}).get("translated") for item in case.inputs):
        limitations.append("Speech was translated to English by the speech model, so wording may differ from the original.")
    if ai_investigation is not None:
        investigation_payload = {**ai_investigation.__dict__, "advisory": True, "escalated": ai_escalated}
        if ai_escalated:
            limitations.append(
                "The AI investigator judged this deceptive, so the verdict was raised from genuine to "
                "needs_review as a precaution. The AI can only make verdicts more cautious."
            )
        elif ai_investigation.verdict != verdict:
            limitations.append(
                f"The AI investigator suggested '{ai_investigation.verdict}' but the "
                f"evidence-based verdict is '{verdict}'; the investigator is advisory only."
            )
    else:
        investigation_payload = (
            {**ai_investigation_error, "advisory": True} if ai_investigation_error else None
        )
    return CaseAnalysisContract(
        case_id=case.case_id,
        status=ProcessingStatus.COMPLETED,
        inputs=inputs,
        claims=claims,
        signals=signals,
        evidence=evidence,
        verdict=verdict,
        risk_score=risk_score,
        ai_investigation=investigation_payload,
        decision_basis=decision_basis,
        recommended_action=recommended_action,
        limitations=limitations,
        content_assessment=content_assessment,
        deception_risk=deception_risk,
        intent_assessment="not_determinable_from_content",
        risk_breakdown=fusion.contributions,
        explanation=build_explanation(
            signals=signals,
            claims=claims,
            evidence=evidence,
            cross_findings=cross_findings,
            trusted_domain=trusted_domain and verdict == "genuine",
            ai_escalated=ai_escalated,
        ),
    )
