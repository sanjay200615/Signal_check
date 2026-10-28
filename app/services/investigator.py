"""Groq-backed contextual investigation for combined manual cases."""

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.services.cache import default_cache
from app.services.multimodal import fetch_url_text
from app.services.url_network import is_public_host

RESULT_TTL_SECONDS = 24 * 3600
DEFAULT_MIN_INTERVAL_SECONDS = 3.0
DEFAULT_COOLDOWN_SECONDS = 30.0
MAX_COOLDOWN_SECONDS = 300.0
MAX_PAGES, MAX_PAGE_CHARS = 2, 3000
MAX_EVIDENCE, MAX_PASSAGE_CHARS = 8, 400
MAX_OCR_CHARS = 1500

_state_lock = threading.Lock()
_cooldown_until = 0.0
_last_call = 0.0
_clock = time.monotonic
_sleep = time.sleep


def reset_rate_limit_state() -> None:
    """Clear the cooldown and call spacing (used by tests)."""
    global _cooldown_until, _last_call
    with _state_lock:
        _cooldown_until = 0.0
        _last_call = 0.0


def _reserve_request_slot() -> None:
    """Refuse during a rate-limit cooldown; otherwise wait until calls are spaced far enough apart."""
    global _last_call
    interval = float(os.getenv("GROQ_MIN_INTERVAL_SECONDS", DEFAULT_MIN_INTERVAL_SECONDS))
    with _state_lock:
        now = _clock()
        if now < _cooldown_until:
            raise RuntimeError(f"Groq is rate-limited; the AI investigator resumes in about {int(_cooldown_until - now) + 1}s.")
        wait = _last_call + interval - now
        if wait > 0:
            _sleep(wait)
        _last_call = _clock()


def _start_cooldown(retry_after: str | None) -> float:
    """Back off for the server's Retry-After (seconds), or a default, never hammering a limited API."""
    global _cooldown_until
    try:
        seconds = float(retry_after) if retry_after is not None else DEFAULT_COOLDOWN_SECONDS
    except ValueError:
        seconds = DEFAULT_COOLDOWN_SECONDS
    seconds = min(max(seconds, 1.0), MAX_COOLDOWN_SECONDS)
    with _state_lock:
        _cooldown_until = _clock() + seconds
    return seconds


def _cache_key(message_text: str, urls: list[str], claims: list[dict[str, Any]], evidence: list[dict[str, Any]],
               images: list[dict[str, Any]], model: str) -> str:
    material = json.dumps(
        {
            "m": message_text, "u": urls, "model": model,
            "c": [(c.get("claim_text"), c.get("verification_status")) for c in claims],
            "e": sorted((e.get("passage") or "")[:200] for e in evidence),
            "i": [(i.get("ocr_text"), i.get("software")) for i in images],
        },
        ensure_ascii=True, sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InvestigatorResult:
    verdict: str
    risk_score: float
    confidence: float
    summary: str
    reasons: tuple[str, ...]
    website_findings: tuple[str, ...]
    provider: str


def investigate_case(
    *,
    message_text: str,
    urls: list[str],
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    images: list[dict[str, Any]] | None = None,
    timeout: float = 30.0,
) -> InvestigatorResult | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    cache = default_cache()
    key = _cache_key(message_text, urls, claims, evidence, images or [], model)
    cached = cache.get("groq_investigation", key, ttl=RESULT_TTL_SECONDS) if cache else None
    if cached is not None:
        _trace("groq_investigation_cache_hit key=%s", key[:12])
        return InvestigatorResult(
            **{**cached, "reasons": tuple(cached["reasons"]), "website_findings": tuple(cached["website_findings"])}
        )

    pages: list[dict[str, str]] = []
    for url in urls[:MAX_PAGES]:
        host = urlsplit(url).hostname or ""
        if not is_public_host(host):
            pages.append({"url": url, "text": "Page inspection skipped: the host is not a public internet address."})
            continue
        try:
            page_text = fetch_url_text(url, timeout=timeout)
            pages.append({"url": url, "text": page_text})
            _trace(
                "investigator_page_fetched url=%s extracted_chars=%d",
                url,
                len(page_text),
            )
        except (OSError, RuntimeError, ValueError) as error:
            pages.append({"url": url, "text": f"Page inspection failed: {error}"})
            _trace(
                "investigator_page_fetch_failed url=%s error=%s",
                url,
                type(error).__name__,
            )

    prompt = _build_prompt(message_text, pages, claims, evidence, images or [])
    _reserve_request_slot()
    _trace(
        "groq_investigation_started model=%s urls=%d claims=%d evidence=%d prompt_chars=%d",
        model,
        len(urls),
        len(claims),
        len(evidence),
        len(prompt),
    )
    _trace("groq_request_prompt_preview=%s", prompt[:4000])
    try:
        response = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a careful deception investigator. Analyze only "
                            "the supplied content and evidence. Do not invent facts. "
                            "Assess content risk, not the sender's private intent. "
                            "Never claim that a person knowingly lied based only on "
                            "the supplied content. "
                            "Treat a webpage's requests for passwords, PINs, payment, "
                            "or urgent account action as risk indicators, but explain "
                            "why. Return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as error:
        failed = getattr(error, "response", None)
        status_code = getattr(failed, "status_code", None)
        logger.warning(
            "groq_investigation_request_failed status_code=%s error=%s",
            status_code,
            type(error).__name__,
        )
        if status_code == 429:
            seconds = _start_cooldown(getattr(failed, "headers", {}).get("retry-after"))
            raise RuntimeError(f"Groq rate limit reached; the AI investigator pauses for {int(seconds)}s.") from error
        raise RuntimeError("The Groq investigation request failed.") from error

    content = _response_content(payload)
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError("The Groq investigation returned invalid JSON.") from error
    parsed = _parse_result(result)
    if cache:
        cache.set("groq_investigation", key, asdict(parsed))
    _trace(
        "groq_investigation_completed provider=%s verdict=%s risk_score=%.3f "
        "confidence=%.3f reasons=%d findings=%d",
        parsed.provider,
        parsed.verdict,
        parsed.risk_score,
        parsed.confidence,
        len(parsed.reasons),
        len(parsed.website_findings),
    )
    _trace(
        "groq_response_result=%s",
        json.dumps(
            {
                "verdict": parsed.verdict,
                "risk_score": parsed.risk_score,
                "confidence": parsed.confidence,
                "summary": parsed.summary,
                "reasons": parsed.reasons,
                "website_findings": parsed.website_findings,
            },
            ensure_ascii=True,
        ),
    )
    return parsed


def _build_prompt(
    message_text: str,
    pages: list[dict[str, str]],
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    images: list[dict[str, Any]],
) -> str:
    pages = [{"url": p["url"], "text": p["text"][:MAX_PAGE_CHARS]} for p in pages[:MAX_PAGES]]
    claims = [{"claim_text": c.get("claim_text"), "verification_status": c.get("verification_status")} for c in claims]
    ranked = sorted(evidence, key=lambda e: e.get("nli_score") or 0, reverse=True)[:MAX_EVIDENCE]
    evidence = [
        {
            "passage": (e.get("passage") or "")[:MAX_PASSAGE_CHARS],
            "source": e.get("source_name"),
            "nli_label": e.get("nli_label"),
            "nli_score": e.get("nli_score"),
            "source_quality": e.get("source_quality"),
        }
        for e in ranked
    ]
    images = [
        {
            "ocr_text": (i.get("ocr_text") or "")[:MAX_OCR_CHARS],
            "visual_truth_status": i.get("visual_truth_status"),
            "edited_with_software": i.get("edited_with_software"),
        }
        for i in images
    ]
    return json.dumps(
        {
            "message": message_text,
            "web_pages": pages,
            "verified_claims": claims,
            "retrieved_evidence": evidence,
            "images": images,
            "required_output": {
                "verdict": "genuine, deceptive, or needs_review",
                "risk_score": "number from 0 to 1",
                "confidence": "number from 0 to 1",
                "summary": "short explanation",
                "reasons": ["specific reasons grounded in supplied content"],
                "website_findings": ["what the inspected page says or requests"],
                "image_findings": ["what the image evidence supports; use needs_review reasoning when pixels are not available"],
            },
        },
        ensure_ascii=True,
    )


def _response_content(payload: Any) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("The Groq investigation response was incomplete.") from error
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The Groq investigation response was empty.")
    return content


def _parse_result(value: Any) -> InvestigatorResult:
    if not isinstance(value, dict):
        raise RuntimeError("The Groq investigation JSON was not an object.")
    verdict = value.get("verdict")
    if verdict not in {"genuine", "deceptive", "needs_review"}:
        raise RuntimeError("The Groq investigation returned an invalid verdict.")
    risk_score = _bounded_number(value.get("risk_score"), "risk_score")
    confidence = _bounded_number(value.get("confidence"), "confidence")
    summary = value.get("summary")
    reasons = value.get("reasons")
    findings = value.get("website_findings", [])
    if not isinstance(summary, str) or not summary.strip():
        raise RuntimeError("The Groq investigation returned no summary.")
    if not _string_list(reasons) or not _is_string_list(findings):
        raise RuntimeError("The Groq investigation returned invalid details.")
    return InvestigatorResult(
        verdict=verdict,
        risk_score=risk_score,
        confidence=confidence,
        summary=summary.strip(),
        reasons=tuple(reasons),
        website_findings=tuple(findings),
        provider="groq",
    )


def _bounded_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise RuntimeError(f"The Groq investigation returned an invalid {name}.")
    return float(value)


def _is_string_list(value: Any) -> bool:
    """A list of non-empty strings; an empty list is fine."""
    return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        return None
    return [item.strip() for item in value]


def _trace(message: str, *args: object) -> None:
    if os.getenv("ANALYSIS_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}:
        logger.info(message, *args)
