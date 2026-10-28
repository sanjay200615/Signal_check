"""End-to-end text claim analysis pipeline."""

from dataclasses import dataclass
import logging
import os
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from app.services.claim_extractor import ExtractedClaim
from app.services.claim_service import ClaimSource, extract_factual_claims
from app.services.evidence_provider import EvidenceProvider
from app.services.verifier import (
    EvidenceAssessment,
    NLIModel,
    VerificationSignal,
    aggregate_evidence_status,
    is_directly_relevant,
    is_fiction_source,
    verify_claim,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineEvidence:
    passage_id: str
    title: str
    source_name: str
    source_url: str | None
    published_date: str | None
    passage: str
    retrieval_similarity: float
    nli_label: str
    nli_score: float
    source_type: str
    source_quality: str
    retrieved_date: str
    provider: str = "local"
    nli_class_scores: dict[str, float] | None = None
    source_quality_weight: float = 0.8
    last_updated_date: str | None = None
    domain: str | None = None


@dataclass(frozen=True)
class PipelineClaim:
    claim: ExtractedClaim
    status: str
    evidence: list[PipelineEvidence]
    reason: str


class TextAnalysisPipeline:
    def __init__(
        self,
        evidence_provider: EvidenceProvider,
        nli_model: NLIModel,
        *,
        top_k: int = 3,
        minimum_similarity: float = 0.3,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not 0 <= minimum_similarity <= 1:
            raise ValueError("minimum_similarity must be between zero and one.")
        self._evidence_provider = evidence_provider
        self._nli_model = nli_model
        self._top_k = top_k
        self._minimum_similarity = minimum_similarity

    @property
    def nli_model(self) -> NLIModel:
        return self._nli_model

    def analyze(self, text: str) -> tuple[list[PipelineClaim], str]:
        logger.info("text_analysis_started input_length=%d", len(text))
        claims = extract_factual_claims(ClaimSource(input_id="text", text=text))
        logger.info(
            "claim_extraction_completed factual_claims=%d claims=%s",
            len(claims),
            [claim.claim_text for claim in claims],
        )
        results = [self._analyze_claim(claim) for claim in claims]
        statuses = [result.status for result in results]
        overall = _overall_status(statuses)
        logger.info(
            "text_analysis_completed final_status=%s claim_statuses=%s",
            overall,
            statuses,
        )
        return results, overall

    def _analyze_claim(self, claim: ExtractedClaim) -> PipelineClaim:
        queries = _search_queries_for_claim(claim.claim_text)
        logger.info(
            "claim_search_started claim=%r generated_queries=%s provider=%s",
            claim.claim_text,
            queries,
            type(self._evidence_provider).__name__,
        )
        retrieved = []
        seen_urls: set[str] = set()
        raw_result_count = 0
        for query in queries:
            query_results = self._evidence_provider.search(query, top_k=self._top_k)
            raw_result_count += len(query_results)
            for item in query_results:
                key = (
                    _canonical_source_url(item.source_url)
                    if item.source_url
                    else f"{item.source_name}:{item.title}".lower()
                )
                if key in seen_urls:
                    continue
                seen_urls.add(key)
                retrieved.append(item)
        logger.info(
            "claim_search_completed claim=%r raw_results=%d deduplicated_results=%d",
            claim.claim_text,
            raw_result_count,
            len(retrieved),
        )
        assessments: list[EvidenceAssessment] = []
        evidence: list[PipelineEvidence] = []
        relevant_count = 0
        similarity_count = 0
        nli_count = 0
        for item in retrieved:
            if is_fiction_source(item.title):
                logger.info(
                    "evidence_rejected stage=fiction_source claim=%r title=%r",
                    claim.claim_text,
                    item.title,
                )
                continue
            if not is_directly_relevant(claim.claim_text, f"{item.title}. {item.passage}"):
                logger.info(
                    "evidence_rejected stage=relevance claim=%r title=%r",
                    claim.claim_text,
                    item.title,
                )
                continue
            relevant_count += 1
            if item.retrieval_similarity < self._minimum_similarity:
                logger.info(
                    "evidence_rejected stage=similarity claim=%r title=%r score=%.3f threshold=%.3f",
                    claim.claim_text,
                    item.title,
                    item.retrieval_similarity,
                    self._minimum_similarity,
                )
                continue
            similarity_count += 1
            signal = verify_claim(claim.claim_text, item.passage, self._nli_model)
            nli_count += 1
            assessments.append(
                EvidenceAssessment(signal, item.retrieval_similarity, item.source_quality, item.domain or item.source_name)
            )
            evidence.append(
                PipelineEvidence(
                    passage_id=item.passage_id,
                    title=item.title,
                    source_name=item.source_name,
                    source_url=item.source_url,
                    published_date=item.published_date,
                    passage=item.passage,
                    retrieval_similarity=item.retrieval_similarity,
                    nli_label=signal.label,
                    nli_score=signal.score,
                    nli_class_scores=signal.class_scores,
                    source_type=item.source_type,
                    source_quality=item.source_quality,
                    source_quality_weight=item.source_quality_weight,
                    retrieved_date=item.retrieved_date,
                    provider=item.provider,
                    last_updated_date=item.last_updated_date,
                    domain=item.domain,
                )
            )
        status = aggregate_evidence_status(assessments)
        logger.info(
            "claim_pipeline_counts claim=%r raw_results=%d deduplicated_results=%d "
            "relevance_results=%d similarity_results=%d nli_results=%d "
            "aggregation_results=%d final_status=%s reason=%s",
            claim.claim_text,
            raw_result_count,
            len(retrieved),
            relevant_count,
            similarity_count,
            nli_count,
            len(assessments),
            status,
            _reason_for_status(status),
        )
        return PipelineClaim(
            claim=claim,
            status=status,
            evidence=evidence,
            reason=_reason_for_status(status),
        )


def build_default_pipeline(
    root: Path,
    *,
    top_k: int = 3,
    minimum_similarity: float = 0.3,
    evidence_provider: str = "live",
) -> TextAnalysisPipeline:
    from app.services.evidence_provider import (
        HybridEvidenceProvider,
        LocalEvidenceProvider,
        TavilyEvidenceProvider,
    )
    from app.services.verifier import DEFAULT_NLI_MODEL, BartNLIModel

    if evidence_provider == "local":
        from app.services.retriever import EvidenceIndex, SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
        index = EvidenceIndex.load(
            root / "data/index/evidence.faiss",
            root / "data/index/metadata.json",
        )
        provider: EvidenceProvider = LocalEvidenceProvider(index, embedder)
    elif evidence_provider == "live":
        from app.services.cache import default_cache

        provider = HybridEvidenceProvider(cache=default_cache())
    elif evidence_provider == "tavily":
        try:
            provider = TavilyEvidenceProvider()
        except ValueError as error:
            raise RuntimeError(
                "The Tavily evidence provider is unavailable. "
                "Set TAVILY_API_KEY or use EVIDENCE_PROVIDER=local."
            ) from error
    else:
        raise ValueError("evidence_provider must be local, live, or tavily.")

    return TextAnalysisPipeline(
        evidence_provider=provider,
        nli_model=BartNLIModel(os.getenv("NLI_MODEL", DEFAULT_NLI_MODEL)),
        top_k=top_k,
        minimum_similarity=minimum_similarity,
    )


def _overall_status(statuses: list[str]) -> str:
    if not statuses:
        return "insufficient_evidence"
    if "conflicting_evidence" in statuses:
        return "conflicting_evidence"
    if "refuted" in statuses:
        return "refuted"
    if all(status == "supported" for status in statuses):
        return "supported"
    if all(status == "insufficient_evidence" for status in statuses):
        return "insufficient_evidence"
    return "uncertain"


def _reason_for_status(status: str) -> str:
    return {
        "supported": "Retrieved evidence appears to support the claim, subject to source and model limitations.",
        "refuted": "Retrieved evidence appears to contradict the claim, subject to source and model limitations.",
        "conflicting_evidence": "Retrieved evidence contains materially inconsistent signals.",
        "insufficient_evidence": "The configured evidence did not establish support or contradiction.",
    }[status]


def _search_queries_for_claim(claim: str) -> list[str]:
    """Create a small set of claim-preserving news queries for life-status claims."""
    normalized = " ".join(claim.split())
    words = set(re.findall(r"[a-z0-9]+", normalized.lower()))
    if words.intersection({"alive", "dead", "died", "dies", "death", "killed"}):
        subject = re.split(
            r"\b(?:is|was|are|were|has|have|had|died|dies|dead|alive|killed|death)\b",
            normalized,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        if subject:
            return list(dict.fromkeys([normalized, f"{subject} death", f"{subject} alive"]))
    return [normalized]


def _canonical_source_url(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))
