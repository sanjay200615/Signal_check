"""Natural-language-inference signals for claim/evidence pairs."""

from dataclasses import dataclass, field
import re
from typing import Any, Protocol


@dataclass(frozen=True)
class VerificationSignal:
    label: str
    score: float
    premise: str
    hypothesis: str
    class_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceAssessment:
    signal: VerificationSignal
    similarity: float
    source_quality: str
    source: str | None = None  # domain or site name, used to count independent sources


_LIFE_STATUS_WORDS = (
    "alive",
    "dead",
    "died",
    "dies",
    "die",
    "dying",
    "death",
    "killed",
    "deceased",
    "survived",
)


_FICTION_TITLE = re.compile(
    r"\((?:\d{4}\s+)?(?:[a-z]+\s+)?"
    r"(?:film|movie|tv series|tv show|television series|miniseries|video game|novel|"
    r"album|song|band|musical|play|comic|comics|anime|manga|franchise|character)\)\s*$",
    re.IGNORECASE,
)


def is_fiction_source(title: str) -> bool:
    """True for entertainment articles (a film, novel, game...) judged by title suffix.

    These describe invented events, so they cannot support or refute a claim
    about the real world even when their text matches the claim closely.
    """
    return bool(_FICTION_TITLE.search(title.strip()))


_MINIMUM_TERM_COVERAGE = 0.5
_STOPWORDS = frozenset(
    "a an the is are was were be been being am do does did has have had of in on at to for from by "
    "with as and or but not no if then than that this these those it its into over under about "
    "can could will would should may might must also very so such just only more most some any "
    "there their they them he she his her we our you your i my me who whom which what when where why how".split()
)


def _content_terms(text: str) -> set[str]:
    """Lower-cased content words with a light plural stem; decimals stay intact."""
    terms = set()
    for token in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()):
        if token in _STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        terms.add(token)
    return terms


def _term_coverage(claim: str, evidence: str) -> float:
    """Fraction of the claim's content terms that appear in the evidence."""
    claim_terms = _content_terms(claim)
    if not claim_terms:
        return 0.0
    return len(claim_terms & _content_terms(evidence)) / len(claim_terms)


_LIST_CONNECTORS = frozenset({"and", "for", "with", "to", "of", "by", "from", "alongside", "featuring", "ft", "feat", "ally", "friend"})
_LIFE_STATUS_SPLIT = re.compile(
    r"\b(?:is|was|are|were|has|have|had|still|really|actually|died|dies|dead|alive|killed|death|"
    r"dying|deceased|survived)\b",
    re.IGNORECASE,
)
_NAME_TITLES = frozenset({"the", "a", "an", "president", "mr", "mrs", "ms", "dr", "prime", "minister"})


def _life_status_subject(claim: str) -> str:
    """The person a life-status claim is about: up to three words before the life-status word.

    "Trump is dead" -> "trump"; "Dua Lipa dead" -> "dua lipa".
    """
    head = _LIFE_STATUS_SPLIT.split(claim, maxsplit=1)[0]
    tokens = [t for t in re.findall(r"[a-z][a-z0-9'-]*", head.lower()) if t not in _NAME_TITLES]
    return " ".join(tokens[-3:])


def is_directly_relevant(claim: str, evidence: str) -> bool:
    """Reject keyword matches that do not describe the claim's subject."""
    claim_words = set(re.findall(r"[a-z0-9]+", claim.lower()))
    evidence_words = set(re.findall(r"[a-z0-9]+", evidence.lower()))
    if not claim_words or not evidence_words:
        return False

    if _term_coverage(claim, evidence) < _MINIMUM_TERM_COVERAGE:
        return False

    subject_phrase = _life_status_subject(claim)
    predicate_is_life_status = any(
        word in claim_words for word in _LIFE_STATUS_WORDS
    )
    if not subject_phrase or not predicate_is_life_status:
        return True

    subject = r"\s+".join(re.escape(token) for token in subject_phrase.split())
    life_words = "|".join(_LIFE_STATUS_WORDS)
    direct_relation = re.compile(
        rf"(?:\b(?:president|donald)\s+)?\b{subject}\b\s+"
        rf"(?:(?:is|was|remains|appears|has been|has|did|does)\s+)?"
        rf"(?:not\s+)?(?:still\s+)?(?:dead|alive|killed|die|dying|died|dies|survived|deceased)\b"
        rf"|\b(?:death|killing|passing)\s+of\s+"
        rf"(?:\b(?:president|donald)\s+)?\b{subject}\b"
        rf"|\b{subject}\b['’]s\s+(?:death|killing|passing)\b",
        re.IGNORECASE,
    )
    for match in direct_relation.finditer(evidence):
        before = re.findall(r"[a-z]+", evidence[: match.start()].lower())[-1:]
        # "...and Dua Lipa dead": the subject is a trailing list item, so the headline is about someone else.
        if before and before[0] in _LIST_CONNECTORS:
            continue
        return True
    return False


class NLIModel(Protocol):
    def __call__(self, premise: str, hypothesis: str) -> dict[str, Any]:
        """Return model labels and scores for a premise/hypothesis pair."""


DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


class BartNLIModel:
    """Run an MNLI-style classifier (DeBERTa-v3 by default) without assuming label indices."""

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL) -> None:
        from transformers import pipeline

        self._classifier = pipeline("text-classification", model=model_name, device=-1)

    def __call__(self, premise: str, hypothesis: str) -> dict[str, Any]:
        result = self._classifier(
            {"text": premise, "text_pair": hypothesis},
            top_k=None,
            truncation=True,
            max_length=512,
        )
        if result and isinstance(result[0], dict):
            result = [result]
        if not result or not isinstance(result[0], list):
            raise ValueError("NLI model returned an unexpected result.")
        labels = {str(item["label"]): float(item["score"]) for item in result[0]}
        return {"labels": labels}


def verify_claim(
    claim: str,
    evidence: str,
    model: NLIModel,
) -> VerificationSignal:
    """Compare evidence as premise against claim as hypothesis."""
    if not claim.strip():
        raise ValueError("claim must not be empty.")
    if not evidence.strip():
        raise ValueError("evidence must not be empty.")

    raw_result = model(evidence, claim)
    labels = raw_result.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError("NLI model response must contain non-empty labels.")

    normalized = {_normalize_label(label): float(score) for label, score in labels.items()}
    if any(not isinstance(score, (int, float)) or not 0 <= score <= 1 for score in normalized.values()):
        raise ValueError("NLI model returned invalid class scores.")
    label, score = max(normalized.items(), key=lambda item: item[1])
    if label not in {"entailment", "contradiction", "neutral"}:
        raise ValueError(f"Unsupported NLI label: {label}")
    return VerificationSignal(
        label=label,
        score=float(score),
        premise=evidence,
        hypothesis=claim,
        class_scores=normalized,
    )


def aggregate_status(signals: list[VerificationSignal], minimum_score: float = 0.5) -> str:
    """Map multiple NLI signals to a cautious application status."""
    if not signals:
        return "insufficient_evidence"
    confident = [signal for signal in signals if signal.score >= minimum_score]
    if not confident:
        return "insufficient_evidence"
    has_support = any(signal.label == "entailment" for signal in confident)
    has_refutation = any(signal.label == "contradiction" for signal in confident)
    if has_support and has_refutation:
        return "conflicting_evidence"
    if has_support:
        return "supported"
    if has_refutation:
        return "refuted"
    return "insufficient_evidence"


def aggregate_evidence_status(
    assessments: list[EvidenceAssessment],
    *,
    minimum_score: float = 0.5,
    minimum_weighted_score: float = 0.45,
    minimum_similarity: float = 0.3,
    minimum_support_score: float = 0.8,
) -> str:
    """Aggregate NLI signals after accounting for relevance and source quality."""
    if not assessments:
        return "insufficient_evidence"
    if not 0 <= minimum_score <= 1 or not 0 <= minimum_weighted_score <= 1:
        raise ValueError("aggregation thresholds must be between zero and one.")

    confident = [
        assessment
        for assessment in assessments
        if assessment.similarity >= minimum_similarity and _is_usable_assessment(assessment, minimum_score)
    ]
    weighted = [
        (
            assessment,
            assessment.signal.score
            * _source_quality_weight(assessment.source_quality),
        )
        for assessment in confident
    ]
    usable = [
        (assessment, score)
        for assessment, score in weighted
        if score >= minimum_weighted_score
    ]
    if not usable:
        return "insufficient_evidence"
    # Reassurance needs stronger evidence than a warning: weak entailment is not support.
    supporting = [
        (a, s) for a, s in usable if a.signal.label == "entailment" and a.signal.score >= minimum_support_score
    ]
    refuting = [(a, s) for a, s in usable if a.signal.label == "contradiction"]
    support = sum(score for _, score in supporting)
    refutation = sum(score for _, score in refuting)
    if support and refutation:
        # Both sides present: the heavier side wins only if it clearly dominates and is corroborated.
        if refutation >= _DOMINANCE_RATIO * support and _corroborated(refuting):
            return "refuted"
        if support >= _DOMINANCE_RATIO * refutation and _corroborated(supporting):
            return "supported"
        return "conflicting_evidence"
    if support:
        return "supported" if _corroborated(supporting) else "insufficient_evidence"
    if refutation:
        return "refuted" if _corroborated(refuting) else "insufficient_evidence"
    return "insufficient_evidence"


def _corroborated(items: list[tuple["EvidenceAssessment", float]]) -> bool:
    """A verdict needs two independent sources, or one official or reputable source.

    A single page from an unknown site is never enough to support or refute a claim.
    """
    sources = {assessment.source or id(assessment) for assessment, _ in items}
    return len(sources) >= 2 or any(a.source_quality in {"official", "reputable"} for a, _ in items)


# Fact-checking sites restate the claims they examine ("Claim: NASA has confirmed..."), so a passage that merely
# repeats a claim must never count as support. Their debunks still count as strong refutation.
FACT_CHECK_DOMAINS = frozenset(
    {"snopes.com", "factcheck.org", "politifact.com", "fullfact.org", "afp.com", "boomlive.in", "altnews.in"}
)


def _is_fact_check_source(source: str | None) -> bool:
    host = (source or "").lower()
    return any(host == domain or host.endswith("." + domain) for domain in FACT_CHECK_DOMAINS)


_DOMINANCE_RATIO = 2.0  # total weight one side needs, relative to the other, to win a conflict
_DEBUNKING_CUES = re.compile(
    r"\b(?:conspiracy|conspiracies|myth|myths|hoax|hoaxes|debunk(?:ed|ing)?|disproven|disproved|"
    r"misinformation|disinformation|baseless|unfounded|pseudoscien\w*|falsely|false claims?|"
    r"no evidence|no scientific evidence|misconception)\b",
    re.IGNORECASE,
)
_ATTRIBUTION_CUES = re.compile(
    r"\b(?:fake news|claimed|claims? that|rumou?rs?|viral|circulating|circling the web|purported|"
    r"alleged(?:ly)?|the article continues|according to (?:a |the )?(?:post|rumou?r|message))\b",
    re.IGNORECASE,
)
_QUOTED_TEXT = re.compile(r'["\u201c]\s*([^"\u201d]{20,})(?:["\u201d]|$)')


def _is_quoted_or_attributed(claim: str, premise: str) -> bool:
    """True if the passage reports the claim (quotes it, or calls it a rumour) rather than asserting it."""
    if _DEBUNKING_CUES.search(premise) or _ATTRIBUTION_CUES.search(premise):
        return True
    return any(_term_coverage(claim, quoted) >= _MINIMUM_TERM_COVERAGE for quoted in _QUOTED_TEXT.findall(premise))


_EXPLICIT_NEGATION = re.compile(r"\b(?:not|n't|false|incorrect|no evidence)\b", re.IGNORECASE)
_STRONG_CONTRADICTION_SCORE = 0.85
_STRONG_CONTRADICTION_SIMILARITY = 0.6


def _is_usable_assessment(
    assessment: EvidenceAssessment,
    minimum_score: float,
) -> bool:
    """Decide whether one NLI result may count as support or refutation."""
    if assessment.signal.score < minimum_score or assessment.similarity < 0:
        return False
    premise = assessment.signal.premise
    if assessment.signal.label == "entailment":
        # A passage that mentions the claim in order to debunk it (a conspiracy
        # theory, a myth) restates the claim without endorsing it, which NLI
        # models read as entailment. It must never count as support.
        if _is_fact_check_source(assessment.source):
            return False
        return not _is_quoted_or_attributed(assessment.signal.hypothesis, premise)
    if assessment.signal.label != "contradiction":
        return True
    if _DEBUNKING_CUES.search(premise) or _ATTRIBUTION_CUES.search(premise) or _EXPLICIT_NEGATION.search(premise):
        return True
    # The relevance gate already screens off-topic passages, so a confident
    # contradiction on a closely matching passage is trusted.
    return (
        assessment.signal.score >= _STRONG_CONTRADICTION_SCORE
        and assessment.similarity >= _STRONG_CONTRADICTION_SIMILARITY
    )


def _source_quality_weight(source_quality: str) -> float:
    return {
        "official": 1.0,
        "reputable": 0.95,
        "unknown": 0.8,
        "context_only": 0.4,  # context, never evidence: stays below the 0.45 cutoff
    }.get(source_quality, 0.8)


def _normalize_label(label: str) -> str:
    normalized = label.lower().replace("_", " ").replace("-", " ").strip()
    aliases = {
        "label 0": "contradiction",
        "label 1": "neutral",
        "label 2": "entailment",
    }
    return aliases.get(normalized, normalized)
