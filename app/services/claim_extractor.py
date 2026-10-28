"""Transparent baseline for extracting claims and entities from text."""

from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class ExtractedClaim:
    claim_id: str
    claim_text: str
    entities: list[str]
    claim_type: str


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd",
    "co", "no", "e.g", "i.e", "u.s", "u.k", "approx", "dept",
}
_ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,}(?:\s+[A-Z]{2,})*)\b"
)
_URL_PATTERN = re.compile(r"\b(?:https?://|www\.)[^\s]+", re.IGNORECASE)
_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:,\s*\d{4})?)\b"
)
_NUMBER_PATTERN = re.compile(r"\b(?:\d+(?:[.,]\d+)?%?|[$€£₹]\s?\d+(?:[.,]\d+)?)\b")
_PERSONAL_PATTERN = re.compile(
    r"\b(?:i|i'm|i've|i'll|i'd|me|my|mine|we|we're|we've|our|ours|you|you're|you've|you'll|your|yours)\b",
    re.IGNORECASE,
)
_IMPERATIVE_START = re.compile(
    r"^(?:(?:please|kindly)\s+)?(?:click|tap|see|check|visit|send|pay|verify|update|reply|call|"
    r"enter|confirm|log\s?in|sign\s?in|open|download|follow|contact|share|forward|read|use|try|"
    r"claim|act|save|hurry|do not|don't|never|look|watch|listen|remember|note|ignore|ensure|make|hey|hi|hello)\b",
    re.IGNORECASE,
)
_MINIMUM_CLAIM_WORDS = 3
_COORDINATION_START = re.compile(
    r"^(?:(?:hi|hey|hello)[,\s]+)?(?:(?:quick|short) note[,:\s]+)?(?:(?:just|quick|friendly)\s+)?(?:confirming|reminding|checking in|following up|"
    r"touching base|writing to|calling to|calling about|letting you know|reminder\b|wanted to|wondering|"
    r"see you|talk soon|on my way|running late)",
    re.IGNORECASE,
)
_DEICTIC_TIME = re.compile(
    r"\b(?:today|tonight|tomorrow|yesterday|last night|right now|this (?:morning|afternoon|evening|week|weekend|month)"
    r"|next (?:week|month|year))\b",
    re.IGNORECASE,
)
_LURE_TAIL = re.compile(
    r"[\s,;:-]*\b(?:watch|see|click|view|check out)\s+(?:the\s+|this\s+|full\s+|leaked\s+|new\s+)*"
    r"(?:video|videos|here|link|now|pics?|photos?|footage)\b.*$",
    re.IGNORECASE,
)
_CONDITIONAL_START =re.compile(r"^(?:if|unless|in case)\b", re.IGNORECASE)
_NON_FACTUAL_PREFIXES = (
    "how ",
    "why ",
    "when ",
    "where ",
    "who ",
    "can you ",
    "please ",
    "i think ",
    "i feel ",
    "in my opinion ",
)


def extract_claims(text: str) -> list[ExtractedClaim]:
    """Extract non-empty candidate claims without verifying them."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string.")

    urls: list[str] = []

    def replace_url(match: re.Match[str]) -> str:
        urls.append(match.group(0))
        return f"URLTOKEN{len(urls) - 1}"

    protected_text = _URL_PATTERN.sub(replace_url, text)
    claims: list[ExtractedClaim] = []
    seen_claims: set[str] = set()
    for sentence in _split_sentences(protected_text):
        normalized = " ".join(sentence.split()).strip(" -")
        if not normalized:
            continue
        for index, url in enumerate(urls):
            normalized = normalized.replace(f"URLTOKEN{index}", url)
        claim_text = normalized.rstrip(".!?")
        without_lure = _LURE_TAIL.sub("", claim_text).strip()
        if len(without_lure.split()) >= _MINIMUM_CLAIM_WORDS:
            claim_text = without_lure  # "X is dead, watch video" -> "X is dead"
        if len(re.sub(r"[^A-Za-z0-9]+", "", claim_text)) < 3:
            continue
        claim_key = claim_text.casefold()
        if claim_key in seen_claims:
            continue
        seen_claims.add(claim_key)
        claims.append(
            ExtractedClaim(
                claim_id=f"claim_{len(claims) + 1}",
                claim_text=claim_text,
                entities=_extract_entities(normalized),
                claim_type=_classify_claim(normalized),
            )
        )
    if not claims:
        raise ValueError("No candidate claims could be extracted.")
    return claims


def claims_to_dicts(claims: list[ExtractedClaim]) -> list[dict[str, object]]:
    """Convert validated extractor results to JSON-compatible dictionaries."""
    return [asdict(claim) for claim in claims]


def _extract_entities(sentence: str) -> list[str]:
    candidates = _ENTITY_PATTERN.findall(sentence)
    candidates.extend(_URL_PATTERN.findall(sentence))
    candidates.extend(_DATE_PATTERN.findall(sentence))
    candidates.extend(_NUMBER_PATTERN.findall(sentence))
    entities: list[str] = []
    for candidate in candidates:
        clean_candidate = candidate.rstrip(".,!?;:)")
        clean_candidate = re.sub(r"^(?:The|A|An)\s+", "", clean_candidate)
        if clean_candidate and clean_candidate not in entities:
            entities.append(clean_candidate)
    return entities


def _has_named_entity(sentence: str) -> bool:
    """A capitalised word that is not just the first word of the sentence (numbers do not identify anything)."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)
    return any(word[0].isupper() for word in words[1:])


def _classify_claim(sentence: str) -> str:
    """Label a sentence ``factual`` only if it states something checkable about the world.

    Personal or transactional statements ("your order has shipped", "I changed my
    number"), instructions ("click here") and conditionals ("if this was you...")
    cannot be confirmed or contradicted by public sources. Searching the web for
    them only finds unrelated pages that can look like contradictions.
    """
    lowered = sentence.lower().strip()
    if lowered.startswith(_NON_FACTUAL_PREFIXES) or sentence.rstrip().endswith("?"):
        return "non_factual"
    if len(re.findall(r"[A-Za-z0-9']+", sentence)) < _MINIMUM_CLAIM_WORDS:
        return "non_factual"  # interjections such as "Thanks" or "Congratulations"
    if _COORDINATION_START.match(sentence.strip()):
        return "non_factual"  # arranging or reminding is not a statement about the world
    if _DEICTIC_TIME.search(sentence) and not _has_named_entity(sentence):
        return "non_factual"  # a local, time-relative event with no named entity cannot be looked up
    if (
        _PERSONAL_PATTERN.search(sentence)
        or _IMPERATIVE_START.match(sentence)
        or _CONDITIONAL_START.match(sentence)
    ):
        return "non_factual"
    return "factual"


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation followed by whitespace.

    Requiring whitespace keeps decimals ("3.13", "$2.99") and dotted tokens
    intact, and known abbreviations ("Dr.", "e.g.") do not end a sentence.
    """
    sentences: list[str] = []
    for piece in _SENTENCE_BOUNDARY.split(text):
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def _ends_with_abbreviation(sentence: str) -> bool:
    match = re.search(r"([A-Za-z.]+)\.$", sentence.strip())
    return bool(match) and match.group(1).lower() in _ABBREVIATIONS
