"""URL and message phishing-risk signals.

Scores are additive indicator weights capped at 1.0. They are prototype signal
strengths, not calibrated probabilities. ``assess_url`` is a pure function: any
network-derived input (such as domain age) is supplied by the caller.
"""

from dataclasses import dataclass
import re
from urllib.parse import urlsplit

from app.services.domain_analysis import analyze_domain, is_free_hosting, is_shortener
from app.services.reputation import BRAND_DOMAINS, Reputation, registered_domain

_SUSPICIOUS_TLDS = {".zip", ".mov", ".click", ".top", ".gq", ".tk", ".cf", ".ml", ".ga", ".xyz", ".icu", ".buzz"}
_SENSITIVE_URL_WORDS = {"login", "signin", "verify", "password", "credential", "payment", "account", "pin"}
_TRUSTED_MAX_RANK = 10_000

_CREDENTIAL_REQUEST = re.compile(
    r"\b(?:enter|provide|share|send|confirm|verify|update|submit|reply with|give|disclose)\b[^.!?]{0,40}"
    r"\b(?:pin|otp|password|passcode|cvv|card number|account number|bank details|login details|credentials|kyc|ssn)\b"
    r"|\b(?:verify|update|confirm)\b[^.!?]{0,20}\b(?:account|kyc|identity|bank details)\b",
    re.IGNORECASE,
)
_URGENCY_OR_THREAT = re.compile(
    r"\b(?:urgent(?:ly)?|immediately|suspend(?:ed)?|blocked|locked|disconnect(?:ed)?|terminated|"
    r"expire[sd]?|expiry|overdue|final notice|act now|tonight|legal action|account closure|"
    r"will be closed|within \d+ (?:hours?|minutes?))\b",
    re.IGNORECASE,
)
_FINANCIAL_BAIT = re.compile(
    r"\b(?:you(?:'ve| have)? won|winner|prize|lottery|gift card|giveaway|congratulations|"
    r"pre-approved|cashback|claim your)\b",
    re.IGNORECASE,
)
_ADVANCE_FEE = re.compile(
    r"\b(?:registration (?:fee|payment)|processing fee|customs fee|re-?delivery fee|clearance fee|"
    r"small fee|refundable|release (?:the )?(?:parcel|delivery|funds)|pay (?:a )?(?:small )?(?:fee|amount))\b",
    re.IGNORECASE,
)
_CRYPTO_ASSET = re.compile(r"\b(?:btc|bitcoin|eth|ethereum|usdt|crypto)\b", re.IGNORECASE)
_CRYPTO_BAIT = re.compile(r"\b(?:double|send|receive|giveaway|invest|returns?)\b", re.IGNORECASE)
_SENSATIONAL = re.compile(
    r"\b(?:dead|died|dies|death|killed|murdered|arrested|shot|stabbed|scandal|leaked|shocking|exposed|"
    r"nude|accident|crash)\b",
    re.IGNORECASE,
)
_LURE_CALL = re.compile(
    r"\b(?:watch|see|click|view)\b[^.!?]{0,25}\b(?:video|videos|link|here|pics?|photos?|footage|now)\b"
    r"|\b(?:full|leaked) video\b",
    re.IGNORECASE,
)
_AUTHORITY = re.compile(
    r"\b(?:this is|i am|i'm|calling from|we are|speaking from)\b[^.!?]{0,40}\b(?:irs|internal revenue|"
    r"social security|police|fbi|sheriff|customs|tax (?:department|office)|your bank|microsoft|amazon|apple|"
    r"visa|mastercard|medicare|immigration)\b",
    re.IGNORECASE,
)
_LEGAL_THREAT = re.compile(
    r"\b(?:arrest(?:ed)?|warrant|legal action|lawsuit|frozen|suspended|deported|prosecuted|police will)\b",
    re.IGNORECASE,
)
_SECRECY = re.compile(
    r"\b(?:do not hang up|don't hang up|do not tell anyone|don't tell anyone|"
    r"keep this (?:confidential|secret)|do not disconnect)\b",
    re.IGNORECASE,
)
_UNTRACEABLE_PAYMENT = re.compile(
    r"\b(?:gift cards?|wire transfer|western union|moneygram|bitcoin atm|crypto atm|prepaid cards?|"
    r"read (?:me )?the (?:numbers|codes?))\b",
    re.IGNORECASE,
)
_REMOTE_ACCESS = re.compile(
    r"\b(?:remote access|anydesk|teamviewer|your computer (?:has|is infected)|virus on your|technical support)\b",
    re.IGNORECASE,
)
_OTP_READOUT = re.compile(
    r"\b(?:tell|read|say|repeat)\b(?: me| us| out| back)*\b[^.!?]{0,30}"
    r"\b(?:one[- ]time (?:password|passcode|code|pin)|otp|verification code|security code|pin)\b",
    re.IGNORECASE,
)
_UNSOLICITED_ALERT = re.compile(
    r"\b(?:we|our (?:system|team|engineers?))\b[^.!?]{0,20}\b(?:detected|noticed|found)\b[^.!?]{0,40}"
    r"\b(?:virus|malware|suspicious|unauthori[sz]ed|hacked|compromised)\b"
    r"|\bsomeone (?:has )?tried to (?:withdraw|access|log in|transfer|hack)\b",
    re.IGNORECASE,
)
_FAMILY_CONTACT = re.compile(r"\b(?:mom|mum|dad|grandma|grandpa)\b", re.IGNORECASE)
_FAMILY_TROUBLE = re.compile(
    r"\b(?:new number|phone (?:broke|is broken|died)|lost my phone|in trouble|bail|accident)\b", re.IGNORECASE
)
_MONEY_ASK = re.compile(r"\b(?:send|transfer|pay|wire|lend)\b[^.!?]{0,30}(?:[$\u00a3\u20ac\u20b9]|\d|money|cash|pounds?|dollars?|euros?|rupees?|bucks|grand)", re.IGNORECASE)
_CLICK_PROMPT = re.compile(r"\b(?:click|tap|follow the link|open the link)\b", re.IGNORECASE)


@dataclass(frozen=True)
class MessageAssessment:
    risk_score: float
    indicators: tuple[str, ...]


@dataclass(frozen=True)
class PhishingAssessment:
    status: str
    risk_score: float
    indicators: tuple[str, ...]
    trusted_domain: bool = False


def assess_message(text: str) -> MessageAssessment:
    """Score scam language. Needs request/threat phrasing, not just topic words."""
    indicators: list[str] = []
    score = 0.0
    credential = bool(_CREDENTIAL_REQUEST.search(text))
    if credential:
        indicators.append("sensitive_request_on_page")
        score += 0.3
    if _URGENCY_OR_THREAT.search(text):
        indicators.append("urgent_message_language")
        score += 0.2
    if _FINANCIAL_BAIT.search(text):
        indicators.append("financial_bait")
        score += 0.25
    if _ADVANCE_FEE.search(text):
        indicators.append("advance_fee_request")
        score += 0.3
    if _CRYPTO_ASSET.search(text) and _CRYPTO_BAIT.search(text):
        indicators.append("crypto_doubling")
        score += 0.4
    if _AUTHORITY.search(text) and _LEGAL_THREAT.search(text):
        indicators.append("authority_impersonation")
        score += 0.4
    if _OTP_READOUT.search(text):
        indicators.append("otp_request")
        score += 0.5
    if _UNSOLICITED_ALERT.search(text):
        indicators.append("unsolicited_alert")
        score += 0.3
    if _SECRECY.search(text):
        indicators.append("secrecy_or_pressure")
        score += 0.25
    if _UNTRACEABLE_PAYMENT.search(text):
        indicators.append("untraceable_payment")
        score += 0.5
    if _REMOTE_ACCESS.search(text):
        indicators.append("remote_access_request")
        score += 0.5
    if _FAMILY_CONTACT.search(text) and _FAMILY_TROUBLE.search(text) and _MONEY_ASK.search(text):
        indicators.append("family_emergency_scam")
        score += 0.35
    if _SENSATIONAL.search(text) and _LURE_CALL.search(text):
        indicators.append("clickbait_lure")
        score += 0.35
    if credential and _CLICK_PROMPT.search(text):
        indicators.append("credential_update_call_to_action")
        score += 0.15
    return MessageAssessment(min(1.0, score), tuple(indicators))


def assess_url(
    url: str,
    page_text: str = "",
    *,
    reputation: Reputation | None = None,
    domain_age_days: int | None = None,
) -> PhishingAssessment:
    """Return deterministic risk signals for a URL and the message that carries it.

    ``reputation`` adds feed, popularity and lookalike checks; ``domain_age_days``
    (from RDAP, looked up by the caller) adds a newly-registered-domain signal.
    """
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url must be an absolute http or https URL.")

    indicators: list[str] = []
    score = 0.0
    hostname = parsed.hostname.lower()
    lowered_url = url.lower()
    effective_reputation = reputation if reputation is not None else Reputation()

    if parsed.scheme != "https":
        indicators.append("unencrypted_http")
        score += 0.15
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", hostname):
        indicators.append("ip_address_host")
        score += 0.3
    if hostname.count(".") >= 3:
        indicators.append("many_subdomains")
        score += 0.15
    if any(hostname.endswith(tld) for tld in _SUSPICIOUS_TLDS):
        indicators.append("suspicious_top_level_domain")
        score += 0.2
    if "@" in parsed.netloc:
        indicators.append("embedded_credentials")
        score += 0.25
    if any(word in lowered_url for word in _SENSITIVE_URL_WORDS):
        indicators.append("sensitive_action_in_url")
        score += 0.15

    for finding in analyze_domain(hostname, effective_reputation):
        indicators.append(finding.indicator)
        score += finding.weight
    path_brand = _brand_domain_in_path(parsed.path, registered_domain(hostname), effective_reputation)
    if path_brand:
        indicators.append("brand_domain_in_path")
        score += 0.4
    if reputation is not None and reputation.in_feed(url):
        indicators.append("known_phishing_url")
        score += 1.0
    if domain_age_days is not None:
        if domain_age_days <= 30:
            indicators.append("newly_registered_domain")
            score += 0.5
        elif domain_age_days <= 90:
            indicators.append("recently_registered_domain")
            score += 0.25

    domain = registered_domain(hostname)
    message = assess_message(page_text)
    indicators.extend(message.indicators)
    score += message.risk_score
    brand_mismatch = _message_names_other_brand(page_text, domain, effective_reputation)
    if brand_mismatch:
        indicators.append("message_brand_mismatch")
        score += 0.4

    risk_score = min(1.0, score)
    status = "likely_phishing" if risk_score >= 0.6 else ("suspicious" if risk_score >= 0.3 else "likely_safe")
    trusted = (
        reputation is not None
        and risk_score < 0.3
        and not brand_mismatch
        and not is_free_hosting(hostname)
        and not is_shortener(hostname)  # a short link hides where it leads
        and reputation.is_trusted(domain, max_rank=_TRUSTED_MAX_RANK)
    )
    return PhishingAssessment(status, risk_score, tuple(dict.fromkeys(indicators)), trusted)


def _brand_domain_in_path(path: str, domain: str, reputation: Reputation) -> bool:
    """True if the URL path spells out a brand's domain that is not the page's own domain."""
    lowered = path.lower()
    return any(
        official in lowered and official != domain
        for domains in reputation.brands.values()
        for official in domains
        if "." in official and len(official) >= 7
    )


_CONFUSABLE = str.maketrans({"i": "l", "1": "l", "|": "l", "0": "o"})


def _confusable_key(token: str) -> str:
    """Fold characters that OCR and lookalike text commonly swap (I/l/1, O/0)."""
    return token.lower().translate(_CONFUSABLE)


# Brand names that are also everyday words: only count them when written capitalised.
_AMBIGUOUS_BRANDS = frozenset({"apple", "steam", "zoom", "chase", "ups", "axis", "dana", "outlook", "irs", "amazon"})


def named_brands(text: str, reputation: Reputation) -> list[str]:
    """Brands whose keyword appears as a whole word in ``text``, tolerant of I/l/1 and O/0 confusions."""
    tokens = re.findall(r"[A-Za-z0-9|]+", text)
    found = []
    for brand in reputation.brands:
        key = _confusable_key(brand)
        for token in tokens:
            if _confusable_key(token) != key:
                continue
            if brand in _AMBIGUOUS_BRANDS and not token[0].isupper():
                continue
            found.append(brand)
            break
    return found


def _message_names_other_brand(text: str, domain: str, reputation: Reputation) -> bool:
    """True if the message names a known brand but links to a domain that brand does not own."""
    if not text or not domain:
        return False
    named = [reputation.brands[brand] for brand in named_brands(text, reputation)]
    return bool(named) and not any(domain in domains for domains in named)
