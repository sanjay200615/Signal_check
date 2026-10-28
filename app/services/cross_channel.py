"""Cross-channel reasoning: do the different inputs in one case tell a consistent story?

Scam campaigns reach people over several channels at once (a text, a call, a link, a screenshot).
Each channel alone may look only mildly suspicious; the inconsistencies and the repeated pressure
between channels are the extra signal. Everything here can only RAISE risk. Agreement between
channels is never evidence of legitimacy, since a campaign is consistent by design.
"""

from dataclasses import dataclass
import itertools
import re
from typing import Any

from app.services.phishing import assess_message, named_brands
from app.services.reputation import Reputation
from app.services.verifier import _term_coverage, verify_claim

MAX_CROSS_CHANNEL_SCORE = 0.6
NLI_CONTRADICTION_THRESHOLD = 0.85
MAX_NLI_PAIRS = 12

# Message cues that mean "pressure or an unusual demand", as opposed to topic words.
PRESSURE_INDICATORS = frozenset(
    {
        "urgent_message_language", "sensitive_request_on_page", "credential_update_call_to_action",
        "financial_bait", "advance_fee_request", "untraceable_payment", "remote_access_request", "otp_request",
        "authority_impersonation", "secrecy_or_pressure", "family_emergency_scam", "crypto_doubling",
        "clickbait_lure", "unsolicited_alert",
    }
)
_TYPE_LABEL = {"text": "the text message", "audio": "the voice note", "image": "the screenshot", "url": "the link"}

_PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{6,17}\d(?!\w|\.\d)")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_AMOUNT = re.compile(
    r"(?:[$£€₹]\s?(\d[\d,]*(?:\.\d+)?))|(?:(\d[\d,]*(?:\.\d+)?)\s?(?:dollars?|pounds?|euros?|rupees?))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChannelInput:
    input_id: str
    input_type: str
    text: str = ""


@dataclass(frozen=True)
class Finding:
    indicator: str
    weight: float
    detail: str
    input_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Facts:
    input_type: str
    brands: frozenset[str]
    phones: frozenset[str]
    emails: frozenset[str]
    amounts: frozenset[str]
    pressure: frozenset[str]


def _facts(item: ChannelInput, reputation: Reputation) -> _Facts:
    text = item.text
    phones = set()
    for match in _PHONE.findall(text):
        digits = re.sub(r"\D", "", match)
        if 8 <= len(digits) <= 15 and not _DATE.search(match):
            phones.add(digits)
    amounts = {format(float((m.group(1) or m.group(2)).replace(",", "")), "g") for m in _AMOUNT.finditer(text)}
    return _Facts(
        input_type=item.input_type,
        brands=frozenset(named_brands(text, reputation)),
        phones=frozenset(phones),
        emails=frozenset(_EMAIL.findall(text.lower())),
        amounts=frozenset(amounts),
        pressure=frozenset(set(assess_message(text).indicators) & PRESSURE_INDICATORS),
    )


def _related_brands(a: frozenset[str], b: frozenset[str], reputation: Reputation) -> bool:
    """Brands are related if they overlap or share an official domain (Microsoft and Outlook)."""
    if a & b:
        return True
    owned_a = {d for brand in a for d in reputation.brands.get(brand, ())}
    owned_b = {d for brand in b for d in reputation.brands.get(brand, ())}
    return bool(owned_a & owned_b)


def _label(input_id: str, facts: dict[str, _Facts]) -> str:
    return _TYPE_LABEL.get(facts[input_id].input_type, "an input")


def analyze_cross_channel(
    inputs: list[ChannelInput],
    reputation: Reputation,
    *,
    claims: list[Any] = (),
    nli: Any | None = None,
) -> list[Finding]:
    """Compare the inputs of a case with each other and return the inconsistencies found."""
    facts = {item.input_id: _facts(item, reputation) for item in inputs if item.text.strip()}
    findings: list[Finding] = []

    for (id_a, fa), (id_b, fb) in itertools.combinations([(i, f) for i, f in facts.items() if f.brands], 2):
        if not _related_brands(fa.brands, fb.brands, reputation):
            findings.append(
                Finding(
                    "sender_brand_inconsistency", 0.4,
                    f"{_label(id_a, facts).capitalize()} names {', '.join(sorted(fa.brands))} "
                    f"but {_label(id_b, facts)} names {', '.join(sorted(fb.brands))}.",
                    (id_a, id_b),
                )
            )
            break

    for kind, indicator, noun in (("phones", "contact_number_mismatch", "phone numbers"), ("emails", "contact_email_mismatch", "email addresses")):
        holders = [(i, getattr(f, kind)) for i, f in facts.items() if getattr(f, kind)]
        if len(holders) >= 2 and not frozenset.intersection(*(values for _, values in holders)):
            findings.append(
                Finding(indicator, 0.25, f"The inputs give different {noun}, so there is no single contact to verify.",
                        tuple(i for i, _ in holders))
            )

    amounts = [(i, f.amounts) for i, f in facts.items() if f.amounts]
    if len(amounts) >= 2 and not frozenset.intersection(*(values for _, values in amounts)):
        findings.append(
            Finding("amount_mismatch", 0.2, "The inputs mention different amounts of money.", tuple(i for i, _ in amounts))
        )

    pressured = [(i, f) for i, f in facts.items() if f.pressure]
    if len({f.input_type for _, f in pressured}) >= 2:
        where = " and ".join(sorted({_TYPE_LABEL.get(f.input_type, "an input") for _, f in pressured}))
        findings.append(
            Finding("multichannel_pressure", 0.3,
                    f"Pressure or an unusual demand appears in several channels at once ({where}), a common campaign pattern.",
                    tuple(i for i, _ in pressured))
        )

    contradiction = _cross_input_contradiction(claims, nli)
    if contradiction:
        findings.append(contradiction)
    return findings


def _cross_input_contradiction(claims: list[Any], nli: Any | None) -> Finding | None:
    """Use NLI to find two inputs asserting things that cannot both be true."""
    if nli is None:
        return None
    pairs = []
    for left, right in itertools.combinations(claims, 2):
        if set(left.source_input_ids) & set(right.source_input_ids):
            continue  # same input
        if _term_coverage(left.claim_text, right.claim_text) >= 0.3:
            pairs.append((left, right))
    for left, right in pairs[:MAX_NLI_PAIRS]:
        signal = verify_claim(right.claim_text, left.claim_text, nli)
        if signal.label == "contradiction" and signal.score >= NLI_CONTRADICTION_THRESHOLD:
            return Finding(
                "cross_input_contradiction", 0.3,
                f"Two inputs contradict each other: “{left.claim_text}” versus “{right.claim_text}”.",
                (*left.source_input_ids, *right.source_input_ids),
            )
    return None
