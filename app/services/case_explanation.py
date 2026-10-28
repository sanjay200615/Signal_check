"""Deterministic, evidence-linked explanation of a case result.

Every line is built from structured signals, claims and evidence. No language model is involved,
so the explanation cannot say anything the analysis did not find.
"""

from typing import Any

INDICATOR_TEXT = {
    "known_phishing_url": "The link is listed in a public phishing feed.",
    "known_phishing_host": "The site is listed in a public phishing feed.",
    "brand_in_subdomain": "The web address puts a well-known brand name in front of a domain that brand does not own.",
    "brand_in_domain_label": "The domain name contains a well-known brand but is not owned by it.",
    "homoglyph_lookalike": "The domain imitates a popular site using look-alike characters.",
    "typosquat_lookalike": "The domain is one typo away from a popular site.",
    "same_label_other_tld": "The domain copies a popular site's name under a different ending.",
    "free_hosting_platform": "The page is on a free hosting platform that anyone can publish to.",
    "random_subdomain_label": "The address contains a machine-generated random string.",
    "brand_on_free_hosting": "A well-known brand name is used on a free hosting platform.",
    "brand_domain_in_path": "The address spells out another brand's domain in its path.",
    "newly_registered_domain": "The domain was registered within the last month.",
    "recently_registered_domain": "The domain was registered within the last three months.",
    "suspicious_top_level_domain": "The domain ends in an extension often used for throwaway sites.",
    "unencrypted_http": "The link does not use an encrypted connection.",
    "ip_address_host": "The link points to a raw IP address instead of a named site.",
    "many_subdomains": "The address has an unusually long chain of subdomains.",
    "embedded_credentials": "The address embeds a username in the link, a common trick to hide the real site.",
    "punycode_domain": "The domain uses look-alike international characters.",
    "short_link_expanded": "The link was a shortened URL that hid its destination.",
    "message_brand_mismatch": "The message names a brand, but the link goes somewhere that brand does not own.",
    "image_brand_domain_mismatch": "The screenshot shows a brand, but the link goes somewhere that brand does not own.",
    "screenshot_address_differs_from_link": "The address shown in the screenshot differs from the link provided.",
    "sensitive_request_on_page": "It asks for a PIN, password, code or account verification.",
    "urgent_message_language": "It uses urgency or threats to pressure the reader.",
    "credential_update_call_to_action": "It tells the reader to click through and update credentials.",
    "financial_bait": "It offers a prize, reward or pre-approved money.",
    "advance_fee_request": "It asks for an upfront fee.",
    "crypto_doubling": "It promises to multiply cryptocurrency.",
    "authority_impersonation": "It claims to be an authority and threatens legal consequences.",
    "secrecy_or_pressure": "It demands secrecy or tells the person not to hang up.",
    "untraceable_payment": "It asks for payment by gift card, wire transfer, prepaid card or crypto.",
    "remote_access_request": "It asks for remote access to a device.",
    "otp_request": "It asks the person to read out a one-time code.",
    "unsolicited_alert": "It claims to have detected a problem nobody asked about.",
    "family_emergency_scam": "It matches the 'family member in trouble needs money' pattern.",
    "clickbait_lure": "It pairs a sensational event with a 'watch the video' lure.",
    "sender_brand_inconsistency": "Different inputs name different senders.",
    "contact_number_mismatch": "The inputs give different phone numbers.",
    "contact_email_mismatch": "The inputs give different email addresses.",
    "amount_mismatch": "The inputs mention different amounts of money.",
    "multichannel_pressure": "Pressure appears in several channels at once.",
    "cross_input_contradiction": "Two inputs contradict each other.",
}


def build_explanation(
    *,
    signals: list[Any],
    claims: list[Any],
    evidence: list[Any],
    cross_findings: list[Any],
    trusted_domain: bool,
    ai_escalated: bool,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Plain-language reasons, risks first, each linked to the inputs or evidence behind it."""
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(kind: str, text: str, refs: list[str] | None = None) -> None:
        if text not in seen:
            seen.add(text)
            items.append({"kind": kind, "text": text, "refs": refs or []})

    cross_indicators = {finding.indicator for finding in cross_findings}
    for signal in signals:
        for indicator in signal.details.get("indicators", []) or []:
            if indicator not in cross_indicators and indicator in INDICATOR_TEXT:
                add("risk", INDICATOR_TEXT[indicator], [signal.input_id])
    for finding in cross_findings:
        add("cross_channel", finding.detail, list(finding.input_ids))

    for claim in claims:
        related = [e for e in evidence if e.claim_id == claim.claim_id]
        label = f"“{claim.claim_text[:90]}”"
        if claim.verification_status == "refuted":
            sources = sorted({e.domain or e.source_name for e in related if e.nli_label == "contradiction"})[:3]
            add("claim", f"The claim {label} is contradicted by {', '.join(sources) or 'retrieved evidence'}.",
                [e.evidence_id for e in related if e.nli_label == "contradiction"][:3])
        elif claim.verification_status == "supported":
            sources = sorted({e.domain or e.source_name for e in related if e.nli_label == "entailment"})[:3]
            add("reassurance", f"The claim {label} is supported by {', '.join(sources) or 'retrieved evidence'}.",
                [e.evidence_id for e in related if e.nli_label == "entailment"][:3])
        elif claim.verification_status == "conflicting_evidence":
            add("claim", f"Sources disagree about the claim {label}.")
        elif claim.verification_status == "insufficient_evidence":
            add("info", f"No reliable source confirms or contradicts the claim {label}.")

    if trusted_domain:
        add("reassurance", "The link goes to a long-established domain, and no risk indicators were found.")
    if ai_escalated:
        add("risk", "The AI investigator judged this likely deceptive, so the verdict was raised to needs review as a precaution.")
    order = {"risk": 0, "cross_channel": 1, "claim": 2, "reassurance": 3, "info": 4}
    return sorted(items, key=lambda item: order[item["kind"]])[:limit]
