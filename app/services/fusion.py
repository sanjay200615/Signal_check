"""Decision fusion for the first manual-case demonstration."""

from dataclasses import dataclass, field


DECEPTIVE_THRESHOLD = 0.6
REVIEW_THRESHOLD = 0.3


@dataclass(frozen=True)
class FusionResult:
    verdict: str
    risk_score: float
    contributions: dict[str, float] = field(default_factory=dict)


def fuse_signals(
    *,
    claim_contradiction: float = 0.0,
    phishing: float = 0.0,
    coordination: float = 0.0,
    image_anomaly: float = 0.0,
    cross_channel: float = 0.0,
    has_evidence: bool = False,
    trusted_domain: bool = False,
) -> FusionResult:
    values = (claim_contradiction, phishing, coordination, image_anomaly, cross_channel)
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("fusion signals must be between zero and one.")
    contributions = {
        "claim_contradiction": 0.7 * claim_contradiction,
        "phishing_and_scam_language": 1.0 * phishing,
        "cross_input_coordination": 0.2 * coordination,
        "image": 0.5 * image_anomaly,
        "cross_channel": 1.0 * cross_channel,
    }
    risk = min(1.0, sum(contributions.values()))
    if risk >= DECEPTIVE_THRESHOLD:
        verdict = "deceptive"
    elif risk >= REVIEW_THRESHOLD:
        # Meaningful risk signals that fall short of a deceptive call must never
        # read as reassurance, whatever evidence was found.
        verdict = "needs_review"
    elif not (has_evidence or trusted_domain):
        verdict = "needs_review"
    else:
        verdict = "genuine"
    return FusionResult(verdict, risk, {k: round(v, 4) for k, v in contributions.items() if v})
