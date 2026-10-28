import pytest

from app.services.fusion import fuse_signals


def test_high_risk_is_deceptive():
    assert fuse_signals(phishing=0.8, has_evidence=True).verdict == "deceptive"


def test_moderate_risk_is_never_genuine_even_with_evidence():
    result = fuse_signals(phishing=0.44, has_evidence=True)
    assert result.verdict == "needs_review"
    assert result.risk_score == pytest.approx(0.44)


def test_low_risk_with_evidence_is_genuine():
    assert fuse_signals(phishing=0.1, has_evidence=True).verdict == "genuine"


def test_low_risk_without_evidence_abstains():
    assert fuse_signals(phishing=0.0, has_evidence=False).verdict == "needs_review"


def test_out_of_range_signal_is_rejected():
    with pytest.raises(ValueError):
        fuse_signals(phishing=1.5)


def test_trusted_domain_lets_low_risk_content_be_genuine_without_claim_evidence():
    assert fuse_signals(phishing=0.0, has_evidence=False, trusted_domain=True).verdict == "genuine"


def test_trusted_domain_never_overrides_risk():
    assert fuse_signals(phishing=0.45, trusted_domain=True).verdict == "needs_review"
    assert fuse_signals(phishing=0.7, trusted_domain=True).verdict == "deceptive"
