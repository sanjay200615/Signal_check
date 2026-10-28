import pytest

from app.services.phishing import assess_url


def test_phishing_assessment_detects_suspicious_url_and_page() -> None:
    assessment = assess_url(
        "http://192.168.1.1/login",
        "Urgent verify your password immediately.",
    )

    assert assessment.status == "likely_phishing"
    assert assessment.risk_score >= 0.6
    assert "ip_address_host" in assessment.indicators
    assert "sensitive_request_on_page" in assessment.indicators


def test_phishing_assessment_accepts_basic_https_url() -> None:
    assessment = assess_url("https://example.org/article", "Public article text.")

    assert assessment.status == "likely_safe"
    assert assessment.risk_score < 0.3


def test_message_context_makes_linked_account_update_suspicious() -> None:
    assessment = assess_url(
        "https://web.fishy.com/",
        "bank acc expiry click to update pin",
    )
    assert assessment.risk_score >= 0.6
    assert "urgent_message_language" in assessment.indicators
    assert "sensitive_request_on_page" in assessment.indicators


def test_phishing_assessment_rejects_invalid_url() -> None:
    with pytest.raises(ValueError):
        assess_url("file:///secret")
