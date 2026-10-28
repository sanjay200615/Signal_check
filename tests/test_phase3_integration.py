import pytest

import app.services.adapters as adapters
from app.services.case_analysis import analyze_case
from app.services.case_store import VerificationCase, build_text_input, build_url_input
from app.services.reputation import Reputation, normalize_feed_url


@pytest.fixture
def reputation(monkeypatch) -> Reputation:
    rep = Reputation(
        tranco={d: i for i, d in enumerate(
            ["google.com", "amazon.com", "paypal.com", "github.com", "mayoclinic.org", "wikipedia.org"], 1)},
        phishing_urls=frozenset({normalize_feed_url("http://listed.example/pay")}),
    )
    monkeypatch.setattr(adapters, "default_reputation", lambda: rep)
    return rep


def run(message: str | None, url: str | None):
    inputs = []
    if message:
        inputs.append(build_text_input(message))
    if url:
        inputs.append(build_url_input(url))
    case = VerificationCase(case_id="c", created_at="", inputs=inputs)
    return analyze_case(case, text_analyzer=lambda text: ([], "insufficient_evidence"))


def test_trusted_link_with_matching_brand_is_genuine_without_claim_evidence(reputation):
    result = run("Your PayPal receipt for your recent payment is available.", "https://www.paypal.com/activity")
    assert result.verdict == "genuine"
    assert result.deception_risk == "low"
    assert "not proof" in result.decision_basis
    signal = next(s for s in result.signals if s.signal_type == "phishing_risk")
    assert signal.details["trusted_domain"] is True


def test_brand_mismatch_blocks_genuine_even_on_a_popular_domain(reputation):
    result = run("PayPal: please review your recent payment.", "https://www.mayoclinic.org/")
    signal = next(s for s in result.signals if s.signal_type == "phishing_risk")
    assert "message_brand_mismatch" in signal.details["indicators"]
    assert result.verdict == "needs_review"
    assert result.deception_risk == "elevated"
    assert "message brand mismatch" in result.decision_basis


def test_unknown_domain_without_evidence_needs_review(reputation):
    assert run("Have a look at this page.", "https://small-unknown-site.example/").verdict == "needs_review"


def test_feed_listed_url_is_deceptive(reputation):
    assert run("Pay your invoice.", "http://listed.example/pay").verdict == "deceptive"


def test_scam_message_without_any_url_is_still_scored(reputation):
    result = run("Congratulations, you have won a gift card! Pay a small customs fee to release delivery immediately.", None)
    assert any(s.signal_type == "scam_language" for s in result.signals)
    assert result.verdict in {"needs_review", "deceptive"}
    assert result.risk_score >= 0.3


def test_short_link_is_judged_by_its_destination(reputation, monkeypatch):
    monkeypatch.setattr(adapters, "network_checks_enabled", lambda: True)
    monkeypatch.setattr(adapters, "default_cache", lambda: None)
    monkeypatch.setattr(adapters, "domain_age_days", lambda domain, cache=None: 3)
    monkeypatch.setattr(adapters, "expand_short_url", lambda url: "http://listed.example/pay")
    result = run("Check this out.", "https://bit.ly/abc")
    signal = next(s for s in result.signals if s.signal_type == "phishing_risk")
    assert "short_link_expanded" in signal.details["indicators"]
    assert "known_phishing_url" in signal.details["indicators"]
    assert signal.details["final_url"] == "http://listed.example/pay"
    assert signal.details["domain_age_days"] == 3
    assert result.verdict == "deceptive"


def test_failed_expansion_falls_back_to_the_short_link_itself(reputation, monkeypatch):
    monkeypatch.setattr(adapters, "network_checks_enabled", lambda: True)
    monkeypatch.setattr(adapters, "default_cache", lambda: None)
    monkeypatch.setattr(adapters, "domain_age_days", lambda domain, cache=None: None)
    monkeypatch.setattr(adapters, "expand_short_url", lambda url: None)
    result = run("Check this out.", "https://bit.ly/abc")
    signal = next(s for s in result.signals if s.signal_type == "phishing_risk")
    assert signal.details["final_url"] == "https://bit.ly/abc"
    assert result.verdict == "needs_review"
