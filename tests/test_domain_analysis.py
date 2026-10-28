import pytest

from app.services.domain_analysis import (
    analyze_domain,
    edit_distance,
    is_free_hosting,
    is_shortener,
    looks_random,
)
from app.services.phishing import assess_message, assess_url
from app.services.reputation import Reputation, normalize_feed_url, registered_domain


@pytest.fixture
def reputation() -> Reputation:
    ranks = {d: i for i, d in enumerate(
        ["google.com", "amazon.com", "paypal.com", "microsoft.com", "github.com",
         "wikipedia.org", "mayoclinic.org", "python.org", "bbc.co.uk", "apple.com"], 1)}
    return Reputation(
        tranco=ranks,
        phishing_urls=frozenset({normalize_feed_url("http://evil.example/login.php")}),
        phishing_hosts=frozenset({"bad-host.example"}),
    )


def indicators(host: str, reputation: Reputation) -> set[str]:
    return {finding.indicator for finding in analyze_domain(host, reputation)}


def test_registered_domain_handles_multi_part_suffixes():
    assert registered_domain("www.bbc.co.uk") == "bbc.co.uk"
    assert registered_domain("a.b.example.com") == "example.com"
    assert registered_domain("203.0.113.9") == ""


def test_edit_distance_counts_transpositions_as_one():
    assert edit_distance("paypal", "paypal") == 0
    assert edit_distance("paypal", "paypa") == 1
    assert edit_distance("paypal", "paypla") == 1
    assert edit_distance("paypal", "amazon") > 2


def test_free_hosting_and_shorteners_are_recognised():
    assert is_free_hosting("pub-1234.r2.dev")
    assert is_free_hosting("r2.dev")
    assert not is_free_hosting("notr2.dev")
    assert is_shortener("bit.ly") and not is_shortener("example.com")


def test_random_labels_are_detected_but_normal_words_are_not():
    assert looks_random("pub-28b6157a0bea4e0083b04d82091bef63")
    assert looks_random("sp30ct10rumlekobizzureti")
    assert not looks_random("floridahealth")
    assert not looks_random("my-portfolio")


def test_homoglyph_and_typosquat_lookalikes(reputation):
    assert "homoglyph_lookalike" in indicators("paypa1.com", reputation)
    assert "typosquat_lookalike" in indicators("amazonn.com", reputation)


def test_official_and_popular_domains_are_never_lookalikes(reputation):
    assert not indicators("www.paypal.com", reputation)
    assert not indicators("en.wikipedia.org", reputation)
    assert not indicators("github.com", reputation)


def test_brand_in_subdomain_or_label_of_foreign_domain(reputation):
    assert "brand_in_subdomain" in indicators("appleid.apple.verify-account.top", reputation)
    assert "brand_in_domain_label" in indicators("paypal-secure-login.gq", reputation)
    assert "brand_in_subdomain" not in indicators("support.apple.com", reputation)


def test_free_hosting_with_random_label_and_brand(reputation):
    found = indicators("pub-28b6157a0bea4e0083b04d82091bef63.r2.dev", reputation)
    assert {"free_hosting_platform", "random_subdomain_label"} <= found
    assert "brand_in_subdomain" in indicators("paypal-login.vercel.app", reputation)
    assert "brand_on_free_hosting" in indicators("paypallogin.vercel.app", reputation)


def test_feed_hits_are_strong_signals(reputation):
    assert "known_phishing_host" in indicators("bad-host.example", reputation)
    assessment = assess_url("http://evil.example/login.php", reputation=reputation)
    assert "known_phishing_url" in assessment.indicators
    assert assessment.status == "likely_phishing"


def test_domain_age_signals():
    young = assess_url("https://example.org/", domain_age_days=5)
    recent = assess_url("https://example.org/", domain_age_days=60)
    old = assess_url("https://example.org/", domain_age_days=4000)
    assert "newly_registered_domain" in young.indicators and young.risk_score >= 0.5
    assert "recently_registered_domain" in recent.indicators
    assert not {"newly_registered_domain", "recently_registered_domain"} & set(old.indicators)


def test_trusted_domain_requires_popularity_and_low_risk(reputation):
    assert assess_url("https://www.mayoclinic.org/", "Your appointment reminder.", reputation=reputation).trusted_domain
    assert not assess_url("https://unknown-site.example/", reputation=reputation).trusted_domain
    assert not assess_url("https://www.mayoclinic.org/", reputation=None).trusted_domain


def test_message_naming_a_brand_but_linking_elsewhere_is_flagged(reputation):
    mismatch = assess_url("https://www.mayoclinic.org/", "Your PayPal account needs attention.", reputation=reputation)
    assert "message_brand_mismatch" in mismatch.indicators
    assert not mismatch.trusted_domain
    match = assess_url("https://www.paypal.com/activity", "Your PayPal receipt is ready.", reputation=reputation)
    assert "message_brand_mismatch" not in match.indicators
    assert match.trusted_domain


def test_message_cues_need_request_or_threat_phrasing():
    assert assess_message("Your Google account password was changed. Ignore this if it was you.").risk_score < 0.3
    assert assess_message("Your PayPal receipt for your recent payment is available.").risk_score < 0.3
    scam = assess_message("URGENT: your account will be suspended. Enter your PIN immediately.")
    assert scam.risk_score >= 0.5
    assert "sensitive_request_on_page" in scam.indicators


def test_message_cue_families():
    assert "crypto_doubling" in assess_message("Send 0.1 BTC and receive 0.2 BTC back").indicators
    assert "advance_fee_request" in assess_message("Pay a small customs fee to release delivery").indicators
    assert "financial_bait" in assess_message("Congratulations, you have won a gift card").indicators


def test_multi_tenant_platforms_are_never_trusted_even_when_the_platform_is_popular():
    from app.services.reputation import is_multi_tenant

    rep = Reputation(tranco={"square.site": 3000, "gitbook.io": 3500, "github.com": 5})
    for host in ("shop.square.site", "evil.gitbook.io", "x.github.io"):
        assert is_free_hosting(host), host
        assert not assess_url(f"https://{host}/", reputation=rep).trusted_domain, host
    assert is_multi_tenant("foo.vercel.app") and not is_multi_tenant("www.google.com")


def test_same_label_on_a_different_tld_is_a_lookalike(reputation):
    rep = Reputation(tranco={**reputation.tranco, "roblox.com": 900}, brands=reputation.brands)
    assert "same_label_other_tld" in indicators("roblox.ly", rep)
    assert "same_label_other_tld" not in indicators("roblox.com", rep)


def test_brand_domain_spelled_out_in_the_path_is_flagged(reputation):
    assessment = assess_url("https://hameem.net/adobe.com/index.html", reputation=reputation)
    assert "brand_domain_in_path" in assessment.indicators
    own = assess_url("https://www.paypal.com/paypal.com/help", reputation=reputation)
    assert "brand_domain_in_path" not in own.indicators


def test_url_shorteners_are_never_trusted_even_though_they_are_popular():
    rep = Reputation(tranco={"bit.ly": 400, "github.com": 5})
    assert not assess_url("https://bit.ly/abc", "Look at this.", reputation=rep).trusted_domain
    assert assess_url("https://github.com/x", "Look at this.", reputation=rep).trusted_domain


def test_named_brands_tolerates_ocr_confusions(reputation):
    from app.services.phishing import named_brands

    assert named_brands("Your SBl account", reputation) == ["sbi"]
    assert named_brands("LinkedIn message", reputation) == ["linkedin"]
    assert named_brands("Sign in to PayPaI", reputation) == ["paypal"]
    assert named_brands("A paypal-like word: paypalish", reputation) == ["paypal"]
    assert named_brands("Nothing relevant here", reputation) == []


def test_everyday_words_that_are_also_brands_need_capitalisation(reputation):
    from app.services.phishing import named_brands

    assert named_brands("I like apple pie and will chase this up", reputation) == []
    assert named_brands("Your Apple ID is locked", reputation) == ["apple"]
    assert named_brands("UPS could not deliver", Reputation()) == ["ups"]
    benign = assess_url("https://www.mayoclinic.org/", "I will zoom in and chase this up later.", reputation=reputation)
    assert "message_brand_mismatch" not in benign.indicators


def test_clickbait_lure_needs_a_sensational_event_and_a_call_to_watch():
    assert "clickbait_lure" in assess_message("dua lipa dead watch video").indicators
    assert "clickbait_lure" in assess_message("Shocking leaked video, click here to see the full video").indicators
    assert "clickbait_lure" not in assess_message("Watch the video tutorial to learn Python").indicators
    assert "clickbait_lure" not in assess_message("The singer died in 2016.").indicators
    assert 0.3 <= assess_message("dua lipa dead watch video").risk_score < 0.6
