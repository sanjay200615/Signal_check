from io import BytesIO
import re

from PIL import Image
import pytest

import app.services.adapters as adapters
from app.services.case_analysis import analyze_case
from app.services.case_store import (
    VerificationCase,
    build_image_input,
    build_text_input,
    build_url_input,
)
from app.services.ocr import domains_in_text, read_image
from app.services.reputation import Reputation
from evaluation.synth_images import render_login_page, render_plain, render_sms


@pytest.fixture
def reputation(monkeypatch) -> Reputation:
    rep = Reputation(
        tranco={d: i for i, d in enumerate(["paypal.com", "google.com", "mayoclinic.org", "amazon.com"], 1)}
    )
    monkeypatch.setattr(adapters, "default_reputation", lambda: rep)
    return rep


def analyze(*inputs):
    case = VerificationCase(case_id="c", created_at="", inputs=list(inputs))
    return analyze_case(case, text_analyzer=lambda text: ([], "insufficient_evidence"))


def image(png: bytes):
    return build_image_input(png, file_name="s.png", mime_type="image/png")


def signal(result, kind):
    return next((s for s in result.signals if s.signal_type == kind), None)


def test_ocr_reads_a_whole_screenshot():
    message = "Your DHL parcel is on hold. Pay a small customs fee to release delivery today."
    text = read_image(render_sms(message)).text
    wanted = re.findall(r"[a-z0-9]+", message.lower())
    got = set(re.findall(r"[a-z0-9]+", text.lower()))
    assert sum(word in got for word in wanted) / len(wanted) >= 0.9


def test_ocr_reports_failures_cleanly():
    with pytest.raises(ValueError):
        read_image(b"not an image")
    with pytest.raises(ValueError):
        read_image(b"")
    assert read_image(render_plain()).text.strip() == ""


def test_domains_in_text_finds_addresses_but_not_email_hosts():
    assert domains_in_text("paypa1-secure-login.top/signin") == ["paypa1-secure-login.top"]
    assert domains_in_text("Visit https://www.example.com/a?b=1 now") == ["www.example.com"]
    assert domains_in_text("write to help@bank.com") == []


def test_scam_text_inside_a_screenshot_is_detected_with_no_other_input(reputation):
    png = render_sms("URGENT: Your account will be suspended today. Click to update your KYC and enter your PIN immediately.")
    result = analyze(image(png))
    assert result.verdict == "deceptive"
    assert "sensitive_request_on_page" in signal(result, "image_text_risk").details["indicators"]


def test_a_harmless_screenshot_alone_can_never_be_genuine(reputation):
    result = analyze(image(render_sms("See you at the cafe at noon, I saved us a table by the window.")))
    assert result.verdict == "needs_review"


def test_lookalike_address_shown_in_a_screenshot_is_flagged(reputation):
    result = analyze(image(render_login_page("PayPal", "paypa1-secure-login.top/signin")))
    details = signal(result, "image_text_risk").details
    assert "message_brand_mismatch" in details["indicators"]
    assert details["shown_domains"] == ["paypa1-secure-login.top"]
    assert result.verdict in {"needs_review", "deceptive"}


def test_brand_shown_in_image_but_linked_elsewhere_is_deceptive(reputation):
    page = image(render_login_page("PayPal", "paypal.com/signin"))
    result = analyze(page, build_url_input("https://www.mayoclinic.org/login"))
    assert "image_brand_domain_mismatch" in signal(result, "image_text_risk").details["indicators"]
    assert result.verdict == "deceptive"


def test_consistent_brand_and_link_raises_no_image_risk(reputation):
    page = image(render_login_page("PayPal", "paypal.com/signin"))
    result = analyze(page, build_url_input("https://www.paypal.com/signin"))
    assert signal(result, "image_text_risk") is None
    assert result.verdict == "genuine"  # cleared by the trusted link, never by the image itself


def test_screenshot_address_that_differs_from_the_link_is_noted(reputation):
    page = image(render_login_page("Google", "google.com/signin"))
    result = analyze(page, build_url_input("https://www.mayoclinic.org/"))
    assert "screenshot_address_differs_from_link" in signal(result, "image_text_risk").details["indicators"]


def test_exif_editing_software_is_reported(reputation):
    img = Image.new("RGB", (60, 40), "white")
    exif = Image.Exif()
    exif[305] = "Adobe Photoshop 25.0"
    buffer = BytesIO()
    img.save(buffer, "JPEG", exif=exif)
    result = analyze(build_image_input(buffer.getvalue(), file_name="p.jpg", mime_type="image/jpeg"))
    details = signal(result, "image_metadata").details
    assert details["software"] == "Adobe Photoshop 25.0" and details["edited_with_software"] is True


def clip_stub(monkeypatch, similarity: float):
    monkeypatch.setattr(adapters, "vision_enabled", lambda: True)
    monkeypatch.setattr(adapters, "image_text_similarity", lambda image_bytes, texts: [similarity])


CAPTION = "A large crowd of people is gathered in the city square after the flood waters receded."


def test_photo_that_clip_finds_inconsistent_with_its_caption_needs_review(reputation, monkeypatch):
    clip_stub(monkeypatch, 0.12)
    result = analyze(build_text_input(CAPTION), image(render_plain()))
    consistency = signal(result, "image_text_consistency")
    assert consistency.details["consistent"] is False and consistency.score == 0.6
    assert result.verdict == "needs_review"


def test_photo_that_matches_its_caption_records_consistency_without_clearing_it(reputation, monkeypatch):
    clip_stub(monkeypatch, 0.31)
    result = analyze(build_text_input(CAPTION), image(render_plain()))
    assert signal(result, "image_text_consistency").details["consistent"] is True
    assert result.verdict == "needs_review"  # a text-free photo can never be verified as authentic


def test_non_descriptive_messages_are_not_compared_with_the_photo(reputation, monkeypatch):
    clip_stub(monkeypatch, 0.05)
    result = analyze(build_text_input("Check this out"), image(render_plain()))
    assert signal(result, "image_text_consistency") is None


def test_screenshot_chrome_is_removed_before_claim_extraction():
    from app.services.ocr import claim_text_from_ocr

    ocr = "Messages\nParis is the capital\nof France.\nToday10:42"
    assert claim_text_from_ocr(ocr) == "Paris is the capital of France."
    assert claim_text_from_ocr("Sign in\nPassword") == ""


def test_claims_from_a_screenshot_do_not_include_app_chrome(reputation):
    seen = []
    case = VerificationCase(case_id="c", created_at="", inputs=[image(render_sms("Paris is the capital of France."))])
    analyze_case(case, text_analyzer=lambda text: (seen.append(text) or [], "insufficient_evidence"))
    assert seen and "Messages" not in seen[0] and "Paris is the capital of France" in seen[0]


def test_caption_mismatch_is_named_in_the_explanation(reputation, monkeypatch):
    clip_stub(monkeypatch, 0.12)
    result = analyze(build_text_input(CAPTION), image(render_plain()))
    assert result.deception_risk == "elevated"
    assert "image does not match its caption" in result.decision_basis
