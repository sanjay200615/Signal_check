from types import SimpleNamespace

import pytest

import app.services.adapters as adapters
from app.schemas.common import ClaimContract, EvidenceContract
from app.services.case_analysis import analyze_case
from app.services.case_explanation import build_explanation
from app.services.case_store import CaseInput, VerificationCase, build_text_input
from app.services.cross_channel import ChannelInput, Finding, analyze_cross_channel
from app.services.fusion import fuse_signals
from app.services.reputation import Reputation


@pytest.fixture
def reputation(monkeypatch) -> Reputation:
    rep = Reputation(tranco={"paypal.com": 20, "amazon.com": 10})
    monkeypatch.setattr(adapters, "default_reputation", lambda: rep)
    return rep


def find(inputs, reputation, **kwargs):
    return {f.indicator: f for f in analyze_cross_channel(inputs, reputation, **kwargs)}


def text(i, t):
    return ChannelInput(f"t{i}", "text", t)


def voice(i, t):
    return ChannelInput(f"a{i}", "audio", t)


# ------------------------------------------------------------------ brand consistency
def test_different_senders_across_channels_are_flagged(reputation):
    found = find([text(1, "PayPal: your payment is ready."), voice(1, "Hello, this is Amazon calling about your order.")], reputation)
    assert "sender_brand_inconsistency" in found
    assert "the text message" in found["sender_brand_inconsistency"].detail.lower()


def test_the_same_or_related_brands_are_consistent(reputation):
    assert "sender_brand_inconsistency" not in find([text(1, "PayPal receipt"), voice(1, "your PayPal payment")], reputation)
    rep = Reputation()  # default brand map: Microsoft and Outlook share official domains
    assert "sender_brand_inconsistency" not in find([text(1, "Microsoft account"), voice(1, "your Outlook inbox")], rep)


# ------------------------------------------------------------------ contacts and amounts
def test_conflicting_contact_details_are_flagged_but_shared_ones_are_not(reputation):
    assert "contact_number_mismatch" in find([text(1, "Call 0800 123 4567 today."), voice(1, "Ring 0800 765 4321.")], reputation)
    assert "contact_number_mismatch" not in find([text(1, "Call 0800 123 4567."), voice(1, "Ring 0800 123 4567.")], reputation)
    assert "contact_email_mismatch" in find([text(1, "Write to help@a.com"), voice(1, "email support@b.org")], reputation)


def test_dates_are_not_mistaken_for_phone_numbers(reputation):
    assert "contact_number_mismatch" not in find([text(1, "Due 2026-10-28 at the desk."), voice(1, "Call 0800 123 4567")], reputation)


def test_amounts_are_normalised_so_equal_amounts_match(reputation):
    assert "amount_mismatch" not in find([text(1, "Pay $100 now"), voice(1, "send 100 dollars")], reputation)
    assert "amount_mismatch" in find([text(1, "Pay $50 now"), voice(1, "send 500 dollars")], reputation)


# ------------------------------------------------------------------ repeated pressure
def test_pressure_in_two_different_channels_is_a_campaign_pattern(reputation):
    found = find(
        [text(1, "Your account will be suspended immediately."), voice(1, "Read me the one-time password we just sent you.")],
        reputation,
    )
    assert "multichannel_pressure" in found and set(found["multichannel_pressure"].input_ids) == {"t1", "a1"}


def test_pressure_in_one_channel_alone_or_in_the_same_channel_twice_is_not_flagged(reputation):
    assert "multichannel_pressure" not in find([text(1, "Your account will be suspended immediately."), voice(1, "See you at lunch.")], reputation)
    assert "multichannel_pressure" not in find(
        [text(1, "Your account will be suspended immediately."), text(2, "Final notice: your account is suspended.")], reputation
    )


def test_a_single_input_or_empty_text_yields_no_findings(reputation):
    assert analyze_cross_channel([text(1, "Pay immediately or be arrested.")], reputation) == []
    assert analyze_cross_channel([ChannelInput("u1", "url", ""), text(1, "hello there")], reputation) == []


# ------------------------------------------------------------------ NLI between inputs
def claim(cid, source, claim_text):
    return ClaimContract(claim_id=cid, claim_text=claim_text, source_input_ids=[source], verification_status="insufficient_evidence")


class StubNLI:
    def __init__(self, label="contradiction", score=0.95):
        self.calls = 0
        self._label, self._score = label, score

    def __call__(self, premise, hypothesis):
        self.calls += 1
        return {"labels": {self._label: self._score, "neutral": 1 - self._score}}


def test_claims_in_different_inputs_that_contradict_are_flagged(reputation):
    claims = [claim("c1", "t1", "The bank branch closes at five on Fridays"), claim("c2", "a1", "The bank branch closes at noon on Fridays")]
    nli = StubNLI()
    found = find([text(1, "x y z"), voice(1, "x y z")], reputation, claims=claims, nli=nli)
    assert "cross_input_contradiction" in found and nli.calls == 1


def test_claims_from_the_same_input_or_unrelated_claims_are_not_compared(reputation):
    nli = StubNLI()
    same = [claim("c1", "t1", "The bank branch closes at five"), claim("c2", "t1", "The bank branch closes at noon")]
    unrelated = [claim("c1", "t1", "Paris is in France"), claim("c2", "a1", "Insulin comes from pancreas")]
    assert "cross_input_contradiction" not in find([], reputation, claims=same, nli=nli)
    assert "cross_input_contradiction" not in find([], reputation, claims=unrelated, nli=nli)
    assert nli.calls == 0


def test_a_weak_or_non_contradicting_verdict_is_not_flagged(reputation):
    claims = [claim("c1", "t1", "The bank branch closes at five"), claim("c2", "a1", "The bank branch closes at noon")]
    assert not find([], reputation, claims=claims, nli=StubNLI(score=0.6))
    assert not find([], reputation, claims=claims, nli=StubNLI(label="entailment"))
    assert not find([], reputation, claims=claims, nli=None)


# ------------------------------------------------------------------ fusion + case
def test_fusion_reports_each_signals_contribution():
    result = fuse_signals(claim_contradiction=0.5, phishing=0.2, cross_channel=0.3, has_evidence=True)
    assert result.contributions == {"claim_contradiction": 0.35, "phishing_and_scam_language": 0.2, "cross_channel": 0.3}
    assert result.risk_score == pytest.approx(0.85)


def audio_item(text_value):
    return CaseInput(
        input_id="a1", input_type="audio", created_at="2026-10-28T00:00:00+00:00", content_hash="0" * 64, text=text_value,
        file_name="v.wav", size_bytes=5, raw_content=b"x", transcription_status="completed", audio_meta={"source": "whisper"},
    )


def run_case(reputation, *inputs, **kwargs):
    case = VerificationCase(case_id="c", created_at="", inputs=list(inputs))
    return analyze_case(case, text_analyzer=lambda t: ([], "insufficient_evidence"), **kwargs)


def test_repeated_pressure_across_channels_raises_a_case_that_each_channel_alone_would_not(reputation):
    sms = build_text_input("Your account will be suspended immediately.")
    call = audio_item("Please read me the one-time password we just sent you.")
    alone_sms, alone_call = run_case(reputation, sms), run_case(reputation, call)
    assert alone_sms.verdict == "needs_review" and alone_call.verdict == "needs_review"
    combined = run_case(reputation, sms, call)
    assert combined.risk_score > max(alone_sms.risk_score, alone_call.risk_score)
    assert combined.verdict == "deceptive"
    assert combined.risk_breakdown["cross_channel"] == pytest.approx(0.3)
    assert any(item.kind == "cross_channel" for item in combined.explanation)


def test_cross_channel_can_be_switched_off_for_ablation(reputation):
    sms, call = build_text_input("Your account will be suspended immediately."), audio_item("Read me the one-time password.")
    assert "cross_channel" not in run_case(reputation, sms, call, cross_channel=False).risk_breakdown


def test_agreement_between_channels_never_clears_a_case(reputation):
    a, b = build_text_input("See you at lunch on Friday."), audio_item("See you at lunch on Friday.")
    assert run_case(reputation, a, b).verdict == "needs_review"


# ------------------------------------------------------------------ explanation
def test_explanation_is_ordered_evidence_linked_and_deduplicated():
    signals = [
        SimpleNamespace(input_id="u1", details={"indicators": ["brand_in_subdomain", "urgent_message_language"]}),
        SimpleNamespace(input_id="t1", details={"indicators": ["urgent_message_language"]}),
    ]
    claims = [SimpleNamespace(claim_id="c1", claim_text="NASA confirms 15 days of darkness", verification_status="refuted")]
    evidence = [
        EvidenceContract(evidence_id="e1", claim_id="c1", passage="No.", source_name="Forbes", domain="forbes.com",
                         retrieval_score=0.5, nli_label="contradiction", nli_score=0.9),
        EvidenceContract(evidence_id="e2", claim_id="c1", passage="Hoax.", source_name="Snopes", domain="snopes.com",
                         retrieval_score=0.5, nli_label="contradiction", nli_score=0.9),
    ]
    finding = Finding("multichannel_pressure", 0.3, "Pressure appears in the text and the voice note.", ("t1", "a1"))
    items = build_explanation(signals=signals, claims=claims, evidence=evidence, cross_findings=[finding], trusted_domain=False, ai_escalated=False)
    kinds = [i["kind"] for i in items]
    assert kinds == sorted(kinds, key=["risk", "cross_channel", "claim", "reassurance", "info"].index)
    texts = [i["text"] for i in items]
    assert sum("pressure" in t.lower() for t in texts if "urgency" in t.lower()) == 1  # urgent language listed once
    claim_line = next(i for i in items if i["kind"] == "claim")
    assert "forbes.com" in claim_line["text"] and "snopes.com" in claim_line["text"] and claim_line["refs"] == ["e1", "e2"]


def test_reassurance_lines_appear_only_when_earned():
    nothing = build_explanation(signals=[], claims=[], evidence=[], cross_findings=[], trusted_domain=False, ai_escalated=False)
    assert nothing == []
    trusted = build_explanation(signals=[], claims=[], evidence=[], cross_findings=[], trusted_domain=True, ai_escalated=False)
    assert [i["kind"] for i in trusted] == ["reassurance"]
