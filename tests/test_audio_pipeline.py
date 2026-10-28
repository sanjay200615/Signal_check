from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

import app.api.routes_cases as routes_cases
import app.services.audio as audio
from app.main import app
from app.services.audio import MAX_AUDIO_SECONDS, Transcript, WhisperTranscriber
from app.services.case_analysis import analyze_case
from app.services.case_store import CaseInput, VerificationCase, build_audio_input
from app.services.claim_extractor import ExtractedClaim
from app.services.phishing import assess_message
from app.services.pipeline import PipelineClaim, PipelineEvidence


def segment(text, start=0.0):
    return SimpleNamespace(text=text, start=start)


class FakeModel:
    """Stands in for faster-whisper: returns canned segments and records the calls."""

    def __init__(self, language="en", probability=0.99, duration=12.0, original=None, translated=None):
        self.info = SimpleNamespace(language=language, language_probability=probability, duration=duration)
        self.original = original if original is not None else [segment(" Hello there. "), segment("Call me back.", 2.0)]
        self.translated = translated or [segment("Hello there. Call me back.")]
        self.calls = []

    def transcribe(self, source, **kwargs):
        self.calls.append(kwargs)
        return iter(self.translated if kwargs.get("task") == "translate" else self.original), self.info


def transcriber(model) -> WhisperTranscriber:
    t = WhisperTranscriber("tiny")
    t._load = lambda: model
    return t


# ------------------------------------------------------------------ transcriber
def test_english_audio_is_transcribed_once_without_translation():
    model = FakeModel()
    result = transcriber(model).transcribe(b"audio")
    assert result.text == "Hello there. Call me back." and result.translated is False
    assert result.language == "en" and result.has_speech
    assert len(model.calls) == 1 and model.calls[0]["vad_filter"] is True


def test_non_english_speech_is_translated_to_english_and_keeps_the_original():
    model = FakeModel(language="hi", original=[segment("Aapka khata band ho jayega")], translated=[segment("Your account will be closed")])
    result = transcriber(model).transcribe(b"audio")
    assert result.translated is True
    assert result.text == "Your account will be closed" and result.original_text == "Aapka khata band ho jayega"
    assert [c.get("task") for c in model.calls] == [None, "translate"]
    assert model.calls[1]["language"] == "hi"
    assert result.metadata()["original_text"] == "Aapka khata band ho jayega"


def test_uncertain_language_detection_is_not_translated():
    model = FakeModel(language="de", probability=0.3)
    assert transcriber(model).transcribe(b"audio").translated is False
    assert len(model.calls) == 1


def test_a_recording_with_no_speech_has_an_empty_transcript():
    result = transcriber(FakeModel(original=[])).transcribe(b"audio")
    assert result.text == "" and result.has_speech is False


def test_audio_beyond_the_duration_cap_is_not_transcribed():
    model = FakeModel(original=[segment("early words", 5.0), segment("far too late", MAX_AUDIO_SECONDS + 1)])
    result = transcriber(model).transcribe(b"audio")
    assert result.text == "early words" and result.truncated is True


def test_empty_and_undecodable_audio_are_rejected():
    with pytest.raises(ValueError):
        transcriber(FakeModel()).transcribe(b"")

    class Broken:
        def transcribe(self, *args, **kwargs):
            raise ValueError("decode failed")

    with pytest.raises(ValueError, match="not valid audio"):
        transcriber(Broken()).transcribe(b"garbage")


def test_a_missing_model_is_reported_as_unavailable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="could not be loaded"):
        WhisperTranscriber("tiny").transcribe(b"audio")


# ------------------------------------------------------------------ routes
class StubTranscriber:
    def __init__(self, transcript=None, error=None):
        self._transcript, self._error = transcript, error

    def transcribe(self, content):
        if self._error:
            raise self._error
        return self._transcript


def english(text: str) -> Transcript:
    return Transcript(text=text, original_text=text, language="en", language_probability=0.99,
                      duration_seconds=7.0, translated=False)


class NoClaimsPipeline:
    """Keeps route tests offline: the real pipeline would load models and touch the global readiness state."""

    def analyze(self, text: str):
        return [], "insufficient_evidence"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(routes_cases, "case_store", routes_cases.CaseStore())
    monkeypatch.setattr(routes_cases, "get_pipeline", lambda: NoClaimsPipeline())
    return TestClient(app)


def upload(client, monkeypatch, transcriber, **data):
    monkeypatch.setattr(routes_cases, "get_transcriber", lambda: transcriber)
    case_id = client.post("/api/cases").json()["case_id"]
    response = client.post(f"/api/cases/{case_id}/inputs/audio", files={"file": ("v.wav", b"audio", "audio/wav")}, data=data)
    return case_id, response


def test_uploaded_audio_is_transcribed_in_the_background(client, monkeypatch):
    case_id, response = upload(client, monkeypatch, StubTranscriber(english("Please call me back about the invoice")))
    assert response.status_code == 202 and response.json()["inputs"][0]["transcription_status"] == "processing"
    case = client.get(f"/api/cases/{case_id}").json()
    item = case["inputs"][0]
    assert item["text"] == "Please call me back about the invoice"
    assert item["transcription_status"] == "completed" and item["audio_meta"]["language"] == "en"
    assert case["status"] == "collecting_inputs"


def test_a_failed_transcription_is_recorded_and_does_not_block_the_case(client, monkeypatch):
    case_id, _ = upload(client, monkeypatch, StubTranscriber(error=ValueError("The uploaded file is not valid audio.")))
    item = client.get(f"/api/cases/{case_id}").json()["inputs"][0]
    assert item["transcription_status"] == "failed" and "not valid audio" in item["transcription_error"]
    result = client.post(f"/api/cases/{case_id}/analyze")
    assert result.status_code == 200 and result.json()["verdict"] == "needs_review"


def test_analysis_waits_while_audio_is_still_processing(client):
    case = routes_cases.case_store.create()
    routes_cases.case_store.add_input(case.case_id, build_audio_input(b"audio", file_name="v.wav", mime_type="audio/wav"))
    response = client.post(f"/api/cases/{case.case_id}/analyze")
    assert response.status_code == 409 and "still being transcribed" in response.json()["detail"]


def test_a_supplied_transcript_skips_speech_recognition(client, monkeypatch):
    def must_not_run():
        raise AssertionError("speech recognition must not run")

    monkeypatch.setattr(routes_cases, "get_transcriber", must_not_run)
    case_id = client.post("/api/cases").json()["case_id"]
    client.post(f"/api/cases/{case_id}/inputs/audio", files={"file": ("v.wav", b"audio", "audio/wav")}, data={"transcript": "Your own words"})
    item = client.get(f"/api/cases/{case_id}").json()["inputs"][0]
    assert item["text"] == "Your own words" and item["transcription_status"] == "completed"
    assert item["audio_meta"]["source"] == "supplied"


# ------------------------------------------------------------------ analysis
def analyze(transcript: str | None, meta: dict | None = None, pipeline=None):
    item = CaseInput(
        input_id="a1", input_type="audio", created_at="2026-10-28T00:00:00+00:00", content_hash="0" * 64,
        text=transcript or None, file_name="v.wav", size_bytes=5, raw_content=b"x",
        transcription_status="completed", audio_meta=meta or {"source": "whisper", "language": "en"},
    )
    case = VerificationCase(case_id="c", created_at="", inputs=[item])
    return analyze_case(case, text_analyzer=pipeline or (lambda text: ([], "insufficient_evidence")))


IRS_SCRIPT = (
    "Hello, this is the Internal Revenue Service. Your social security number has been suspended. "
    "Do not hang up. Buy gift cards and read me the numbers immediately to avoid arrest."
)


def test_a_spoken_scam_script_is_deceptive():
    result = analyze(IRS_SCRIPT)
    signal = next(s for s in result.signals if s.signal_type == "scam_language")
    assert {"authority_impersonation", "secrecy_or_pressure", "untraceable_payment"} <= set(signal.details["indicators"])
    assert result.verdict == "deceptive"


def test_a_harmless_voice_note_is_not_cleared_without_support():
    result = analyze("Hi, just confirming dinner at seven on Friday. See you there.")
    assert result.verdict == "needs_review"


def test_a_recording_with_no_speech_can_never_be_cleared():
    result = analyze(None)
    assert result.verdict == "needs_review"
    assert next(s for s in result.signals if s.signal_type == "audio_transcript").details["transcript_chars"] == 0


def test_a_supported_spoken_claim_can_be_genuine_and_limitations_mention_voices():
    claim = ExtractedClaim("claim_1", "Insulin is a hormone produced by the pancreas", [], "factual")
    evidence = PipelineEvidence(
        passage_id="e1", title="T", source_name="S", source_url="https://e.test", published_date=None, passage="It is.",
        retrieval_similarity=0.9, nli_label="entailment", nli_score=0.95, source_type="web_search",
        source_quality="reputable", retrieved_date="2026-10-28",
    )
    result = analyze("Insulin is a hormone produced by the pancreas.", pipeline=lambda t: ([PipelineClaim(claim, "supported", [evidence], "ok")], "supported"))
    assert result.verdict == "genuine"
    assert any("who is speaking" in item for item in result.limitations)


def test_translation_is_disclosed_in_the_limitations():
    result = analyze("Your account will be closed.", meta={"source": "whisper", "language": "hi", "translated": True})
    assert any("translated to English" in item for item in result.limitations)


# ------------------------------------------------------------------ spoken-scam cues
def test_each_spoken_scam_cue_fires_on_its_own_pattern():
    assert "authority_impersonation" in assess_message("This is the FBI. A warrant has been issued for your arrest.").indicators
    assert "secrecy_or_pressure" in assess_message("Don't hang up and don't tell anyone about this call.").indicators
    assert "untraceable_payment" in assess_message("Pay the fee with gift cards or a wire transfer.").indicators
    assert "remote_access_request" in assess_message("Install AnyDesk so we can fix the virus on your computer.").indicators
    mum = assess_message("Hi Mum, my phone broke so this is my new number. Can you send £200 for the repair?")
    assert "family_emergency_scam" in mum.indicators and 0.3 <= mum.risk_score < 0.6


def test_ordinary_calls_do_not_trigger_the_spoken_scam_cues():
    for text in (
        "This is Amazon customer service calling about your delivery tomorrow.",
        "Hi Mum, can you pick up some milk on the way home?",
        "I will pay you back on Friday for the concert tickets.",
        "The technical support team resolved the ticket yesterday.",
        "Please stay on the line while I transfer you to the billing department.",
    ):
        flagged = set(assess_message(text).indicators) & {
            "authority_impersonation", "secrecy_or_pressure", "untraceable_payment", "family_emergency_scam",
        }
        assert not flagged, (text, flagged)


def test_reading_out_a_one_time_code_and_unsolicited_alerts_are_flagged():
    otp = assess_message("To cancel it, read me the one-time password we just sent you.")
    assert "otp_request" in otp.indicators
    alert = assess_message("We detected a virus on your computer. Someone tried to withdraw $5,000 from your account.")
    assert "unsolicited_alert" in alert.indicators


def test_a_spoken_amount_and_hyphenated_redelivery_are_recognised():
    mum = assess_message("Hi mom, I'm in trouble and this is my new number. Please send two hundred pounds for the repair.")
    assert "family_emergency_scam" in mum.indicators
    assert "advance_fee_request" in assess_message("Pay a small re-delivery fee today").indicators


def test_a_single_untraceable_payment_or_remote_access_demand_is_serious():
    assert assess_message("Please pay with a prepaid card.").risk_score >= 0.5
    assert assess_message("Install AnyDesk to give us remote access.").risk_score >= 0.5


def test_legitimate_security_and_support_language_does_not_trigger_the_new_cues():
    for text in (
        "We noticed you updated your address, so your next delivery will go to the new one.",
        "The IT team found a problem with the printer and fixed it this morning.",
        "Can you tell me the time of the meeting? I will read the notes later.",
        "Please say your name and the reason for your call after the tone.",
        "Someone tried to book the same room, so I moved our meeting to Thursday.",
    ):
        flagged = set(assess_message(text).indicators) & {"otp_request", "unsolicited_alert", "untraceable_payment", "remote_access_request"}
        assert not flagged, (text, flagged)
