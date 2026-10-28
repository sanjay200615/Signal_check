from types import SimpleNamespace

from app.services.adapters import (
    ADAPTERS,
    AdapterContext,
    ImageAdapter,
    TextAdapter,
    UrlAdapter,
)
from app.services.case_store import (
    build_audio_input,
    build_text_input,
    build_url_input,
)


def _context(analyzer=None) -> AdapterContext:
    return AdapterContext(text_analyzer=analyzer or (lambda text: ([], "insufficient_evidence")))


def test_every_supported_input_type_has_an_adapter():
    assert set(ADAPTERS) == {"text", "audio", "url", "image"}


def test_text_adapter_records_claims_and_contradiction():
    evidence = SimpleNamespace(
        passage_id="e1", passage="no", source_url=None, source_name="S",
        retrieval_similarity=0.8, nli_label="contradiction", nli_score=0.9,
        source_type="t", source_quality="official", source_quality_weight=1.0,
        published_date=None, retrieved_date="2026-10-28", provider="local", domain=None,
    )
    claim = SimpleNamespace(claim_id="claim_1", claim_text="X", entities=[], claim_type="factual")
    result = SimpleNamespace(claim=claim, status="refuted", evidence=[evidence])
    item = build_text_input("X is true")

    output = TextAdapter().analyze(item, _context(lambda text: ([result], "refuted")))

    assert output.contradiction == 0.9
    assert output.claims[0].verification_status == "refuted"
    assert output.claims[0].source_input_ids == [item.input_id]
    assert output.evidence[0].claim_id == output.claims[0].claim_id
    assert output.text_items == [(item.input_id, "X is true")]


def test_text_adapter_ignores_inputs_without_text():
    item = build_audio_input(b"audio-bytes", file_name="a.wav", mime_type="audio/wav")
    output = ADAPTERS["audio"].analyze(item, _context())
    assert output.claims == [] and output.text_items == []


def test_audio_adapter_uses_the_supplied_transcript():
    item = build_audio_input(
        b"audio-bytes", file_name="a.wav", mime_type="audio/wav", transcript="hello there"
    )
    output = ADAPTERS["audio"].analyze(item, _context())
    assert output.text_items == [(item.input_id, "hello there")]


def test_url_adapter_emits_a_phishing_signal():
    item = build_url_input("http://203.0.113.9/login")
    output = UrlAdapter().analyze(item, _context())
    assert output.phishing > 0
    assert output.signals[0].signal_type == "phishing_risk"
    assert output.text_items == [(item.input_id, "http://203.0.113.9/login")]


def test_image_adapter_without_bytes_does_nothing():
    item = build_text_input("not an image")
    output = ImageAdapter().analyze(item, _context())
    assert output.signals == [] and output.image_context == []
