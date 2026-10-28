import pytest

from app.services.audio import TranscriptAudioTranscriber
from app.services.correlation import correlate_texts
from app.services.fusion import fuse_signals
from app.services.phishing import assess_url


def test_correlation_detects_shared_url_and_similar_content() -> None:
    result = correlate_texts(
        [
            ("one", "Verify now at https://example.test/login"),
            ("two", "Verify now at https://example.test/login"),
        ]
    )

    assert result.score > 0
    assert result.shared_urls == ("https://example.test/login",)
    assert result.similar_pairs == (("one", "two"),)


def test_fusion_returns_deceptive_for_combined_high_risk_signals() -> None:
    result = fuse_signals(
        claim_contradiction=0.8,
        phishing=0.9,
        coordination=0.8,
        has_evidence=True,
    )

    assert result.verdict == "deceptive"
    assert result.risk_score >= 0.6


def test_fusion_abstains_without_evidence() -> None:
    result = fuse_signals()

    assert result.verdict == "needs_review"


def test_audio_adapter_requires_a_transcript() -> None:
    assert TranscriptAudioTranscriber("A transcript").transcribe(b"audio") == "A transcript"
    with pytest.raises(RuntimeError):
        TranscriptAudioTranscriber("").transcribe(b"audio")
