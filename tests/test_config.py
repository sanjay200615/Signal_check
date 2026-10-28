import pytest

from app.core.config import Settings


def test_settings_validate_and_read_environment(monkeypatch) -> None:
    monkeypatch.setenv("TOP_K", "5")
    monkeypatch.setenv("MINIMUM_SIMILARITY", "0.4")

    settings = Settings.from_environment()

    assert settings.top_k == 5
    assert settings.minimum_similarity == 0.4


def test_invalid_similarity_configuration_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("MINIMUM_SIMILARITY", "1.5")

    with pytest.raises(ValueError, match="between zero and one"):
        Settings.from_environment()


def test_evidence_provider_defaults_to_live_and_accepts_tavily(monkeypatch) -> None:
    monkeypatch.delenv("EVIDENCE_PROVIDER", raising=False)
    assert Settings.from_environment().evidence_provider == "live"

    monkeypatch.setenv("EVIDENCE_PROVIDER", "TAVILY")
    assert Settings.from_environment().evidence_provider == "tavily"


def test_invalid_evidence_provider_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("EVIDENCE_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="EVIDENCE_PROVIDER"):
        Settings.from_environment()
