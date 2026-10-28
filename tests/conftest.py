import pytest


@pytest.fixture(autouse=True)
def hermetic_environment(monkeypatch):
    """Keep every test offline: no downloads, RDAP/redirect lookups, disk cache or API keys."""
    monkeypatch.setenv("REPUTATION_DATA", "off")
    monkeypatch.setenv("URL_NETWORK_CHECKS", "off")
    monkeypatch.setenv("EVIDENCE_CACHE", "off")
    monkeypatch.setenv("VISION_MODELS", "off")
    monkeypatch.setenv("GROQ_MIN_INTERVAL_SECONDS", "0")
    from app.services.investigator import reset_rate_limit_state

    reset_rate_limit_state()
    # The project's .env may hold real keys; tests must never call external services with them.
    for name in ("GROQ_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)
