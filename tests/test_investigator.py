import json

from app.services.investigator import investigate_case


def test_investigator_parses_groq_json(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "verdict": "deceptive",
                                    "risk_score": 0.91,
                                    "confidence": 0.88,
                                    "summary": "The page requests sensitive account action.",
                                    "reasons": ["It asks for a PIN update."],
                                    "website_findings": ["The page contains an account action request."],
                                }
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: Response())
    monkeypatch.setattr(
        "app.services.investigator.fetch_url_text",
        lambda url, timeout: "The page asks visitors to update their PIN.",
    )
    result = investigate_case(
        message_text="bank acc expiry click to update pin",
        urls=["https://web.fishy.com/"],
        claims=[],
        evidence=[],
        images=[],
    )

    assert result is not None
    assert result.verdict == "deceptive"
    assert result.risk_score == 0.91
    assert result.provider == "groq"


def test_investigator_is_optional_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert investigate_case(
        message_text="hello",
        urls=[],
        claims=[],
        evidence=[],
    ) is None


def test_investigator_trace_is_opt_in(monkeypatch, caplog) -> None:
    monkeypatch.setenv("ANALYSIS_TRACE", "true")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "verdict": "genuine",
                                    "risk_score": 0.1,
                                    "confidence": 0.9,
                                    "summary": "No issue found.",
                                    "reasons": ["No suspicious request."],
                                    "website_findings": ["No page supplied."],
                                }
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: Response())
    with caplog.at_level("INFO"):
        investigate_case(
            message_text="hello",
            urls=[],
            claims=[],
            evidence=[],
        )
    assert "groq_investigation_started" in caplog.text
