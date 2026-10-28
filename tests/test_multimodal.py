from pathlib import Path

import httpx
import pytest

from app.services.multimodal import fetch_url_text


def test_fetch_url_text_extracts_title_and_body(monkeypatch) -> None:
    class Response:
        headers = {"content-type": "text/html"}
        text = "<html><title>Claim page</title><body>NASA published a report.</body></html>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: Response())
    assert fetch_url_text("https://example.com/story") == (
        "Claim page NASA published a report."
    )


def test_fetch_url_text_rejects_non_http_urls() -> None:
    with pytest.raises(ValueError, match="absolute http"):
        fetch_url_text("file:///tmp/story.html")
