import pytest

from app.services.evidence_provider import (
    LocalEvidenceProvider,
    DuckDuckGoEvidenceProvider,
    LiveNewsEvidenceProvider,
    HybridEvidenceProvider,
    RetrievedEvidence,
    TavilyEvidenceProvider,
    WikipediaEvidenceProvider,
    _canonical_url,
)


def test_tavily_provider_requires_api_key() -> None:
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        TavilyEvidenceProvider(api_key="")


def test_canonical_url_removes_tracking_query_and_fragment() -> None:
    assert _canonical_url("HTTPS://Example.com/page/?utm_source=test#section") == (
        "https://example.com/page"
    )


def test_local_evidence_is_marked_official() -> None:
    from app.services.corpus import EvidencePassage

    passage = EvidencePassage(
        "p1", "d1", "Title", "Agency", "https://example.com", None,
        "2026-10-27", "official_reference", "Official", "Text",
    )
    class Index:
        def search(self, query, embedder, top_k):
            return [(passage, 0.8)]
    result = LocalEvidenceProvider(Index(), object()).search("claim", top_k=1)[0]
    assert result.source_quality == "official"


def test_live_news_provider_parses_feed(monkeypatch) -> None:
    class Response:
        content = b"""<rss><channel><item>
          <title>Trump is alive, officials say</title>
          <link>https://example.com/story</link>
          <description>Officials said the claim is false.</description>
          <source>Example News</source>
        </item></channel></rss>"""

        def raise_for_status(self):
            return None

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: Response())
    result = LiveNewsEvidenceProvider().search("Trump is dead", top_k=1)
    assert result[0].provider == "live"
    assert result[0].source_name == "Example News"
    assert result[0].passage == "Officials said the claim is false."


def test_live_news_provider_returns_empty_for_empty_feed(monkeypatch) -> None:
    class Response:
        content = b"<rss><channel /></rss>"
        status_code = 200

        def raise_for_status(self):
            return None

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: Response())
    assert LiveNewsEvidenceProvider().search("no matching story", top_k=3) == []


def test_live_news_provider_surfaces_request_failure(monkeypatch) -> None:
    import httpx

    def fail(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr("httpx.get", fail)
    with pytest.raises(RuntimeError, match="Live news search failed"):
        LiveNewsEvidenceProvider().search("claim", top_k=1)


def test_wikipedia_provider_parses_reference_pages(monkeypatch) -> None:
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "pages": [
                    {
                            "id": 123,
                            "title": "Michael Jackson",
                            "excerpt": "Michael Jackson died in 2009.",
                            "key": "Michael_Jackson",
                    }
                ]
            }

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: Response())
    result = WikipediaEvidenceProvider().search("Michael Jackson", top_k=1)
    assert result[0].provider == "wikipedia"
    assert result[0].source_quality == "reputable"
    assert result[0].source_type == "encyclopedia_reference"


def test_hybrid_provider_skips_optional_tavily_without_key(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    provider = HybridEvidenceProvider()
    kinds = {type(source).__name__ for source in provider._providers}
    assert kinds == {
        "WikipediaEvidenceProvider",
        "DuckDuckGoEvidenceProvider",
        "LiveNewsEvidenceProvider",
    }


def test_hybrid_provider_does_not_retry_failed_wikipedia_queries(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs.get("params", {}).get("q"))
        raise RuntimeError("rate limited")

    monkeypatch.setattr(WikipediaEvidenceProvider, "search", fail)
    monkeypatch.setattr(LiveNewsEvidenceProvider, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(DuckDuckGoEvidenceProvider, "search", lambda *args, **kwargs: [])
    provider = HybridEvidenceProvider(failure_cooldown_seconds=60)

    provider.search("Michael Jackson is still alive", top_k=3)
    provider.search("Michael Jackson death", top_k=3)
    provider.search("Michael Jackson alive", top_k=3)

    assert len(calls) == 1


def test_strip_html_removes_markup_and_decodes_entities() -> None:
    from app.services.evidence_provider import _strip_html

    raw = '<span class="searchmatch">Paris</span> is the &quot;capital&quot;&nbsp;of  France'
    assert _strip_html(raw) == 'Paris is the "capital" of France'


def test_duckduckgo_provider_maps_results_and_downgrades_ai_mirrors(monkeypatch) -> None:
    import ddgs
    from app.services.evidence_provider import DuckDuckGoEvidenceProvider

    class FakeDDGS:
        def text(self, query, max_results):
            return [
                {"title": "Flat Earth", "href": "https://en.wikipedia.org/wiki/Flat_Earth",
                 "body": "Flat Earth is a <b>scientifically disproven</b> model."},
                {"title": "Mirror", "href": "https://grokipedia.com/page/Flat_Earth", "body": "Mirror text."},
                {"title": "", "href": "https://x.test", "body": "dropped, no title"},
            ]

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    results = DuckDuckGoEvidenceProvider().search("Earth is flat", top_k=3)

    assert [r.provider for r in results] == ["duckduckgo", "duckduckgo"]
    assert results[0].passage == "Flat Earth is a scientifically disproven model."
    assert results[0].source_quality == "official"
    assert results[1].source_quality == "context_only"
    assert results[1].source_quality_weight == 0.5


def test_duckduckgo_provider_wraps_library_errors(monkeypatch) -> None:
    import ddgs
    import pytest
    from ddgs.exceptions import RatelimitException
    from app.services.evidence_provider import DuckDuckGoEvidenceProvider

    class Failing:
        def text(self, query, max_results):
            raise RatelimitException("slow down")

    monkeypatch.setattr(ddgs, "DDGS", Failing)
    with pytest.raises(RuntimeError):
        DuckDuckGoEvidenceProvider().search("anything", top_k=2)


def test_reputable_and_fact_check_sources_match_with_www_and_subdomains() -> None:
    from app.services.evidence_provider import _live_source_quality

    for url in (
        "https://www.bbc.com/news/x", "https://www.bbc.co.uk/news", "https://www.snopes.com/fact-check/x",
        "https://www.factcheck.org/2017/11/y", "https://apnews.com/article/z", "https://news.sky.example/",
    ):
        expected = "unknown" if "sky.example" in url else "reputable"
        assert _live_source_quality(url) == expected, url
    assert _live_source_quality("https://en.wikipedia.org/wiki/X") == "official"
    assert _live_source_quality("https://grokipedia.com/page/X") == "context_only"


def test_established_references_are_reputable_but_content_farms_are_not() -> None:
    from app.services.evidence_provider import _live_source_quality

    for url in ("https://www.britannica.com/event/Apollo-11", "https://www.scientificamerican.com/x", "https://www.cnn.com/world"):
        assert _live_source_quality(url) == "reputable", url
    for url in ("https://commonplacefacts.com/einstein", "https://www.worldatlas.com/oceans", "https://ludwig.guru/x"):
        assert _live_source_quality(url) == "unknown", url
    for url in ("https://www.ox.ac.uk/", "https://www.nhs.uk/x".replace("nhs.uk", "gov.uk"), "https://who.int/news", "https://web.mit.edu/"):
        assert _live_source_quality(url) == "official", url
