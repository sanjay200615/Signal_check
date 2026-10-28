import pytest

from app.services.cache import CachedEvidenceProvider, SqliteCache
from app.services.evidence_provider import HybridEvidenceProvider, RetrievedEvidence


def _evidence(title: str = "Paris") -> RetrievedEvidence:
    return RetrievedEvidence(
        passage_id="p1",
        title=title,
        source_name="Wikipedia",
        source_url="https://en.wikipedia.org/wiki/Paris",
        published_date=None,
        passage="Paris is the capital of France.",
        retrieval_similarity=0.9,
        source_type="encyclopedia_reference",
        source_quality="reputable",
        retrieved_date="2026-10-28",
        provider="wikipedia",
        source_quality_weight=0.95,
        domain="wikipedia.org",
    )


class FlakyProvider:
    def __init__(self, failures: int) -> None:
        self.calls = 0
        self._failures = failures

    def search(self, query: str, *, top_k: int):
        self.calls += 1
        if self.calls <= self._failures:
            raise RuntimeError("temporary outage")
        return [_evidence()]


def test_cache_round_trips_and_expires():
    cache = SqliteCache(":memory:")
    assert cache.get("ns", "k") is None
    cache.set("ns", "k", {"a": [1, 2]})
    assert cache.get("ns", "k") == {"a": [1, 2]}
    assert cache.get("other", "k") is None
    assert cache.get("ns", "k", ttl=-1) is None  # already older than the ttl
    cache.clear("ns")
    assert cache.get("ns", "k") is None


def test_second_search_is_served_from_cache():
    inner = FlakyProvider(failures=0)
    provider = CachedEvidenceProvider(inner, SqliteCache(":memory:"))
    first = provider.search("Paris  capital", top_k=3)
    second = provider.search("paris capital", top_k=3)  # same after normalisation
    assert inner.calls == 1
    assert first == second == [_evidence()]


def test_different_top_k_is_a_different_cache_entry():
    inner = FlakyProvider(failures=0)
    provider = CachedEvidenceProvider(inner, SqliteCache(":memory:"))
    provider.search("q", top_k=1)
    provider.search("q", top_k=2)
    assert inner.calls == 2


def test_transient_failures_are_retried_with_backoff():
    sleeps: list[float] = []
    inner = FlakyProvider(failures=2)
    provider = CachedEvidenceProvider(
        inner, SqliteCache(":memory:"), retries=2, backoff_seconds=1.0, sleep=sleeps.append
    )
    assert provider.search("q", top_k=3) == [_evidence()]
    assert inner.calls == 3
    assert sleeps == [1.0, 2.0]


def test_persistent_failure_raises_and_is_not_cached():
    inner = FlakyProvider(failures=99)
    cache = SqliteCache(":memory:")
    provider = CachedEvidenceProvider(inner, cache, retries=1, sleep=lambda _: None)
    with pytest.raises(RuntimeError):
        provider.search("q", top_k=3)
    assert inner.calls == 2
    inner._failures = 0  # the outage ends
    assert provider.search("q", top_k=3) == [_evidence()]


def test_hybrid_provider_with_cache_wraps_each_source():
    provider = HybridEvidenceProvider(cache=SqliteCache(":memory:"), include_tavily=False)
    assert all(hasattr(source, "inner") for source in provider._providers)


def test_cache_entries_from_an_older_logic_version_are_ignored(monkeypatch):
    import app.services.cache as cache_module

    cache = SqliteCache(":memory:")
    inner = FlakyProvider(failures=0)
    CachedEvidenceProvider(inner, cache).search("q", top_k=3)
    assert inner.calls == 1
    monkeypatch.setattr(cache_module, "EVIDENCE_CACHE_VERSION", cache_module.EVIDENCE_CACHE_VERSION + 1)
    CachedEvidenceProvider(inner, cache).search("q", top_k=3)
    assert inner.calls == 2  # the old entry was not reused
