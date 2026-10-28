"""Pluggable evidence sources for the text verification pipeline."""

from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime
import html
import logging
import os
import re
import time
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit
from xml.etree import ElementTree

from app.services.retriever import EmbeddingModel, EvidenceIndex

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedEvidence:
    passage_id: str
    title: str
    source_name: str
    source_url: str | None
    published_date: str | None
    passage: str
    retrieval_similarity: float
    source_type: str
    source_quality: str
    retrieved_date: str
    provider: str
    source_quality_weight: float
    last_updated_date: str | None = None
    domain: str | None = None


class EvidenceProvider(Protocol):
    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]:
        """Return ranked evidence for a claim."""


class LocalEvidenceProvider:
    def __init__(
        self,
        evidence_index: EvidenceIndex,
        embedder: EmbeddingModel,
    ) -> None:
        self._evidence_index = evidence_index
        self._embedder = embedder

    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]:
        return [
            RetrievedEvidence(
                passage_id=passage.passage_id,
                title=passage.title,
                source_name=passage.source_name,
                source_url=passage.source_url,
                published_date=passage.published_date,
                passage=passage.text,
                retrieval_similarity=similarity,
                source_type=passage.source_type,
                source_quality=_quality_from_source_type(passage.source_type),
                retrieved_date=passage.retrieved_date,
                provider="local",
                source_quality_weight=_source_quality_weight(
                    _quality_from_source_type(passage.source_type)
                ),
                last_updated_date=passage.last_updated_date,
                domain=passage.domain,
            )
            for passage, similarity in self._evidence_index.search(
                query,
                self._embedder,
                top_k=top_k,
            )
        ]


class LiveNewsEvidenceProvider:
    """Search current news through Google's public RSS feed without an API key."""

    endpoint = "https://news.google.com/rss/search"

    def __init__(self, timeout: float = 20.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")
        self._timeout = timeout

    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]:
        if not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        import httpx

        try:
            response = httpx.get(
                self.endpoint,
                params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                headers={"User-Agent": "evidence-verification/0.1"},
                timeout=self._timeout,
            )
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
        except httpx.HTTPError as error:
            logger.warning(
                "evidence_search_failed provider=google_news query=%r error=%s",
                query,
                type(error).__name__,
            )
            raise RuntimeError("Live news search failed.") from error
        except ElementTree.ParseError as error:
            logger.warning(
                "evidence_rss_parse_failed provider=google_news query=%r error=%s",
                query,
                type(error).__name__,
            )
            raise RuntimeError("Live news search returned invalid RSS.") from error

        results: list[RetrievedEvidence] = []
        seen_urls: set[str] = set()
        raw_items = root.findall("./channel/item")
        for index, item in enumerate(raw_items[:top_k]):
            title = _xml_text(item.find("title"))
            url = _xml_text(item.find("link"))
            description = _xml_text(item.find("description"))
            source = _xml_text(item.find("source")) or _hostname(url)
            if not title or not url or not description:
                continue
            canonical_url = _canonical_url(url)
            if canonical_url and canonical_url in seen_urls:
                continue
            if canonical_url:
                seen_urls.add(canonical_url)
            passage = _strip_html(description)
            published_date = _parse_published_date(_xml_text(item.find("pubDate")))
            results.append(
                RetrievedEvidence(
                    passage_id=f"live_{index + 1}",
                    title=title,
                    source_name=source,
                    source_url=url,
                    published_date=published_date,
                    passage=passage,
                    retrieval_similarity=_query_similarity(query, title, passage),
                    source_type="live_news_result",
                    source_quality=_live_source_quality(url),
                    retrieved_date=date.today().isoformat(),
                    provider="live",
                    source_quality_weight=_source_quality_weight(
                        _live_source_quality(url)
                    ),
                    domain=_hostname(url),
                )
            )
        logger.info(
            "evidence_search provider=google_news query=%r http_status=%s "
            "raw_results=%d parsed_results=%d deduplicated_results=%d",
            query,
            getattr(response, "status_code", "unknown"),
            len(raw_items),
            len(results),
            len(seen_urls),
        )
        return results


class WikipediaEvidenceProvider:
    """Search Wikipedia for biographies and historical reference material."""

    endpoint = "https://en.wikipedia.org/w/rest.php/v1/search/page"

    def __init__(self, timeout: float = 20.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")
        self._timeout = timeout

    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]:
        if not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        import httpx

        try:
            response = httpx.get(
                self.endpoint,
                params={
                    "q": query,
                    "limit": top_k,
                },
                headers={"User-Agent": _wikimedia_user_agent()},
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as error:
            logger.warning(
                "evidence_search_failed provider=wikipedia query=%r error=%s",
                query,
                type(error).__name__,
            )
            raise RuntimeError("Wikipedia search failed.") from error

        pages = payload.get("pages", []) if isinstance(payload, dict) else []
        if not isinstance(pages, list):
            raise RuntimeError("Wikipedia search returned an invalid response.")

        results: list[RetrievedEvidence] = []
        for index, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            title = page.get("title")
            passage = page.get("excerpt")
            if isinstance(passage, str):
                passage = _strip_html(passage)
            key = page.get("key")
            url = f"https://en.wikipedia.org/wiki/{key}" if isinstance(key, str) else None
            if not all(
                isinstance(value, str) and value.strip()
                for value in (title, passage, url)
            ):
                continue
            results.append(
                RetrievedEvidence(
                    passage_id=f"wikipedia_{page.get('id', index + 1)}",
                    title=title,
                    source_name="Wikipedia",
                    source_url=url,
                    published_date=None,
                    passage=passage,
                    retrieval_similarity=_query_similarity(query, title, passage),
                    source_type="encyclopedia_reference",
                    source_quality="reputable",
                    retrieved_date=date.today().isoformat(),
                    provider="wikipedia",
                    source_quality_weight=_source_quality_weight("reputable"),
                    domain="wikipedia.org",
                )
            )
        return results


class DuckDuckGoEvidenceProvider:
    """General web search through DuckDuckGo. Needs no API key.

    Result bodies are real text snippets, which gives the NLI model far more to
    work with than a bare news headline.
    """

    _max_passage_chars = 1200

    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]:
        if not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        from ddgs import DDGS
        from ddgs.exceptions import DDGSException

        try:
            rows = DDGS().text(query, max_results=top_k)
        except DDGSException as error:
            logger.warning(
                "evidence_search_failed provider=duckduckgo query=%r error=%s",
                query,
                type(error).__name__,
            )
            raise RuntimeError("DuckDuckGo search failed.") from error

        results: list[RetrievedEvidence] = []
        for index, row in enumerate(rows):
            title = _strip_html(str(row.get("title") or ""))
            url = str(row.get("href") or "")
            passage = _strip_html(str(row.get("body") or ""))[: self._max_passage_chars]
            if not title or not url or not passage:
                continue
            quality = _live_source_quality(url)
            results.append(
                RetrievedEvidence(
                    passage_id=f"ddg_{index + 1}",
                    title=title,
                    source_name=_hostname(url),
                    source_url=url,
                    published_date=None,
                    passage=passage,
                    retrieval_similarity=_query_similarity(query, title, passage),
                    source_type="web_search",
                    source_quality=quality,
                    retrieved_date=date.today().isoformat(),
                    provider="duckduckgo",
                    source_quality_weight=_source_quality_weight(quality),
                    domain=_hostname(url),
                )
            )
        return results


class HybridEvidenceProvider:
    """Combine reference, current-news, and optional general-web evidence."""

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        include_tavily: bool | None = None,
        failure_cooldown_seconds: float = 60.0,
        cache: Any | None = None,
    ) -> None:
        if failure_cooldown_seconds < 0:
            raise ValueError("failure_cooldown_seconds must not be negative.")
        self._providers: list[EvidenceProvider] = [
            WikipediaEvidenceProvider(timeout=timeout),
            DuckDuckGoEvidenceProvider(),
            LiveNewsEvidenceProvider(),
        ]
        if cache is not None:
            from app.services.cache import CachedEvidenceProvider

            self._providers = [
                CachedEvidenceProvider(provider, cache) for provider in self._providers
            ]
        self._reference_cache: dict[str, list[RetrievedEvidence]] = {}
        self._reference_failures: dict[str, float] = {}
        self._failure_cooldown_seconds = failure_cooldown_seconds
        tavily_key_configured = bool(os.getenv("TAVILY_API_KEY"))
        if include_tavily is True or (include_tavily is None and tavily_key_configured):
            self._providers.append(TavilyEvidenceProvider(timeout=timeout))

    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]:
        results: list[RetrievedEvidence] = []
        seen_urls: set[str] = set()
        for provider in self._providers:
            try:
                if isinstance(getattr(provider, "inner", provider), WikipediaEvidenceProvider):
                    cache_key = _reference_query_key(query)
                    provider_results = self._reference_cache.get(cache_key)
                    failed_at = self._reference_failures.get(cache_key)
                    failure_is_recent = (
                        failed_at is not None
                        and time.monotonic() - failed_at < self._failure_cooldown_seconds
                    )
                    if provider_results is None and not failure_is_recent:
                        try:
                            provider_results = provider.search(query, top_k=top_k)
                        except RuntimeError:
                            self._reference_failures[cache_key] = time.monotonic()
                            logger.warning(
                                "hybrid_provider_source_failed source=WikipediaEvidenceProvider "
                                "query_key=%r cooldown_seconds=%.0f",
                                cache_key,
                                self._failure_cooldown_seconds,
                            )
                            provider_results = []
                        else:
                            self._reference_cache[cache_key] = provider_results
                    elif failure_is_recent:
                        logger.info(
                            "hybrid_provider_source_skipped source=WikipediaEvidenceProvider "
                            "query_key=%r reason=recent_failure",
                            cache_key,
                        )
                        provider_results = []
                    if provider_results is None:
                        provider_results = provider.search(query, top_k=top_k)
                        self._reference_cache[cache_key] = provider_results
                else:
                    provider_results = provider.search(query, top_k=top_k)
            except RuntimeError:
                logger.warning(
                    "hybrid_provider_source_failed source=%s",
                    type(provider).__name__,
                    exc_info=True,
                )
                continue
            for item in provider_results:
                key = _canonical_url(item.source_url) or (
                    f"{item.source_name}:{item.title}".lower()
                )
                if key in seen_urls:
                    continue
                seen_urls.add(key)
                results.append(item)
        return sorted(
            results,
            key=lambda item: (
                item.retrieval_similarity * item.source_quality_weight,
                item.source_quality_weight,
            ),
            reverse=True,
        )[: top_k * 2]


class TavilyEvidenceProvider:
    """Search Tavily results; requires TAVILY_API_KEY and never silently falls back."""

    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None, timeout: float = 15.0) -> None:
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self._api_key:
            raise ValueError(
                "TAVILY_API_KEY is required to enable live evidence retrieval."
            )
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")
        self._timeout = timeout

    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]:
        if not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        import httpx

        try:
            response = httpx.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": top_k,
                    "include_answer": False,
                    "include_raw_content": False,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise RuntimeError("Live evidence search failed.") from error

        raw_results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(raw_results, list):
            raise RuntimeError("Live evidence search returned an invalid response.")
        results: list[RetrievedEvidence] = []
        seen_urls: set[str] = set()
        for index, item in enumerate(raw_results):
            evidence = _parse_tavily_result(item, index)
            canonical_url = _canonical_url(evidence.source_url)
            if canonical_url and canonical_url in seen_urls:
                continue
            if canonical_url:
                seen_urls.add(canonical_url)
            results.append(evidence)
        return results


def _parse_tavily_result(item: Any, index: int) -> RetrievedEvidence:
    if not isinstance(item, dict):
        raise RuntimeError(f"Live evidence result {index} is invalid.")
    title = item.get("title")
    url = item.get("url")
    content = item.get("content")
    if not all(isinstance(value, str) and value.strip() for value in (title, url, content)):
        raise RuntimeError(f"Live evidence result {index} is missing required fields.")
    score = item.get("score", 0.0)
    if not isinstance(score, (int, float)):
        raise RuntimeError(f"Live evidence result {index} has an invalid score.")
    return RetrievedEvidence(
        passage_id=f"live_{index + 1}",
        title=title,
        source_name=url.split("/")[2] if "://" in url else url,
        source_url=url,
        published_date=None,
        passage=content,
        retrieval_similarity=float(score),
        source_type="live_web_result",
        source_quality="unknown",
        retrieved_date=__import__("datetime").date.today().isoformat(),
        provider="tavily",
        source_quality_weight=_source_quality_weight("unknown"),
    )


def _xml_text(element: ElementTree.Element | None) -> str:
    return "".join(element.itertext()).strip() if element is not None else ""


def _wikimedia_user_agent() -> str:
    """Wikimedia's API policy asks for a contact in the user agent; set WIKIMEDIA_CONTACT to provide one."""
    contact = os.getenv("WIKIMEDIA_CONTACT", "").strip() or "research@example.com"
    return f"MultimodalEvidenceResearch/0.1 (contact: {contact})"


def _strip_html(value: str) -> str:
    """Remove tags, decode entities, and collapse whitespace."""
    text = html.unescape(re.sub(r"<[^>]+>", "", value))
    return " ".join(text.replace(" ", " ").split())


def _hostname(url: str | None) -> str:
    return urlsplit(url or "").netloc.lower() or "unknown"


def _parse_published_date(value: str) -> str | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _query_similarity(query: str, title: str, passage: str) -> float:
    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    result_words = set(re.findall(r"[a-z0-9]+", f"{title} {passage}".lower()))
    if not query_words:
        return 0.0
    overlap = len(query_words & result_words) / len(query_words)
    return min(0.95, max(0.3, overlap))


def _reference_query_key(query: str) -> str:
    words = re.findall(r"[a-z0-9]+", query.lower())
    return " ".join(
        word
        for word in words
        if word not in {"is", "was", "are", "were", "still", "alive", "dead", "died", "death"}
    )


_REPUTABLE_DOMAINS = frozenset(
    {
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org", "theguardian.com", "nytimes.com",
        "washingtonpost.com", "nature.com", "science.org", "who.int",
        # fact-checking organisations
        "snopes.com", "factcheck.org", "politifact.com", "fullfact.org", "afp.com", "boomlive.in", "altnews.in",
        # encyclopedias and established science / health references
        "britannica.com", "scientificamerican.com", "nationalgeographic.com", "smithsonianmag.com",
        "newscientist.com", "sciencenews.org", "livescience.com", "space.com", "mayoclinic.org",
        "clevelandclinic.org", "hopkinsmedicine.org", "ourworldindata.org", "khanacademy.org", "merriam-webster.com",
        # established news organisations
        "cnn.com", "cbsnews.com", "nbcnews.com", "abcnews.go.com", "aljazeera.com", "dw.com", "france24.com",
        "economist.com", "wsj.com", "bloomberg.com", "ft.com", "time.com", "pbs.org", "theatlantic.com",
    }
)
_OFFICIAL_SUFFIXES = (".gov", ".edu", ".mil", ".int", ".ac.uk", ".gov.uk", ".gov.au", ".edu.au", ".gov.in", ".ac.in")
_AI_GENERATED_REFERENCE_HOSTS = {"grokipedia.com"}


def _live_source_quality(url: str) -> str:
    hostname = _hostname(url)
    if hostname in _AI_GENERATED_REFERENCE_HOSTS:
        return "context_only"
    if hostname.endswith(_OFFICIAL_SUFFIXES) or hostname.endswith(".wikipedia.org"):
        return "official"
    if any(hostname == domain or hostname.endswith("." + domain) for domain in _REPUTABLE_DOMAINS):
        return "reputable"
    return "unknown"


def _quality_from_source_type(source_type: str) -> str:
    if source_type.startswith("official_"):
        return "official"
    if source_type in {
        "reputable_news",
        "academic_reference",
        "encyclopedia_reference",
    }:
        return "reputable"
    if source_type == "scope_note":
        return "context_only"
    return "unknown"


def _source_quality_weight(source_quality: str) -> float:
    return {
        "official": 1.0,
        "reputable": 0.95,
        "unknown": 0.8,
        "context_only": 0.5,
    }.get(source_quality, 0.8)


def _canonical_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlsplit(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return url.strip().lower().rstrip("/")
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", "")
    )
