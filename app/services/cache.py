"""Persistent response cache and retry wrapper for evidence providers."""

from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import sqlite3
from threading import Lock
import time
from typing import Any, Callable

from app.services.evidence_provider import EvidenceProvider, RetrievedEvidence

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 7 * 24 * 3600
# Cached evidence carries derived fields (source quality and weight). Bump this whenever the logic that computes
# them changes, so stale entries are ignored instead of silently keeping the old classification.
EVIDENCE_CACHE_VERSION = 2


class SqliteCache:
    """Small JSON key-value store keyed by (namespace, key) with a per-entry TTL."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._connection = sqlite3.connect(self._path, check_same_thread=False)
        with self._lock, self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, "
                "stored_at REAL NOT NULL, PRIMARY KEY (namespace, key))"
            )

    def get(self, namespace: str, key: str, *, ttl: float | None = None) -> Any | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT value, stored_at FROM cache WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
        if row is None:
            return None
        value, stored_at = row
        if ttl is not None and time.time() - stored_at > ttl:
            return None
        return json.loads(value)

    def set(self, namespace: str, key: str, value: Any) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO cache (namespace, key, value, stored_at) VALUES (?, ?, ?, ?)",
                (namespace, key, json.dumps(value), time.time()),
            )

    def clear(self, namespace: str | None = None) -> None:
        with self._lock, self._connection:
            if namespace is None:
                self._connection.execute("DELETE FROM cache")
            else:
                self._connection.execute("DELETE FROM cache WHERE namespace = ?", (namespace,))

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class CachedEvidenceProvider:
    """Cache successful searches and retry transient failures with backoff.

    Failures are never cached, so a temporary outage cannot poison later runs.
    """

    def __init__(
        self,
        inner: EvidenceProvider,
        cache: SqliteCache,
        *,
        namespace: str | None = None,
        ttl: float = DEFAULT_TTL_SECONDS,
        retries: int = 2,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if retries < 0:
            raise ValueError("retries must not be negative.")
        if backoff_seconds < 0 or ttl <= 0:
            raise ValueError("backoff_seconds must not be negative and ttl must be positive.")
        self.inner = inner
        self._cache = cache
        self._namespace = f"{namespace or type(inner).__name__}:v{EVIDENCE_CACHE_VERSION}"
        self._ttl = ttl
        self._retries = retries
        self._backoff = backoff_seconds
        self._sleep = sleep

    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]:
        key = f"{top_k}|{' '.join(query.split()).lower()}"
        cached = self._cache.get(self._namespace, key, ttl=self._ttl)
        if cached is not None:
            logger.info("evidence_cache_hit provider=%s query=%r", self._namespace, query)
            return [RetrievedEvidence(**item) for item in cached]

        last_error: RuntimeError | None = None
        for attempt in range(self._retries + 1):
            try:
                results = self.inner.search(query, top_k=top_k)
            except RuntimeError as error:
                last_error = error
                if attempt < self._retries:
                    self._sleep(self._backoff * (2**attempt))
                continue
            self._cache.set(self._namespace, key, [asdict(item) for item in results])
            return results
        assert last_error is not None
        raise last_error


def default_cache() -> SqliteCache | None:
    """Open the shared cache unless EVIDENCE_CACHE=off."""
    if os.getenv("EVIDENCE_CACHE", "on").strip().lower() in {"off", "0", "false", "no"}:
        return None
    root = Path(__file__).resolve().parents[2]
    path = str(os.getenv("EVIDENCE_CACHE_PATH", root / "data" / "cache" / "evidence.sqlite"))
    if path not in _shared_caches:
        _shared_caches[path] = SqliteCache(path)
    return _shared_caches[path]


_shared_caches: dict[str, SqliteCache] = {}
