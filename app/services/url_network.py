"""Network-backed URL signals: domain age (RDAP) and short-link expansion.

Both are best-effort. Any failure returns ``None`` so the signal is simply
absent. Results are cached. Set ``URL_NETWORK_CHECKS=off`` to disable.
"""

from datetime import datetime, timezone
import ipaddress
import logging
import os
import socket
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit

import httpx

from app.services.cache import SqliteCache

logger = logging.getLogger(__name__)

IANA_RDAP_BOOTSTRAP = "https://data.iana.org/rdap/dns.json"
AGE_TTL_SECONDS = 7 * 24 * 3600
_USER_AGENT = {"User-Agent": "SignalCheck/0.1"}


def network_checks_enabled() -> bool:
    return os.getenv("URL_NETWORK_CHECKS", "on").strip().lower() not in {"off", "0", "false", "no"}


def is_public_host(host: str, resolver: Callable[..., Any] = socket.getaddrinfo) -> bool:
    """False if ``host`` resolves to a private, loopback, link-local or otherwise internal address."""
    try:
        addresses = {info[4][0] for info in resolver(host, None)}
    except (OSError, UnicodeError):
        return False
    if not addresses:
        return False
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
            or ip.is_reserved or ip.is_unspecified
        ):
            return False
    return True


def _rdap_base(tld: str, cache: SqliteCache | None, get: Callable[..., Any]) -> str | None:
    mapping = cache.get("rdap_bootstrap", "dns", ttl=AGE_TTL_SECONDS) if cache else None
    if mapping is None:
        response = get(IANA_RDAP_BOOTSTRAP, timeout=10.0, follow_redirects=True, headers=_USER_AGENT)
        response.raise_for_status()
        mapping = {
            suffix: urls[0]
            for services, urls in response.json()["services"]
            for suffix in services
            if urls
        }
        if cache:
            cache.set("rdap_bootstrap", "dns", mapping)
    return mapping.get(tld.lower())


def domain_age_days(
    domain: str,
    *,
    cache: SqliteCache | None = None,
    get: Callable[..., Any] = httpx.get,
    now: datetime | None = None,
) -> int | None:
    """Age of a registered domain in days from its RDAP registration event, or None."""
    if not domain or "." not in domain or _is_ip_literal(domain):
        return None
    if cache is not None:
        cached = cache.get("domain_age", domain, ttl=AGE_TTL_SECONDS)
        if cached is not None:
            registered = cached.get("registered")
            return _days_since(registered, now) if registered else None
    registered: str | None = None
    try:
        base = _rdap_base(domain.rsplit(".", 1)[-1], cache, get)
        if base:
            response = get(
                f"{base.rstrip('/')}/domain/{domain}",
                timeout=8.0,
                follow_redirects=True,
                headers=_USER_AGENT,
            )
            if response.status_code == 200:
                for event in response.json().get("events", []):
                    if event.get("eventAction") == "registration":
                        registered = event.get("eventDate")
                        break
            elif response.status_code != 404:
                return None  # transient failure: do not cache
    except (httpx.HTTPError, ValueError, KeyError) as error:
        logger.info("rdap_failed domain=%s error=%s", domain, type(error).__name__)
        return None
    if cache is not None:
        cache.set("domain_age", domain, {"registered": registered})
    return _days_since(registered, now) if registered else None


def _is_ip_literal(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _days_since(timestamp: str | None, now: datetime | None) -> int | None:
    if not timestamp:
        return None
    try:
        registered = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if registered.tzinfo is None:
        registered = registered.replace(tzinfo=timezone.utc)
    return max(0, ((now or datetime.now(timezone.utc)) - registered).days)


def expand_short_url(
    url: str,
    *,
    max_hops: int = 5,
    timeout: float = 6.0,
    request: Callable[..., Any] = httpx.head,
    resolver: Callable[..., Any] = socket.getaddrinfo,
) -> str | None:
    """Follow redirects manually and return the final URL, or None if it cannot be resolved safely.

    Each hop must be http(s) on a standard port and resolve to a public address.
    """
    current = url
    for _ in range(max_hops + 1):
        parts = urlsplit(current)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return None
        if parts.port not in (None, 80, 443) or not is_public_host(parts.hostname, resolver):
            return None
        try:
            response = request(current, timeout=timeout, follow_redirects=False, headers=_USER_AGENT)
        except httpx.HTTPError:
            return None
        location = response.headers.get("location")
        if response.status_code in (301, 302, 303, 307, 308) and location:
            current = urljoin(current, location)
            continue
        return current
    return None
