"""Keyless reputation data: popular-site ranking, phishing feeds, and brand domains.

Everything here is downloaded from free public sources on first use and cached
under ``data/reputation/``. If a source cannot be reached the corresponding
data is simply empty, so lookups degrade to "no information" rather than
failing the analysis.
"""

from dataclasses import dataclass, field
import csv
import io
import logging
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import zipfile

import httpx
import tldextract

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "reputation"

TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
OPENPHISH_URL = "https://openphish.com/feed.txt"
PHISHING_DB_DOMAINS_URL = (
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-domains-ACTIVE.txt"
)

TRANCO_LIMIT = 100_000
FEED_TTL_SECONDS = 12 * 3600
TRANCO_TTL_SECONDS = 7 * 24 * 3600

_extractor = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
_private_extractor = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)

# Brand keyword -> registered domains that legitimately belong to the brand.
BRAND_DOMAINS: dict[str, frozenset[str]] = {
    "paypal": frozenset({"paypal.com", "paypal.me"}),
    "amazon": frozenset({"amazon.com", "amazon.in", "amazon.co.uk", "amazon.de", "amazon.ca", "amazon.com.au", "amazon.co.jp", "a.co", "amzn.to"}),
    "apple": frozenset({"apple.com", "icloud.com"}),
    "microsoft": frozenset({"microsoft.com", "live.com", "office.com", "outlook.com", "microsoftonline.com"}),
    "google": frozenset({"google.com", "gmail.com", "youtube.com", "goo.gl"}),
    "netflix": frozenset({"netflix.com"}),
    "facebook": frozenset({"facebook.com", "fb.com", "messenger.com"}),
    "instagram": frozenset({"instagram.com"}),
    "whatsapp": frozenset({"whatsapp.com", "wa.me"}),
    "linkedin": frozenset({"linkedin.com"}),
    "twitter": frozenset({"twitter.com", "x.com", "t.co"}),
    "github": frozenset({"github.com", "githubusercontent.com", "github.io"}),
    "dropbox": frozenset({"dropbox.com"}),
    "dhl": frozenset({"dhl.com", "dhl.de"}),
    "fedex": frozenset({"fedex.com"}),
    "ups": frozenset({"ups.com"}),
    "usps": frozenset({"usps.com"}),
    "sbi": frozenset({"sbi.co.in", "onlinesbi.sbi", "onlinesbi.com", "sbicard.com"}),
    "hdfc": frozenset({"hdfcbank.com", "hdfc.com"}),
    "icici": frozenset({"icicibank.com"}),
    "axis": frozenset({"axisbank.com"}),
    "paytm": frozenset({"paytm.com"}),
    "phonepe": frozenset({"phonepe.com"}),
    "chase": frozenset({"chase.com"}),
    "wellsfargo": frozenset({"wellsfargo.com"}),
    "citibank": frozenset({"citi.com", "citibank.com"}),
    "bankofamerica": frozenset({"bankofamerica.com"}),
    "coinbase": frozenset({"coinbase.com"}),
    "binance": frozenset({"binance.com"}),
    "wikipedia": frozenset({"wikipedia.org", "wikimedia.org"}),
    "spotify": frozenset({"spotify.com"}),
    "adobe": frozenset({"adobe.com"}),
    "irs": frozenset({"irs.gov"}),
    "roblox": frozenset({"roblox.com"}),
    "metamask": frozenset({"metamask.io"}),
    "discord": frozenset({"discord.com", "discord.gg"}),
    "steam": frozenset({"steampowered.com", "steamcommunity.com"}),
    "telegram": frozenset({"telegram.org", "t.me"}),
    "tiktok": frozenset({"tiktok.com"}),
    "snapchat": frozenset({"snapchat.com"}),
    "ebay": frozenset({"ebay.com"}),
    "walmart": frozenset({"walmart.com"}),
    "docusign": frozenset({"docusign.com", "docusign.net"}),
    "zoom": frozenset({"zoom.us"}),
    "outlook": frozenset({"outlook.com", "office.com", "live.com", "microsoft.com"}),
    "dana": frozenset({"dana.id"}),
    "gcash": frozenset({"gcash.com"}),
    "santander": frozenset({"santander.com"}),
    "hsbc": frozenset({"hsbc.com"}),
    "barclays": frozenset({"barclays.co.uk"}),
}


def registered_domain(host: str) -> str:
    """Return the registrable domain of ``host`` ("" for IP addresses or bare names)."""
    extracted = _extractor(host)
    return extracted.top_domain_under_public_suffix


def is_multi_tenant(host: str) -> bool:
    """True if ``host`` is a subdomain on a platform where anyone can publish (github.io, vercel.app...).

    The Public Suffix List's private section lists such platforms; the bundled
    snapshot is used, so this needs no network.
    """
    public = _extractor(host).top_domain_under_public_suffix
    private = _private_extractor(host)
    return bool(public) and private.top_domain_under_public_suffix != public and bool(private.subdomain or private.domain != public.split(".")[0])


def subdomain_of(host: str) -> str:
    return _extractor(host).subdomain


@dataclass
class Reputation:
    """Lookup tables used by the URL assessor."""

    tranco: dict[str, int] = field(default_factory=dict)
    phishing_urls: frozenset[str] = frozenset()
    phishing_hosts: frozenset[str] = frozenset()
    brands: dict[str, frozenset[str]] = field(default_factory=lambda: dict(BRAND_DOMAINS))

    def rank(self, domain: str) -> int | None:
        return self.tranco.get(domain)

    def is_trusted(self, domain: str, *, max_rank: int = 10_000) -> bool:
        rank = self.tranco.get(domain)
        return rank is not None and rank <= max_rank

    def in_feed(self, url: str) -> bool:
        return normalize_feed_url(url) in self.phishing_urls

    def host_in_feed(self, host: str) -> bool:
        return host.lower() in self.phishing_hosts

    @property
    def top_labels(self) -> dict[str, str]:
        """Label -> registered domain for the most popular sites (used for lookalikes)."""
        cached = getattr(self, "_top_labels", None)
        if cached is None:
            cached = {}
            for domain, rank in sorted(self.tranco.items(), key=lambda item: item[1])[:1000]:
                label = domain.split(".")[0]
                if len(label) >= 5:
                    cached.setdefault(label, domain)
            for domains in self.brands.values():
                for domain in domains:
                    cached.setdefault(domain.split(".")[0], domain)
            self._top_labels = cached
        return cached


def normalize_feed_url(url: str) -> str:
    parts = urlsplit(url.strip())
    return f"{parts.netloc.lower()}{parts.path.rstrip('/')}{'?' + parts.query if parts.query else ''}"


def _fresh(path: Path, ttl: float) -> bool:
    return path.exists() and time.time() - path.stat().st_mtime < ttl


def _download(url: str, timeout: float = 90.0) -> httpx.Response:
    response = httpx.get(url, follow_redirects=True, timeout=timeout, headers={"User-Agent": "SignalCheck/0.1"})
    response.raise_for_status()
    return response


def _load_tranco(directory: Path) -> dict[str, int]:
    path = directory / "tranco_top.csv"
    try:
        if not _fresh(path, TRANCO_TTL_SECONDS):
            directory.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(_download(TRANCO_URL, 120.0).content)) as archive:
                with archive.open(archive.namelist()[0]) as handle:
                    reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8"))
                    rows = [row for _, row in zip(range(TRANCO_LIMIT), reader) if len(row) >= 2]
            path.write_text("\n".join(f"{rank},{domain}" for rank, domain in rows), encoding="utf-8")
    except (httpx.HTTPError, OSError, zipfile.BadZipFile) as error:
        logger.warning("tranco_unavailable error=%s", type(error).__name__)
        if not path.exists():
            return {}
    ranks: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rank, _, domain = line.partition(",")
        if rank.isdigit() and domain:
            ranks[domain] = int(rank)
    return ranks


def _load_lines(directory: Path, name: str, url: str, ttl: float) -> list[str]:
    path = directory / name
    try:
        if not _fresh(path, ttl):
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text(_download(url).text, encoding="utf-8")
    except (httpx.HTTPError, OSError) as error:
        logger.warning("feed_unavailable name=%s error=%s", name, type(error).__name__)
        if not path.exists():
            return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_reputation(directory: Path = DATA_DIR) -> Reputation:
    """Download (or reuse cached) reputation data. Never raises on network failure."""
    openphish = _load_lines(directory, "openphish.txt", OPENPHISH_URL, FEED_TTL_SECONDS)
    domains = _load_lines(directory, "phishing_domains.txt", PHISHING_DB_DOMAINS_URL, 24 * 3600)
    urls = frozenset(normalize_feed_url(line) for line in openphish if line.startswith(("http://", "https://")))
    hosts = frozenset(
        {urlsplit(line).netloc.lower() for line in openphish if line.startswith(("http://", "https://"))}
        | {line.lower() for line in domains if not line.startswith("#")}
    )
    reputation = Reputation(tranco=_load_tranco(directory), phishing_urls=urls, phishing_hosts=hosts)
    logger.info(
        "reputation_loaded tranco=%d feed_urls=%d feed_hosts=%d",
        len(reputation.tranco),
        len(urls),
        len(hosts),
    )
    return reputation


_default: Reputation | None = None


def default_reputation() -> Reputation:
    """Shared lazily-loaded instance used by the application."""
    global _default
    if os.getenv("REPUTATION_DATA", "on").strip().lower() in {"off", "0", "false", "no"}:
        return Reputation()
    if _default is None:
        _default = load_reputation()
    return _default
