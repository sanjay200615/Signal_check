"""Domain-level phishing indicators: lookalikes, brand abuse, free hosting."""

from dataclasses import dataclass
import math
import re

from app.services.reputation import Reputation, is_multi_tenant, registered_domain, subdomain_of

# Platforms where anyone can publish a page under a subdomain.
FREE_HOSTING_SUFFIXES = frozenset(
    {
        "r2.dev", "pages.dev", "workers.dev", "vercel.app", "netlify.app", "github.io", "gitlab.io",
        "herokuapp.com", "web.app", "firebaseapp.com", "glitch.me", "replit.app", "replit.dev",
        "repl.co", "onrender.com", "fly.dev", "surge.sh", "weebly.com", "wixsite.com", "blogspot.com",
        "wordpress.com", "webflow.io", "azurewebsites.net", "ngrok.io", "ngrok-free.app",
        "trycloudflare.com", "my.id", "biz.id", "web.id", "godaddysites.com", "sites.google.com",
        "000webhostapp.com", "infinityfreeapp.com", "rf.gd", "epizy.com", "gitbook.io", "notion.site",
        "carrd.co", "canva.site", "framer.website", "bubbleapps.io", "strikingly.com", "jimdosite.com",
    }
)
URL_SHORTENERS = frozenset(
    {
        "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly", "is.gd", "buff.ly", "cutt.ly",
        "rebrand.ly", "shorturl.at", "rb.gy", "tiny.cc", "t.ly", "lnkd.in", "s.id",
    }
)

_HOMOGLYPHS = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s"})
_HEX_LABEL = re.compile(r"^(?:[a-z]{1,4}-)?[0-9a-f]{16,}$")
_POPULAR_RANK = 50_000
_LOOKALIKE_WEIGHTS = {"homoglyph_lookalike": 0.7, "typosquat_lookalike": 0.55, "same_label_other_tld": 0.5}


@dataclass(frozen=True)
class DomainFinding:
    indicator: str
    weight: float


def is_free_hosting(host: str) -> bool:
    """True for platforms where anyone can publish a page under a subdomain."""
    host = host.lower()
    if any(host == suffix or host.endswith("." + suffix) for suffix in FREE_HOSTING_SUFFIXES):
        return True
    return is_multi_tenant(host)


def is_shortener(host: str) -> bool:
    return host.lower() in URL_SHORTENERS


def looks_random(label: str) -> bool:
    """True for machine-generated labels such as ``pub-28b6157a0bea4e08``."""
    label = label.lower()
    if _HEX_LABEL.match(label):
        return True
    if len(label) < 14 or not any(ch.isdigit() for ch in label):
        return False
    counts = {ch: label.count(ch) for ch in set(label)}
    entropy = -sum(n / len(label) * math.log2(n / len(label)) for n in counts.values())
    return entropy >= 3.4


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance with adjacent transpositions counted as one edit."""
    previous2: list[int] | None = None
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if previous2 is not None and i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                value = min(value, previous2[j - 2] + 1)
            current.append(value)
        previous2, previous = previous, current
    return previous[-1]


def _normalize_label(label: str) -> str:
    return label.lower().translate(_HOMOGLYPHS).replace("rn", "m").replace("vv", "w").replace("-", "")


def analyze_domain(host: str, reputation: Reputation | None) -> list[DomainFinding]:
    """Return indicators for a hostname. Without ``reputation`` only structural checks run."""
    host = host.lower()
    findings: list[DomainFinding] = []
    domain = registered_domain(host)
    label = domain.split(".")[0] if domain else ""
    subdomain = subdomain_of(host)

    if host.startswith("xn--") or ".xn--" in host:
        findings.append(DomainFinding("punycode_domain", 0.3))

    free = is_free_hosting(host)
    if free:
        findings.append(DomainFinding("free_hosting_platform", 0.25))
        first_label = host.split(".")[0]
        if looks_random(first_label):
            findings.append(DomainFinding("random_subdomain_label", 0.2))

    if reputation is not None and reputation.host_in_feed(host):
        findings.append(DomainFinding("known_phishing_host", 0.9))
    if reputation is None or not domain:
        return findings

    official = {d for domains in reputation.brands.values() for d in domains}
    brand_hit = False
    for brand, brand_domains in reputation.brands.items():
        if domain in brand_domains:
            continue
        if len(brand) >= 4 and re.search(rf"(?<![a-z0-9]){re.escape(brand)}(?![a-z0-9])", subdomain):
            findings.append(DomainFinding("brand_in_subdomain", 0.5))
            brand_hit = True
            break
        if len(brand) >= 4 and brand in label:
            findings.append(DomainFinding("brand_in_domain_label", 0.35))
            brand_hit = True
            break
    if free and not brand_hit:
        for brand in reputation.brands:
            if len(brand) >= 4 and brand in host.split(".")[0]:
                findings.append(DomainFinding("brand_on_free_hosting", 0.35))
                break

    if domain not in official and not _is_popular(domain, reputation) and not free:
        lookalike = _lookalike(label, domain, reputation)
        if lookalike:
            findings.append(DomainFinding(lookalike, _LOOKALIKE_WEIGHTS[lookalike]))
    return findings


def _is_popular(domain: str, reputation: Reputation) -> bool:
    rank = reputation.rank(domain)
    return rank is not None and rank <= _POPULAR_RANK


def _lookalike(label: str, domain: str, reputation: Reputation) -> str | None:
    if len(label) < 5:
        return None
    normalized = _normalize_label(label)
    for target_label, target_domain in reputation.top_labels.items():
        if target_domain == domain:
            continue
        if target_label == label.lower():
            return "same_label_other_tld"
        if normalized == _normalize_label(target_label) and normalized != label.lower().replace("-", ""):
            return "homoglyph_lookalike"
        if abs(len(label) - len(target_label)) <= 1 and edit_distance(label.lower(), target_label) == 1:
            return "typosquat_lookalike"
    return None
