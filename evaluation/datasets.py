"""Dataset loaders. Public downloads are cached under data/eval_cache/."""

import csv
import io
import json
from pathlib import Path
import random
import zipfile

import httpx

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "evaluation"
CACHE_DIR = ROOT / "data" / "eval_cache"

OPENPHISH_FEED = "https://openphish.com/feed.txt"
TRANCO_TOP = "https://tranco-list.eu/top-1m.csv.zip"
_BENIGN_PATHS = ["/", "/login", "/account", "/signin", "/help", "/about", "/support", "/products"]


def load_claims() -> list[dict[str, str]]:
    return _load_json(EVAL_DIR / "claims.json")


def load_scam_cases() -> list[dict[str, object]]:
    return _load_json(EVAL_DIR / "scam_cases.json")


def load_phishing_urls(n_each: int = 150, seed: int = 7) -> list[dict[str, str]]:
    """Return a balanced phishing/benign URL set from OpenPhish and Tranco.

    Benign URLs are popular domains with a mix of plain and login-style paths,
    so path words such as "login" cannot be used as a shortcut.
    """
    rng = random.Random(seed)
    phishing = _cached_lines("openphish.txt", OPENPHISH_FEED)
    phishing = [url for url in phishing if url.startswith(("http://", "https://"))]
    domains = _tranco_domains(limit=2000)
    if len(phishing) < n_each or len(domains) < n_each:
        raise RuntimeError("Not enough URLs available to build the dataset.")
    rows = [{"url": url, "label": "deceptive"} for url in rng.sample(phishing, n_each)]
    for domain in rng.sample(domains, n_each):
        path = rng.choice(_BENIGN_PATHS)
        rows.append({"url": f"https://{domain}{path}", "label": "genuine"})
    rng.shuffle(rows)
    return rows


def _load_json(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def _cached_lines(name: str, url: str) -> list[str]:
    path = CACHE_DIR / name
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        response = httpx.get(url, follow_redirects=True, timeout=60.0)
        response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tranco_domains(limit: int) -> list[str]:
    path = CACHE_DIR / "tranco_top.txt"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        response = httpx.get(TRANCO_TOP, follow_redirects=True, timeout=120.0)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            with archive.open(archive.namelist()[0]) as handle:
                reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8"))
                domains = [row[1] for _, row in zip(range(limit), reader) if len(row) >= 2]
        path.write_text("\n".join(domains), encoding="utf-8")
    return path.read_text(encoding="utf-8").splitlines()


PHISHING_DB_LINKS = (
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt"
)


def load_phishingdb_urls(n: int = 150, seed: int = 11) -> list[str]:
    """Phishing URLs from Phishing.Database, an independent source from OpenPhish."""
    urls = [u for u in _cached_lines("phishingdb_links.txt", PHISHING_DB_LINKS) if u.startswith(("http://", "https://"))]
    return random.Random(seed).sample(urls, n)


def sample_lowrank_benign(ranks: dict[str, int], n: int = 150, seed: int = 13) -> list[str]:
    """Benign-site URLs from Tranco ranks 20k-100k: popular enough to be real, obscure enough to test lookalike logic."""
    pool = sorted(d for d, rank in ranks.items() if 20_000 <= rank <= 100_000)
    rng = random.Random(seed)
    return [f"https://{domain}{rng.choice(_BENIGN_PATHS)}" for domain in rng.sample(pool, n)]


def load_caption_pairs(n: int = 60) -> list[dict[str, object]]:
    """Real photos with a human caption each, from Flickr30k via the Hugging Face datasets server. Cached."""
    meta_path = CACHE_DIR / "flickr_pairs.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not meta_path.exists():
        response = httpx.get(
            "https://datasets-server.huggingface.co/first-rows",
            params={"dataset": "nlphuji/flickr30k", "config": "TEST", "split": "test"},
            timeout=60.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        pairs = []
        for index, row in enumerate(response.json()["rows"]):
            data = row["row"]
            image_path = CACHE_DIR / f"flickr_{index}.jpg"
            image_path.write_bytes(httpx.get(data["image"]["src"], timeout=60.0, follow_redirects=True).content)
            pairs.append({"file": image_path.name, "caption": data["caption"][0]})
        meta_path.write_text(json.dumps(pairs), encoding="utf-8")
    pairs = json.loads(meta_path.read_text(encoding="utf-8"))[:n]
    return [{"image": (CACHE_DIR / p["file"]).read_bytes(), "caption": p["caption"]} for p in pairs]
