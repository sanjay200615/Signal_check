import json
from types import SimpleNamespace

import httpx
import pytest

import app.services.investigator as investigator
from app.services.cache import SqliteCache
from app.services.investigator import investigate_case, reset_rate_limit_state

VALID = {
    "verdict": "genuine", "risk_score": 0.1, "confidence": 0.8, "summary": "Looks fine.",
    "reasons": ["No risk indicators."], "website_findings": ["No page inspected."],
}


class Response:
    def __init__(self, status=200, headers=None, body=None):
        self.status_code = status
        self.headers = headers or {}
        self._body = body if body is not None else {"choices": [{"message": {"content": json.dumps(VALID)}}]}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._body


@pytest.fixture(autouse=True)
def groq_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "default_cache", lambda: None)
    reset_rate_limit_state()


def call(message="hello world"):
    return investigate_case(message_text=message, urls=[], claims=[], evidence=[], images=[])


def test_a_429_starts_a_cooldown_and_later_calls_make_no_request(monkeypatch):
    posts = []

    def post(*args, **kwargs):
        posts.append(1)
        return Response(429, {"retry-after": "42"})

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(RuntimeError, match="rate limit"):
        call("first")
    assert len(posts) == 1
    for _ in range(3):
        with pytest.raises(RuntimeError, match="rate-limited"):
            call("second")
    assert len(posts) == 1  # nothing was sent while cooling down


def test_the_cooldown_honours_retry_after_and_ends(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(investigator, "_clock", lambda: now[0])
    posts = []
    responses = [Response(429, {"retry-after": "10"}), Response()]
    monkeypatch.setattr(httpx, "post", lambda *a, **k: posts.append(1) or responses.pop(0))
    with pytest.raises(RuntimeError):
        call()
    now[0] += 5
    with pytest.raises(RuntimeError, match="rate-limited"):
        call()
    now[0] += 6  # 11 seconds after the 429
    assert call().verdict == "genuine"
    assert len(posts) == 2


def test_missing_or_absurd_retry_after_values_are_bounded():
    assert investigator._start_cooldown(None) == investigator.DEFAULT_COOLDOWN_SECONDS
    assert investigator._start_cooldown("not-a-number") == investigator.DEFAULT_COOLDOWN_SECONDS
    assert investigator._start_cooldown("999999") == investigator.MAX_COOLDOWN_SECONDS
    assert investigator._start_cooldown("0") == 1.0


def test_calls_are_spaced_by_the_minimum_interval(monkeypatch):
    monkeypatch.setenv("GROQ_MIN_INTERVAL_SECONDS", "3")
    now, slept = [100.0], []
    monkeypatch.setattr(investigator, "_clock", lambda: now[0])
    monkeypatch.setattr(investigator, "_sleep", lambda seconds: (slept.append(seconds), now.__setitem__(0, now[0] + seconds)))
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Response())
    call("one")
    call("two")  # immediately afterwards
    assert slept and slept[-1] == pytest.approx(3.0)


def test_an_identical_case_is_served_from_cache_without_a_request(monkeypatch):
    cache = SqliteCache(":memory:")
    monkeypatch.setattr(investigator, "default_cache", lambda: cache)
    posts = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: posts.append(1) or Response())
    first, second = call("same message"), call("same message")
    assert first == second and len(posts) == 1
    call("a different message")
    assert len(posts) == 2


def test_failed_requests_are_not_cached(monkeypatch):
    cache = SqliteCache(":memory:")
    monkeypatch.setattr(investigator, "default_cache", lambda: cache)
    responses = [Response(500), Response()]
    monkeypatch.setattr(httpx, "post", lambda *a, **k: responses.pop(0))
    with pytest.raises(RuntimeError):
        call("flaky")
    assert call("flaky").verdict == "genuine"


def test_the_prompt_is_compact(monkeypatch):
    sent = {}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: sent.update(k) or Response())
    big = "x" * 5000
    evidence = [{"evidence_id": str(i), "passage": big, "source_name": "S", "nli_label": "neutral", "nli_score": i / 20,
                 "source_quality": "unknown", "source_url": "https://e.test", "retrieval_score": 0.5} for i in range(20)]
    investigate_case(message_text="m", urls=[], claims=[], evidence=evidence, images=[{"ocr_text": big, "perceptual_hash": "ff"}])
    prompt = json.loads(sent["json"]["messages"][1]["content"])
    assert len(prompt["retrieved_evidence"]) == investigator.MAX_EVIDENCE
    assert all(len(e["passage"]) <= investigator.MAX_PASSAGE_CHARS for e in prompt["retrieved_evidence"])
    assert prompt["retrieved_evidence"][0]["nli_score"] == 0.95  # the strongest evidence is kept
    assert len(prompt["images"][0]["ocr_text"]) <= investigator.MAX_OCR_CHARS and "perceptual_hash" not in prompt["images"][0]


def test_non_public_hosts_are_not_fetched(monkeypatch):
    fetched = []
    monkeypatch.setattr(investigator, "fetch_url_text", lambda url, timeout: fetched.append(url) or "page")
    monkeypatch.setattr(investigator, "is_public_host", lambda host: host != "localhost")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Response())
    investigate_case(message_text="m", urls=["http://localhost:8000/secret", "https://public.example/"], claims=[], evidence=[])
    assert fetched == ["https://public.example/"]


def test_cache_key_ignores_per_case_random_ids(monkeypatch):
    cache = SqliteCache(":memory:")
    monkeypatch.setattr(investigator, "default_cache", lambda: cache)
    posts = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: posts.append(1) or Response())

    def evidence(case_id):
        return [{"evidence_id": f"{case_id}_p1", "passage": "Same passage.", "source_name": "S", "nli_label": "entailment"}]

    claims = [{"claim_text": "Insulin is a hormone", "verification_status": "supported"}]
    for case_id in ("input_aaa", "input_bbb"):
        investigate_case(message_text="m", urls=[], claims=claims, evidence=evidence(case_id))
    assert len(posts) == 1


def test_a_reply_with_no_website_findings_is_valid_when_there_is_no_page(monkeypatch):
    body = {"choices": [{"message": {"content": json.dumps({**VALID, "website_findings": []})}}]}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Response(body=body))
    assert call().website_findings == ()
    missing = {k: v for k, v in VALID.items() if k != "website_findings"}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Response(body={"choices": [{"message": {"content": json.dumps(missing)}}]}))
    reset_rate_limit_state()
    assert call("other").website_findings == ()


def test_malformed_findings_are_still_rejected(monkeypatch):
    body = {"choices": [{"message": {"content": json.dumps({**VALID, "website_findings": [1, 2]})}}]}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Response(body=body))
    with pytest.raises(RuntimeError, match="invalid details"):
        call()
