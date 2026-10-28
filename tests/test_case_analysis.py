from fastapi.testclient import TestClient
from io import BytesIO
from types import SimpleNamespace
from PIL import Image

import app.api.routes_cases as routes_cases
from app.main import app
from app.services.claim_extractor import ExtractedClaim
from app.services.pipeline import PipelineClaim, PipelineEvidence
from app.services.case_store import build_image_input


class StubPipeline:
    def analyze(self, text: str):
        claim = ExtractedClaim("claim_1", text, ["Bank"], "factual")
        evidence = PipelineEvidence(
            passage_id="evidence-1",
            title="Contradictory guidance",
            source_name="Example source",
            source_url="https://example.test/evidence",
            published_date=None,
            passage="The claim is contradicted.",
            retrieval_similarity=0.9,
            nli_label="contradiction",
            nli_score=0.9,
            source_type="official_guidance",
            source_quality="official",
            retrieved_date="2026-10-27",
        )
        return [PipelineClaim(claim, "refuted", [evidence], "Contradicted.")], "refuted"


def test_case_analysis_fuses_claim_url_and_correlation_signals(monkeypatch) -> None:
    monkeypatch.setattr(routes_cases, "case_store", routes_cases.CaseStore())
    monkeypatch.setattr(routes_cases, "get_pipeline", lambda: StubPipeline())
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = TestClient(app)
    case_id = client.post("/api/cases").json()["case_id"]
    client.post(
        f"/api/cases/{case_id}/inputs/text",
        json={"text": "Bank asks you to verify at http://192.168.1.1/login"},
    )
    client.post(
        f"/api/cases/{case_id}/inputs/url",
        json={"url": "http://192.168.1.1/login"},
    )

    result = client.post(f"/api/cases/{case_id}/analyze")

    assert result.status_code == 200
    body = result.json()
    assert body["verdict"] == "deceptive"
    assert body["risk_score"] >= 0.6
    assert body["claims"][0]["verification_status"] == "refuted"
    assert any(signal["signal_type"] == "phishing_risk" for signal in body["signals"])
    assert body["deception_risk"] == "high"
    assert body["intent_assessment"] == "not_determinable_from_content"


def test_case_analysis_marks_strong_phishing_url_deceptive(monkeypatch) -> None:
    monkeypatch.setattr(routes_cases, "case_store", routes_cases.CaseStore())
    monkeypatch.setattr(routes_cases, "get_pipeline", lambda: StubPipeline())
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = TestClient(app)
    case_id = client.post("/api/cases").json()["case_id"]
    client.post(
        f"/api/cases/{case_id}/inputs/url",
        json={"url": "http://192.168.1.1/login/verify-bank-account"},
    )

    result = client.post(f"/api/cases/{case_id}/analyze")

    assert result.status_code == 200
    body = result.json()
    assert body["verdict"] == "deceptive"
    assert body["risk_score"] >= 0.6


def test_case_analysis_marks_refuted_text_deceptive(monkeypatch) -> None:
    monkeypatch.setattr(routes_cases, "case_store", routes_cases.CaseStore())
    monkeypatch.setattr(routes_cases, "get_pipeline", lambda: StubPipeline())
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = TestClient(app)
    case_id = client.post("/api/cases").json()["case_id"]
    client.post(
        f"/api/cases/{case_id}/inputs/text",
        json={"text": "The bank claim is false."},
    )

    result = client.post(f"/api/cases/{case_id}/analyze")

    assert result.status_code == 200
    body = result.json()
    assert body["verdict"] == "deceptive"
    assert body["risk_score"] >= 0.6


def test_case_analysis_uses_message_context_for_linked_url(monkeypatch) -> None:
    monkeypatch.setattr(routes_cases, "case_store", routes_cases.CaseStore())
    monkeypatch.setattr(routes_cases, "get_pipeline", lambda: StubPipeline())
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = TestClient(app)
    case_id = client.post("/api/cases").json()["case_id"]
    client.post(
        f"/api/cases/{case_id}/inputs/text",
        json={"text": "bank acc expiry click to update pin"},
    )
    client.post(
        f"/api/cases/{case_id}/inputs/url",
        json={"url": "https://web.fishy.com/"},
    )

    result = client.post(f"/api/cases/{case_id}/analyze")

    assert result.status_code == 200
    body = result.json()
    assert body["verdict"] == "deceptive"
    assert body["risk_score"] >= 0.6


def test_case_analysis_runs_ocr_and_verifies_image_text(monkeypatch) -> None:
    monkeypatch.setattr(routes_cases, "case_store", routes_cases.CaseStore())
    monkeypatch.setattr(routes_cases, "get_pipeline", lambda: StubPipeline())
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(
        "app.services.adapters.read_image",
        lambda content: SimpleNamespace(text="The bank asks for your PIN."),
    )
    client = TestClient(app)
    image_buffer = BytesIO()
    Image.new("RGB", (2, 2), "white").save(image_buffer, format="PNG")
    case = routes_cases.case_store.create()
    routes_cases.case_store.add_input(
        case.case_id,
        build_image_input(image_buffer.getvalue(), file_name="test.png", mime_type="image/png"),
    )

    result = client.post(f"/api/cases/{case.case_id}/analyze")

    assert result.status_code == 200
    body = result.json()
    assert body["claims"][0]["source_input_ids"] == [body["inputs"][0]["input_id"]]
    metadata = next(s for s in body["signals"] if s["signal_type"] == "image_metadata")
    assert metadata["details"]["ocr_text"] == "The bank asks for your PIN."


def test_image_without_readable_text_needs_review(monkeypatch) -> None:
    monkeypatch.setattr(routes_cases, "case_store", routes_cases.CaseStore())
    monkeypatch.setattr(routes_cases, "get_pipeline", lambda: StubPipeline())
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(
        "app.services.adapters.read_image",
        lambda content: (_ for _ in ()).throw(ValueError("no text")),
    )
    image_buffer = BytesIO()
    Image.new("RGB", (2, 2), "white").save(image_buffer, format="PNG")
    case = routes_cases.case_store.create()
    routes_cases.case_store.add_input(
        case.case_id,
        build_image_input(image_buffer.getvalue(), file_name="scene.png", mime_type="image/png"),
    )

    result = TestClient(app).post(f"/api/cases/{case.case_id}/analyze")

    assert result.status_code == 200
    assert result.json()["verdict"] == "needs_review"


def _ai(verdict: str, confidence: float = 0.9):
    from app.services.investigator import InvestigatorResult

    return InvestigatorResult(
        verdict=verdict, risk_score=0.9 if verdict == "deceptive" else 0.0, confidence=confidence,
        summary="AI opinion.", reasons=("reason",), website_findings=(), provider="groq",
    )


def _run_with_ai(monkeypatch, pipeline, ai_result, text: str, url: str | None = None):
    import app.services.case_analysis as case_analysis

    calls = []

    def fake_investigate(**kwargs):
        calls.append(kwargs)
        return ai_result

    monkeypatch.setattr(case_analysis, "investigate_case", fake_investigate)
    monkeypatch.setattr(routes_cases, "case_store", routes_cases.CaseStore())
    monkeypatch.setattr(routes_cases, "get_pipeline", lambda: pipeline)
    client = TestClient(app)
    case_id = client.post("/api/cases").json()["case_id"]
    client.post(f"/api/cases/{case_id}/inputs/text", json={"text": text})
    if url:
        client.post(f"/api/cases/{case_id}/inputs/url", json={"url": url})
    return client.post(f"/api/cases/{case_id}/analyze").json(), calls


class _SupportedPipeline:
    def analyze(self, text: str):
        claim = ExtractedClaim("claim_1", text, [], "factual")
        evidence = PipelineEvidence(
            passage_id="e1", title="T", source_name="S", source_url="https://example.test/a", published_date=None,
            passage="It is so.", retrieval_similarity=0.9, nli_label="entailment", nli_score=0.95,
            source_type="web_search", source_quality="reputable", retrieved_date="2026-10-28",
        )
        return [PipelineClaim(claim, "supported", [evidence], "ok")], "supported"


def test_ai_that_calls_a_genuine_verdict_deceptive_raises_it_to_needs_review(monkeypatch) -> None:
    body, calls = _run_with_ai(monkeypatch, _SupportedPipeline(), _ai("deceptive"), "Water is wet in most conditions")
    assert calls  # the system said genuine, so the AI was consulted
    assert body["verdict"] == "needs_review"
    assert body["deception_risk"] == "elevated"
    assert "AI investigator judged this likely deceptive" in body["decision_basis"]
    assert body["ai_investigation"]["escalated"] is True
    assert any("raised from genuine to needs_review" in item for item in body["limitations"])


def test_low_confidence_ai_opinion_does_not_escalate(monkeypatch) -> None:
    body, _ = _run_with_ai(monkeypatch, _SupportedPipeline(), _ai("deceptive", confidence=0.4), "Water is wet in most conditions")
    assert body["verdict"] == "genuine" and body["ai_investigation"]["escalated"] is False


def test_ai_can_never_clear_or_lower_a_verdict(monkeypatch) -> None:
    body, _ = _run_with_ai(monkeypatch, _InsufficientPipeline(), _ai("genuine", 0.99), "Some unverifiable statement about an event")
    assert body["verdict"] == "needs_review"  # an AI "genuine" does not clear it


def test_ai_alone_cannot_make_a_case_deceptive(monkeypatch) -> None:
    body, _ = _run_with_ai(monkeypatch, _InsufficientPipeline(), _ai("deceptive", 0.99), "Some unverifiable statement about an event")
    assert body["verdict"] == "needs_review"  # raised caution is capped at needs_review


def test_ai_is_not_called_when_the_verdict_is_already_deceptive(monkeypatch) -> None:
    body, calls = _run_with_ai(
        monkeypatch, StubPipeline(), _ai("genuine"), "Bank asks you to verify at http://192.168.1.1/login", "http://192.168.1.1/login"
    )
    assert body["verdict"] == "deceptive"
    assert calls == []  # saves a request: the AI could not make this any more cautious
    assert body["ai_investigation"] is None


class _InsufficientPipeline:
    """Returns retrieved evidence but no verdict, as happens for unverifiable or hoax claims."""

    def __init__(self, status: str = "insufficient_evidence") -> None:
        self._status = status

    def analyze(self, text: str):
        claim = ExtractedClaim("claim_1", text, [], "factual")
        evidence = PipelineEvidence(
            passage_id="e1", title="Related article", source_name="Example", source_url="https://example.test/a",
            published_date=None, passage="A related passage.", retrieval_similarity=0.8, nli_label="neutral",
            nli_score=0.9, source_type="web_search", source_quality="unknown", retrieved_date="2026-10-28",
        )
        return [PipelineClaim(claim, self._status, [evidence], "reason")], self._status


def _case_verdict(monkeypatch, pipeline, text: str) -> dict:
    monkeypatch.setattr(routes_cases, "case_store", routes_cases.CaseStore())
    monkeypatch.setattr(routes_cases, "get_pipeline", lambda: pipeline)
    client = TestClient(app)
    case_id = client.post("/api/cases").json()["case_id"]
    client.post(f"/api/cases/{case_id}/inputs/text", json={"text": text})
    return client.post(f"/api/cases/{case_id}/analyze").json()


def test_retrieved_evidence_alone_never_makes_a_case_genuine(monkeypatch) -> None:
    body = _case_verdict(monkeypatch, _InsufficientPipeline(), "Some unverifiable statement about an event")
    assert body["evidence"]  # evidence was retrieved...
    assert body["content_assessment"] == "insufficient_evidence"
    assert body["verdict"] == "needs_review"  # ...but it did not support anything
    assert body["deception_risk"] != "low"


def test_conflicting_evidence_never_reads_as_low_risk(monkeypatch) -> None:
    body = _case_verdict(monkeypatch, _InsufficientPipeline("conflicting_evidence"), "A contested statement about a topic")
    assert body["verdict"] == "needs_review"
    assert body["deception_risk"] != "low"


def test_ai_is_not_consulted_when_genuine_rests_on_a_trusted_link(monkeypatch) -> None:
    import app.services.adapters as adapters
    from app.services.reputation import Reputation

    rep = Reputation(tranco={"paypal.com": 20})
    monkeypatch.setattr(adapters, "default_reputation", lambda: rep)
    body, calls = _run_with_ai(
        monkeypatch, _InsufficientPipeline(), _ai("deceptive", 0.99),
        "Your PayPal receipt for your recent payment is available.", "https://www.paypal.com/myaccount/transactions",
    )
    assert body["verdict"] == "genuine"  # cleared by hard domain evidence
    assert calls == []  # so the AI was never asked, and could not wrongly escalate it
