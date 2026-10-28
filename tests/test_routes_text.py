from fastapi.testclient import TestClient

import app.api.routes_text as routes_text
from app.main import app
from app.services.claim_extractor import ExtractedClaim
from app.services.pipeline import PipelineClaim, PipelineEvidence


class FailingPipeline:
    def analyze(self, text: str):
        raise RuntimeError("model unavailable")


class WorkingPipeline:
    def analyze(self, text: str):
        claim = ExtractedClaim("claim_1", "NASA published a climate report", ["NASA"], "factual")
        evidence = PipelineEvidence(
            passage_id="nasa-climate_p1",
            title="Climate evidence",
            source_name="NASA",
            source_url="https://example.com",
            published_date=None,
            passage="Climate evidence passage.",
            retrieval_similarity=0.8,
            nli_label="entailment",
            nli_score=0.9,
            source_type="official_scientific_reference",
            source_quality="official",
            retrieved_date="2026-10-27",
        )
        return [PipelineClaim(claim, "supported", [evidence], "Evidence supports the claim.")], "supported"


def test_analyze_text_endpoint_returns_documented_shape() -> None:
    original_get_pipeline = routes_text.get_pipeline
    routes_text.get_pipeline = lambda: FailingPipeline()
    response = TestClient(app).post(
        "/api/analyze/text",
        json={"text": "NASA published a climate report."},
    )
    routes_text.get_pipeline = original_get_pipeline

    assert response.status_code == 503
    assert response.json()["detail"] == "Text analysis components are unavailable."


def test_analyze_text_rejects_empty_input() -> None:
    response = TestClient(app).post("/api/analyze/text", json={"text": ""})

    assert response.status_code == 422


def test_analyze_text_endpoint_returns_evidence_response() -> None:
    original_get_pipeline = routes_text.get_pipeline
    routes_text.get_pipeline = lambda: WorkingPipeline()
    response = TestClient(app).post(
        "/api/analyze/text",
        json={"text": "NASA published a climate report."},
    )
    routes_text.get_pipeline = original_get_pipeline

    body = response.json()
    assert response.status_code == 200
    assert body["input_type"] == "text"
    assert body["extracted_text"] == "NASA published a climate report."
    assert body["claims"][0]["verification_status"] == "supported"
    assert body["claims"][0]["evidence"][0]["passage_id"] == "nasa-climate_p1"
    assert "likely supported" in body["claims"][0]["analysis"]


def test_url_endpoint_extracts_and_analyzes_page(monkeypatch) -> None:
    monkeypatch.setattr(
        routes_text,
        "fetch_url_text",
        lambda url: "NASA published a climate report.",
    )
    original_get_pipeline = routes_text.get_pipeline
    routes_text.get_pipeline = lambda: WorkingPipeline()
    response = TestClient(app).post(
        "/api/analyze/url",
        json={"url": "https://example.com/story"},
    )
    routes_text.get_pipeline = original_get_pipeline

    assert response.status_code == 200
    assert response.json()["input_type"] == "url"
    assert response.json()["extracted_text"] == "NASA published a climate report."
    assert response.json()["claims"][0]["claim_text"] == "NASA published a climate report"


def test_image_endpoint_extracts_and_analyzes_ocr_text(monkeypatch) -> None:
    monkeypatch.setattr(
        routes_text,
        "extract_image_text",
        lambda image: "NASA published a climate report.",
    )
    original_get_pipeline = routes_text.get_pipeline
    routes_text.get_pipeline = lambda: WorkingPipeline()
    response = TestClient(app).post(
        "/api/analyze/image",
        files={"file": ("claim.png", b"fake-image", "image/png")},
    )
    routes_text.get_pipeline = original_get_pipeline

    assert response.status_code == 200
    assert response.json()["input_type"] == "image"
    assert response.json()["extracted_text"] == "NASA published a climate report."
    assert response.json()["claims"][0]["verification_status"] == "supported"
