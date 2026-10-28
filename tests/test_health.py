from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["index_loaded"] == "false"
    assert body["embedding_model_loaded"] == "false"
    assert body["nli_model_loaded"] == "false"
    assert body["analysis_ready"] == "false"
