from fastapi.testclient import TestClient

from app.main import app


def test_audio_input_is_accepted_as_an_async_case_item() -> None:
    client = TestClient(app)
    case_id = client.post("/api/cases").json()["case_id"]

    response = client.post(
        f"/api/cases/{case_id}/inputs/audio",
        files={"file": ("message.wav", b"audio-bytes", "audio/wav")},
        data={"transcript": "Your account needs verification."},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] in {"processing", "completed"}
    assert body["inputs"][0]["input_type"] == "audio"
    assert body["inputs"][0]["text"] == "Your account needs verification."
