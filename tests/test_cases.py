from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_case_accepts_related_text_url_and_image_inputs() -> None:
    created = client.post("/api/cases")
    assert created.status_code == 201
    case_id = created.json()["case_id"]

    text = client.post(
        f"/api/cases/{case_id}/inputs/text",
        json={"text": "Your account will be closed today."},
    )
    url = client.post(
        f"/api/cases/{case_id}/inputs/url",
        json={"url": "https://example.test/verify"},
    )
    image = client.post(
        f"/api/cases/{case_id}/inputs/image",
        files={"file": ("alert.png", b"temporary-image", "image/png")},
    )

    assert text.status_code == 201
    assert url.status_code == 201
    assert image.status_code == 201
    body = image.json()
    assert body["case_id"] == case_id
    assert [item["input_type"] for item in body["inputs"]] == [
        "text",
        "url",
        "image",
    ]
    assert body["inputs"][2]["size_bytes"] == len(b"temporary-image")
    assert "raw_content" not in body["inputs"][2]


def test_case_routes_return_not_found_for_unknown_case() -> None:
    response = client.post(
        "/api/cases/missing/inputs/text",
        json={"text": "A claim."},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "case not found."


def test_case_can_be_deleted() -> None:
    created = client.post("/api/cases")
    case_id = created.json()["case_id"]

    deleted = client.delete(f"/api/cases/{case_id}")
    fetched = client.get(f"/api/cases/{case_id}")

    assert deleted.status_code == 204
    assert fetched.status_code == 404
