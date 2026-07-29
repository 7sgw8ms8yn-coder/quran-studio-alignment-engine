from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "not_ready"
    assert payload["model_loaded"] is False
    assert payload["corpus_loaded"] is False


def test_job_endpoint_requires_authentication() -> None:
    response = client.post("/jobs")

    assert response.status_code == 401