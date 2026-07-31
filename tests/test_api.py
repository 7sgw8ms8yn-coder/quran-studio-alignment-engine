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
def test_quran_lookup_requires_authentication() -> None:
    with TestClient(app) as test_client:
        response = test_client.get("/quran/1/1")

        assert response.status_code == 401


def test_quran_search_requires_authentication() -> None:
    with TestClient(app) as test_client:
        response = test_client.get(
            "/quran/search",
            params={"query": "الحمد لله"},
        )

        assert response.status_code == 401


def test_quran_corpus_loads_during_startup() -> None:
    with TestClient(app) as test_client:
        response = test_client.get("/health")

        assert response.status_code == 200

        payload = response.json()

        assert payload["status"] == "not_ready"
        assert payload["corpus_loaded"] is True
        assert payload["model_loaded"] is False


def test_speech_model_status_requires_authentication() -> None:
    with TestClient(app) as test_client:
        response = test_client.get("/speech/model-status")

        assert response.status_code == 401


def test_health_reports_speech_model_unloaded() -> None:
    with TestClient(app) as test_client:
        response = test_client.get("/health")

        assert response.status_code == 200

        payload = response.json()

        assert payload["corpus_loaded"] is True
        assert payload["model_loaded"] is False
        assert payload["status"] == "not_ready"


def test_quran_audio_alignment_requires_authentication() -> None:
    with TestClient(app) as test_client:
        response = test_client.post(
            "/quran/align-audio",
            files={
                "file": (
                    "recitation.mp3",
                    b"temporary-test-data",
                    "audio/mpeg",
                )
            },
        )

        assert response.status_code == 401
