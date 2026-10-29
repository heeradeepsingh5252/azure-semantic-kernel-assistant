"""API tests using fake services; no live Azure calls are made."""

from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from src.app.config import get_settings
from src.app.main import ApplicationServices, create_app, get_services
from src.app.models import (
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentChunk,
)


class FakeSearchStore:
    """Small in-memory substitute for Azure AI Search."""

    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []

    def upsert_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        incoming_sources = {chunk.source for chunk in chunks}
        self.chunks = [
            chunk for chunk in self.chunks
            if chunk.source not in incoming_sources
        ]
        self.chunks.extend(chunks)

    def delete_by_source(self, source: str) -> None:
        self.chunks = [
            chunk for chunk in self.chunks if chunk.source != source
        ]

    def list_sources(self) -> list[str]:
        return sorted({chunk.source for chunk in self.chunks})

    def hybrid_search(
        self, query: str, top_k: int = 5
    ) -> list[DocumentChunk]:
        return self.chunks[:top_k]

    def close(self) -> None:
        pass


class FakeAssistant:
    """Fake assistant that returns a structured answer without model calls."""

    async def answer(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            session_id=request.session_id,
            answer="Hybrid retrieval combines keyword and vector search [1].",
            citations=[
                Citation(
                    number=1,
                    source="sample.txt",
                    page=1,
                    chunk_id="sample-chunk-1",
                    text="Hybrid retrieval combines keyword and vector search.",
                )
            ],
            pii_detected=["EMAIL"] if "@" in request.question else [],
        )

    async def aclose(self) -> None:
        pass


@pytest.fixture
def api():
    """Create an isolated app with injectable fake services."""
    settings = get_settings(
        {
            "APP_MODE": "local",
            "MAX_UPLOAD_MB": "1",
            "ALLOWED_ORIGINS": "http://localhost:5173",
        }
    )
    search_store = FakeSearchStore()
    services = ApplicationServices(
        embedder=object(),
        search_store=search_store,
        assistant=FakeAssistant(),
    )
    application = create_app(settings=settings)
    application.dependency_overrides[get_services] = lambda: services

    with TestClient(application, raise_server_exceptions=False) as client:
        yield client, application, services

    application.dependency_overrides.clear()


def test_health_and_swagger_docs_are_available(api) -> None:
    client, _, _ = api

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200


def test_chat_returns_answer_citations_and_pii_types_only(api) -> None:
    client, _, _ = api

    response = client.post(
        "/chat",
        json={
            "session_id": "test-session",
            "question": "Please explain this for alice@example.com",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].endswith("[1].")
    assert body["citations"][0]["source"] == "sample.txt"
    assert body["citations"][0]["page"] == 1
    assert body["pii_detected"] == ["EMAIL"]
    assert "alice@example.com" not in response.text


def test_chat_validation_error_does_not_echo_submitted_values(api) -> None:
    client, _, _ = api

    response = client.post(
        "/chat",
        json={"session_id": "test-session", "question": ""},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "Request validation failed.",
        }
    }


def test_ingest_txt_and_list_documents(api) -> None:
    client, _, _ = api

    response = client.post(
        "/ingest",
        files={
            "file": (
                "sample.txt",
                b"Azure AI Search supports hybrid retrieval by combining "
                b"keyword search with vector similarity.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "sample.txt"
    assert body["pages"] == 1
    assert body["chunk_count"] >= 1

    documents_response = client.get("/documents")
    assert documents_response.status_code == 200
    assert documents_response.json() == {"sources": ["sample.txt"]}


def test_reupload_replaces_prior_chunks(api) -> None:
    client, _, services = api

    first = client.post(
        "/ingest",
        files={"file": ("sample.txt", b"First document version.", "text/plain")},
    )
    second = client.post(
        "/ingest",
        files={"file": ("sample.txt", b"Updated document version.", "text/plain")},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    stored = services.search_store.chunks
    assert stored
    assert {chunk.source for chunk in stored} == {"sample.txt"}
    assert "Updated document version." in " ".join(
        chunk.content for chunk in stored
    )


def test_ingest_rejects_unsupported_file_type(api) -> None:
    client, _, _ = api

    response = client.post(
        "/ingest",
        files={"file": ("script.exe", b"not a supported document", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_ingest_rejects_file_over_configured_limit(api) -> None:
    client, _, _ = api

    response = client.post(
        "/ingest",
        files={"file": ("large.txt", b"x" * (1024 * 1024 + 1), "text/plain")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_delete_document_removes_its_chunks(api) -> None:
    client, _, services = api

    response = client.post(
        "/ingest",
        files={"file": ("sample.txt", b"Text to be removed.", "text/plain")},
    )
    assert response.status_code == 200
    assert services.search_store.list_sources() == ["sample.txt"]

    deletion = client.delete("/documents/sample.txt")

    assert deletion.status_code == 200
    assert deletion.json() == {"deleted": True, "source": "sample.txt"}
    assert services.search_store.list_sources() == []


def test_delete_missing_document_returns_json_404(api) -> None:
    client, _, _ = api

    response = client.delete("/documents/missing.txt")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_cors_allows_only_configured_origin(api) -> None:
    client, _, _ = api

    allowed = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert allowed.headers.get("access-control-allow-origin") == (
        "http://localhost:5173"
    )

    denied = client.options(
        "/chat",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in denied.headers


def test_rate_limit_returns_429_after_configured_limit(api) -> None:
    client, application, _ = api
    application.state.rate_limit_max_requests = 1

    payload = {
        "session_id": "rate-test",
        "question": "What does hybrid retrieval combine?",
    }

    first = client.post("/chat", json=payload)
    second = client.post("/chat", json=payload)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers.get("retry-after") is not None
    assert second.json()["error"]["code"] == "rate_limit_exceeded"
