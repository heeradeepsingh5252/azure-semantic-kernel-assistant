import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from src.app.main import (
    _deny_public_demo_mutations,
    create_app,
    enforce_public_demo_chat_budget,
)


def test_local_mode_allows_document_mutations(monkeypatch):
    monkeypatch.delenv("PUBLIC_DEMO_READ_ONLY", raising=False)

    _deny_public_demo_mutations()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/ingest"),
        ("DELETE", "/documents/sample.txt"),
    ],
)
def test_public_demo_blocks_document_mutations(monkeypatch, method, path):
    monkeypatch.setenv("PUBLIC_DEMO_READ_ONLY", "true")
    client = TestClient(create_app())

    if method == "POST":
        response = client.post(
            path,
            files={"file": ("demo.txt", b"demo content", "text/plain")},
        )
    else:
        response = client.delete(path)

    assert response.status_code == 403


def test_public_demo_enforces_daily_chat_budget(monkeypatch):
    monkeypatch.setenv("PUBLIC_DEMO_READ_ONLY", "true")
    monkeypatch.setenv("PUBLIC_DEMO_MAX_CHAT_REQUESTS_PER_DAY", "1")

    app = create_app()
    request = Request({"type": "http", "app": app, "headers": []})

    enforce_public_demo_chat_budget(request)

    with pytest.raises(HTTPException) as error:
        enforce_public_demo_chat_budget(request)

    assert error.value.status_code == 429
