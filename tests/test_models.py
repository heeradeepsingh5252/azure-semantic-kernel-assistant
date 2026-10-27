"""Validation tests for the core application models."""

import pytest
from pydantic import ValidationError

from src.app.models import ChatRequest, ChatResponse, Citation, DocumentChunk


def test_document_chunk_accepts_valid_metadata() -> None:
    chunk = DocumentChunk(
        id="document-1-chunk-1",
        chunk_id="chunk-1",
        content="  Semantic Kernel connects AI components.  ",
        source="guide.txt",
        page=1,
        embedding=[0.1, 0.2, 0.3],
    )

    assert chunk.content == "Semantic Kernel connects AI components."
    assert chunk.source == "guide.txt"
    assert chunk.page == 1


def test_citation_rejects_zero_number() -> None:
    with pytest.raises(ValidationError):
        Citation(
            number=0,
            source="guide.txt",
            page=1,
            chunk_id="chunk-1",
            text="Supporting passage.",
        )


def test_chat_request_strips_question_and_defaults_top_k() -> None:
    request = ChatRequest(
        session_id="session-1",
        question="  What is Semantic Kernel?  ",
    )

    assert request.question == "What is Semantic Kernel?"
    assert request.top_k == 5


@pytest.mark.parametrize("question", ["", "   "])
def test_chat_request_rejects_empty_question(question: str) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(session_id="session-1", question=question)


def test_chat_request_rejects_overlong_question() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            session_id="session-1",
            question="x" * 4001,
        )


@pytest.mark.parametrize("top_k", [0, 21])
def test_chat_request_rejects_bad_top_k(top_k: int) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            session_id="session-1",
            question="Explain retrieval.",
            top_k=top_k,
        )


def test_chat_response_contains_structured_citations() -> None:
    response = ChatResponse(
        session_id="session-1",
        answer="Semantic Kernel connects AI components. [1]",
        citations=[
            Citation(
                number=1,
                source="guide.pdf",
                page=2,
                chunk_id="chunk-7",
                text="Semantic Kernel connects AI components.",
            )
        ],
        pii_detected=["EMAIL_ADDRESS"],
    )

    assert response.citations[0].number == 1
    assert response.citations[0].source == "guide.pdf"
    assert response.pii_detected == ["EMAIL_ADDRESS"]


def test_chat_response_default_lists_are_independent() -> None:
    first = ChatResponse(session_id="session-1", answer="First answer")
    second = ChatResponse(session_id="session-2", answer="Second answer")

    first.pii_detected.append("EMAIL_ADDRESS")

    assert second.pii_detected == []