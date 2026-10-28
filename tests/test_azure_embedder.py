"""Offline tests for the Azure embedding adapter."""

from types import SimpleNamespace

import pytest

from src.app.azure_embeddings import (
    EMBEDDING_DIMENSIONS,
    AzureOpenAIEmbedder,
)


class FakeEmbeddings:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[dict[str, object]] = []

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        self.calls.append({"model": model, "input": input})
        data = [
            SimpleNamespace(index=index, embedding=vector)
            for index, vector in enumerate(self.vectors[: len(input)])
        ]
        return SimpleNamespace(data=list(reversed(data)))


class FakeClient:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.embeddings = FakeEmbeddings(vectors)


def vector(start: float) -> list[float]:
    return [start + index / 1000 for index in range(EMBEDDING_DIMENSIONS)]


def make_embedder(
    vectors: list[list[float]],
) -> tuple[AzureOpenAIEmbedder, FakeClient]:
    fake_client = FakeClient(vectors)
    embedder = AzureOpenAIEmbedder(
        client=fake_client,  # type: ignore[arg-type]
        deployment="text-embedding-3-small",
    )
    return embedder, fake_client


def test_document_embeddings_preserve_input_order() -> None:
    first = vector(1.0)
    second = vector(2.0)
    embedder, fake_client = make_embedder([first, second])

    result = embedder.embed_documents(["first document", "second document"])

    assert result == [first, second]
    assert fake_client.embeddings.calls[0]["model"] == "text-embedding-3-small"
    assert fake_client.embeddings.calls[0]["input"] == [
        "first document",
        "second document",
    ]


def test_query_embedding_returns_one_vector() -> None:
    expected = vector(3.0)
    embedder, _ = make_embedder([expected])

    result = embedder.embed_query("How does retrieval work?")

    assert result == expected
    assert len(result) == EMBEDDING_DIMENSIONS


def test_empty_document_batch_skips_api_call() -> None:
    embedder, fake_client = make_embedder([])

    assert embedder.embed_documents([]) == []
    assert fake_client.embeddings.calls == []


def test_blank_input_is_rejected_before_api_call() -> None:
    embedder, fake_client = make_embedder([])

    with pytest.raises(ValueError, match="non-empty string"):
        embedder.embed_documents(["valid text", "  "])

    assert fake_client.embeddings.calls == []


def test_single_string_is_not_accepted_as_a_batch() -> None:
    embedder, _ = make_embedder([])

    with pytest.raises(TypeError, match="sequence of strings"):
        embedder.embed_documents("this is one string")


def test_wrong_embedding_dimension_is_rejected() -> None:
    embedder, _ = make_embedder([[0.1, 0.2, 0.3]])

    with pytest.raises(RuntimeError, match="Embedding dimension mismatch"):
        embedder.embed_query("test")


def test_missing_embedding_result_is_rejected() -> None:
    embedder, _ = make_embedder([vector(1.0)])

    with pytest.raises(RuntimeError, match="every input"):
        embedder.embed_documents(["first", "second"])


def test_empty_deployment_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="deployment name"):
        AzureOpenAIEmbedder(
            client=FakeClient([]),  # type: ignore[arg-type]
            deployment=" ",
        )
