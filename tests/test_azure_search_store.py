"""Offline tests for Azure AI Search storage and retrieval."""

from types import SimpleNamespace

import pytest
from azure.core.exceptions import ResourceNotFoundError

from src.app.azure_search_store import (
    EMBEDDING_DIMENSIONS,
    VECTOR_FIELD_NAME,
    AzureSearchStore,
    SearchStoreError,
)
from src.app.models import DocumentChunk


def make_vector(value: float = 0.1) -> list[float]:
    return [value] * EMBEDDING_DIMENSIONS


class FakeEmbedder:
    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(list(texts))
        return [make_vector() for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return make_vector(0.2)


class FakeIndexClient:
    def __init__(self, exists: bool = False) -> None:
        self.exists = exists
        self.created_indexes = []
        self.checked_names: list[str] = []
        self.closed = False

    def get_index(self, name: str):
        self.checked_names.append(name)
        if not self.exists:
            raise ResourceNotFoundError(message="Index not found")
        return SimpleNamespace(name=name)

    def create_index(self, index):
        self.created_indexes.append(index)
        self.exists = True
        return index

    def close(self) -> None:
        self.closed = True


class FakeSearchClient:
    def __init__(self) -> None:
        self.upload_calls: list[list[dict[str, object]]] = []
        self.delete_calls: list[list[dict[str, str]]] = []
        self.search_calls: list[dict[str, object]] = []
        self.search_rows: list[dict[str, object]] = []
        self.upload_results: list[SimpleNamespace] | None = None
        self.closed = False

    def upload_documents(self, *, documents):
        self.upload_calls.append(documents)
        if self.upload_results is not None:
            return self.upload_results
        return [
            SimpleNamespace(succeeded=True, key=doc["id"])
            for doc in documents
        ]

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return iter(self.search_rows)

    def delete_documents(self, *, documents):
        self.delete_calls.append(documents)
        return [
            SimpleNamespace(succeeded=True, key=doc["id"])
            for doc in documents
        ]

    def close(self) -> None:
        self.closed = True


def build_store(
    *,
    index_exists: bool = False,
) -> tuple[AzureSearchStore, FakeIndexClient, FakeSearchClient, FakeEmbedder]:
    index_client = FakeIndexClient(exists=index_exists)
    search_client = FakeSearchClient()
    embedder = FakeEmbedder()
    store = AzureSearchStore(
        index_client=index_client,  # type: ignore[arg-type]
        search_client=search_client,  # type: ignore[arg-type]
        index_name="test-index",
        embedder=embedder,
    )
    return store, index_client, search_client, embedder


def make_chunk(
    *,
    content: str = "Azure Search combines keyword and vector search.",
    source: str = "sample.txt",
    page: int | None = 1,
    embedding: list[float] | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        id="chunk-001",
        chunk_id="chunk-001",
        content=content,
        source=source,
        page=page,
        embedding=embedding,
    )


def test_ensure_index_creates_missing_index() -> None:
    store, index_client, _, _ = build_store()

    store.ensure_index()

    assert index_client.checked_names == ["test-index"]
    assert len(index_client.created_indexes) == 1
    assert index_client.created_indexes[0].name == "test-index"


def test_ensure_index_keeps_existing_index() -> None:
    store, index_client, _, _ = build_store(index_exists=True)

    store.ensure_index()

    assert index_client.checked_names == ["test-index"]
    assert index_client.created_indexes == []


def test_upsert_chunks_embeds_and_preserves_metadata() -> None:
    store, _, search_client, embedder = build_store()
    chunk = make_chunk()

    count = store.upsert_chunks([chunk])

    assert count == 1
    assert embedder.document_batches == [
        ["Azure Search combines keyword and vector search."]
    ]
    stored = search_client.upload_calls[0][0]
    assert stored["id"] == "chunk-001"
    assert stored["source"] == "sample.txt"
    assert stored["page"] == 1
    assert stored["chunk_id"] == "chunk-001"
    assert len(stored[VECTOR_FIELD_NAME]) == EMBEDDING_DIMENSIONS


def test_upsert_empty_batch_does_not_call_clients() -> None:
    store, _, search_client, embedder = build_store()

    assert store.upsert_chunks([]) == 0
    assert search_client.upload_calls == []
    assert embedder.document_batches == []


def test_upsert_rejects_wrong_embedding_dimension() -> None:
    store, _, search_client, _ = build_store()

    with pytest.raises(SearchStoreError, match="unexpected dimension"):
        store.upsert_chunks([make_chunk(embedding=[0.1, 0.2])])

    assert search_client.upload_calls == []


def test_upsert_reports_partial_indexing_failures() -> None:
    store, _, search_client, _ = build_store()
    search_client.upload_results = [
        SimpleNamespace(succeeded=False, key="chunk-001")
    ]

    with pytest.raises(SearchStoreError, match="failed to index"):
        store.upsert_chunks([make_chunk()])


def test_hybrid_search_uses_keyword_and_vector_queries() -> None:
    store, _, search_client, embedder = build_store()
    search_client.search_rows = [
        {
            "id": "chunk-002",
            "chunk_id": "chunk-002",
            "content": "Hybrid retrieval combines keyword and vector search.",
            "source": "guide.pdf",
            "page": 4,
        }
    ]

    chunks = store.hybrid_search("  What is hybrid retrieval?  ", top_k=3)

    assert len(chunks) == 1
    assert chunks[0].source == "guide.pdf"
    assert chunks[0].page == 4
    assert chunks[0].chunk_id == "chunk-002"
    assert embedder.queries == ["What is hybrid retrieval?"]

    call = search_client.search_calls[0]
    assert call["search_text"] == "What is hybrid retrieval?"
    assert call["top"] == 3
    assert len(call["vector_queries"]) == 1


@pytest.mark.parametrize(
    ("query", "top_k"),
    [
        ("", 5),
        ("   ", 5),
        ("question", 0),
        ("question", 21),
        ("question", True),
    ],
)
def test_hybrid_search_rejects_invalid_arguments(
    query: str,
    top_k: int,
) -> None:
    store, _, search_client, _ = build_store()

    with pytest.raises(ValueError):
        store.hybrid_search(query, top_k=top_k)

    assert search_client.search_calls == []


def test_delete_by_source_escapes_quotes_and_deletes_all_matches() -> None:
    store, _, search_client, _ = build_store()
    search_client.search_rows = [{"id": "a"}, {"id": "b"}]

    count = store.delete_by_source("O'Brien.pdf")

    assert count == 2
    assert "O''Brien.pdf" in search_client.search_calls[0]["filter"]
    assert search_client.delete_calls == [
        [{"id": "a"}, {"id": "b"}]
    ]


def test_list_sources_returns_unique_sorted_names() -> None:
    store, _, search_client, _ = build_store()
    search_client.search_rows = [
        {"source": "zeta.txt"},
        {"source": "guide.pdf"},
        {"source": "zeta.txt"},
    ]

    assert store.list_sources() == ["guide.pdf", "zeta.txt"]


def test_close_releases_both_search_clients() -> None:
    store, index_client, search_client, _ = build_store()

    store.close()

    assert index_client.closed is True
    assert search_client.closed is True
