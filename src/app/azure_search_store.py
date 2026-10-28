"""Azure AI Search index management, ingestion, and hybrid retrieval."""

import os
from collections.abc import Sequence

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv

from src.app.interfaces import Embedder, SearchStore
from src.app.models import DocumentChunk

EMBEDDING_DIMENSIONS = 1536
VECTOR_FIELD_NAME = "content_vector"
VECTOR_PROFILE_NAME = "content-vector-profile"
HNSW_ALGORITHM_NAME = "content-hnsw"
MAX_TOP_K = 20
INDEX_BATCH_SIZE = 1000


class SearchStoreError(RuntimeError):
    """Raised when an Azure AI Search operation fails."""


def build_search_index(index_name: str) -> SearchIndex:
    """Build the index definition without making a network request."""
    if not index_name or not index_name.strip():
        raise ValueError("index_name must not be empty.")

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="page",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="chunk_id",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SearchField(
            name=VECTOR_FIELD_NAME,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            retrievable=False,
            stored=False,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=HNSW_ALGORITHM_NAME,
                parameters=HnswParameters(metric="cosine"),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME,
                algorithm_configuration_name=HNSW_ALGORITHM_NAME,
            )
        ],
    )

    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
    )


class AzureSearchStore(SearchStore):
    """Store document chunks and retrieve them with hybrid search."""

    def __init__(
        self,
        *,
        index_client: SearchIndexClient,
        search_client: SearchClient,
        index_name: str,
        embedder: Embedder,
    ) -> None:
        if not index_name.strip():
            raise ValueError("Index name must not be empty.")

        self._index_client = index_client
        self._search_client = search_client
        self._index_name = index_name.strip()
        self._embedder = embedder

    @classmethod
    def from_environment(cls, embedder: Embedder) -> "AzureSearchStore":
        """Create clients using settings from environment or local .env."""
        load_dotenv()

        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "").strip()
        api_key = os.getenv("AZURE_SEARCH_API_KEY", "").strip()
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "").strip()

        missing = [
            name
            for name, value in (
                ("AZURE_SEARCH_ENDPOINT", endpoint),
                ("AZURE_SEARCH_API_KEY", api_key),
                ("AZURE_SEARCH_INDEX_NAME", index_name),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing Azure Search configuration: " + ", ".join(missing)
            )

        credential = AzureKeyCredential(api_key)
        index_client = SearchIndexClient(
            endpoint=endpoint,
            credential=credential,
        )
        search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential,
        )

        store = cls(
            index_client=index_client,
            search_client=search_client,
            index_name=index_name,
            embedder=embedder,
        )
        try:
            store.ensure_index()
        except Exception:
            store.close()
            raise

        return store

    def ensure_index(self) -> None:
        """Create the index if it does not exist."""
        try:
            self._index_client.get_index(self._index_name)
            return
        except ResourceNotFoundError:
            pass
        except HttpResponseError as exc:
            raise SearchStoreError(
                "Could not check whether the Azure Search index exists."
            ) from exc

        try:
            self._index_client.create_index(
                build_search_index(self._index_name)
            )
        except HttpResponseError as exc:
            raise SearchStoreError(
                "Could not create the Azure Search index."
            ) from exc

    def upsert_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        """Embed chunks when needed and upload them to the index."""
        if not chunks:
            return 0

        vectors_by_position: dict[int, list[float]] = {}
        missing_positions = [
            position
            for position, chunk in enumerate(chunks)
            if chunk.embedding is None
        ]

        if missing_positions:
            try:
                generated = self._embedder.embed_documents(
                    [chunks[position].content for position in missing_positions]
                )
            except Exception:
                # Preserve the originating error while avoiding document text
                # in messages or logs.
                raise

            if len(generated) != len(missing_positions):
                raise SearchStoreError(
                    "The embedding adapter returned an incorrect vector count."
                )

            for position, vector in zip(
                missing_positions, generated, strict=True
            ):
                vectors_by_position[position] = vector

        documents: list[dict[str, object]] = []
        for position, chunk in enumerate(chunks):
            vector = (
                chunk.embedding
                if chunk.embedding is not None
                else vectors_by_position[position]
            )

            if len(vector) != EMBEDDING_DIMENSIONS:
                raise SearchStoreError(
                    "A document embedding has an unexpected dimension."
                )

            document: dict[str, object] = {
                "id": chunk.id,
                "content": chunk.content,
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                VECTOR_FIELD_NAME: vector,
            }
            if chunk.page is not None:
                document["page"] = chunk.page

            documents.append(document)

        try:
            results = self._search_client.upload_documents(
                documents=documents
            )
        except HttpResponseError as exc:
            raise SearchStoreError(
                "Azure AI Search could not upload document chunks."
            ) from exc

        failures = [
            result for result in results if not result.succeeded
        ]
        if failures:
            failed_keys = [
                str(getattr(result, "key", "unknown"))
                for result in failures[:5]
            ]
            raise SearchStoreError(
                f"Azure AI Search failed to index {len(failures)} chunk(s). "
                f"Failed keys: {', '.join(failed_keys)}"
            )

        return len(documents)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[DocumentChunk]:
        """Combine keyword search and vector similarity in one query."""
        if not query or not query.strip():
            raise ValueError("Search query must not be empty.")
        if isinstance(top_k, bool) or not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")

        try:
            query_vector = self._embedder.embed_query(query.strip())
            if len(query_vector) != EMBEDDING_DIMENSIONS:
                raise SearchStoreError(
                    "The query embedding has an unexpected dimension."
                )

            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields=VECTOR_FIELD_NAME,
            )

            results = self._search_client.search(
                search_text=query.strip(),
                vector_queries=[vector_query],
                top=top_k,
                select=["id", "content", "source", "page", "chunk_id"],
            )

            chunks = []
            for record in results:
                page = record.get("page")
                chunks.append(
                    DocumentChunk(
                        id=str(record["id"]),
                        chunk_id=str(record.get("chunk_id") or record["id"]),
                        content=str(record["content"]),
                        source=str(record["source"]),
                        page=int(page) if page is not None else None,
                    )
                )
            return chunks
        except HttpResponseError as exc:
            raise SearchStoreError(
                "Azure AI Search hybrid retrieval failed."
            ) from exc

    def delete_by_source(self, source: str) -> int:
        """Delete all indexed chunks for one exact source filename."""
        if not source or not source.strip():
            raise ValueError("Source filename must not be empty.")

        escaped_source = source.strip().replace("'", "''")
        try:
            results = self._search_client.search(
                search_text="*",
                filter=f"source eq '{escaped_source}'",
                select=["id"],
                top=INDEX_BATCH_SIZE,
            )
            ids = [str(record["id"]) for record in results]

            deleted = 0
            for start in range(0, len(ids), INDEX_BATCH_SIZE):
                batch = ids[start : start + INDEX_BATCH_SIZE]
                statuses = self._search_client.delete_documents(
                    documents=[{"id": item} for item in batch]
                )
                failures = [
                    result for result in statuses if not result.succeeded
                ]
                if failures:
                    raise SearchStoreError(
                        f"Azure AI Search failed to delete "
                        f"{len(failures)} chunk(s) for this source."
                    )
                deleted += len(batch)

            return deleted
        except HttpResponseError as exc:
            raise SearchStoreError(
                "Azure AI Search could not delete the requested source."
            ) from exc

    def list_sources(self) -> list[str]:
        """Return the unique source filenames present in the index."""
        try:
            results = self._search_client.search(
                search_text="*",
                select=["source"],
                top=INDEX_BATCH_SIZE,
            )
            return sorted(
                {
                    str(record["source"])
                    for record in results
                    if record.get("source")
                }
            )
        except HttpResponseError as exc:
            raise SearchStoreError(
                "Azure AI Search could not list indexed sources."
            ) from exc

    def close(self) -> None:
        """Release Azure SDK clients."""
        self._search_client.close()
        self._index_client.close()
