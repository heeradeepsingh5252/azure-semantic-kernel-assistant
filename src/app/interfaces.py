"""Provider-independent interfaces for the RAG application."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.app.models import DocumentChunk


class Embedder(ABC):
    """Contract for generating document and query embeddings."""

    @abstractmethod
    async def embed_documents(
        self, texts: Sequence[str]
    ) -> list[list[float]]:
        """Generate a vector for each document text."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Generate a vector for a query."""


class SearchStore(ABC):
    """Contract for storing and retrieving document chunks."""

    @abstractmethod
    async def upsert_chunks(
        self, chunks: Sequence[DocumentChunk]
    ) -> None:
        """Insert or update document chunks."""

    @abstractmethod
    async def hybrid_search(
        self, query: str, top_k: int = 5
    ) -> list[DocumentChunk]:
        """Retrieve relevant chunks using keyword and vector search."""

    @abstractmethod
    async def delete_by_source(self, source: str) -> None:
        """Delete indexed chunks belonging to one source."""

    @abstractmethod
    async def list_sources(self) -> list[str]:
        """List the indexed source documents."""


class ChatService(ABC):
    """Contract for generating answers from retrieved passages."""

    @abstractmethod
    async def generate_answer(
        self,
        question: str,
        passages: Sequence[DocumentChunk],
    ) -> str:
        """Generate an answer grounded in supplied passages."""
