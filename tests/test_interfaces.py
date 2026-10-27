"""Tests for provider-independent interface contracts."""

from collections.abc import Sequence

import pytest

from src.app.interfaces import ChatService, Embedder, SearchStore
from src.app.models import DocumentChunk


def test_interfaces_are_abstract() -> None:
    for interface in (Embedder, SearchStore, ChatService):
        with pytest.raises(TypeError):
            interface()


class FakeEmbedder(Embedder):
    """In-memory embedding fake for tests."""

    async def embed_documents(
        self, texts: Sequence[str]
    ) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class FakeSearchStore(SearchStore):
    """In-memory search fake for tests."""

    async def upsert_chunks(
        self, chunks: Sequence[DocumentChunk]
    ) -> None:
        return None

    async def hybrid_search(
        self, query: str, top_k: int = 5
    ) -> list[DocumentChunk]:
        return []

    async def delete_by_source(self, source: str) -> None:
        return None

    async def list_sources(self) -> list[str]:
        return []


class FakeChatService(ChatService):
    """Fake answer generator for tests."""

    async def generate_answer(
        self,
        question: str,
        passages: Sequence[DocumentChunk],
    ) -> str:
        return "Fake answer"


def test_fake_implementations_satisfy_interface_contracts() -> None:
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(FakeSearchStore(), SearchStore)
    assert isinstance(FakeChatService(), ChatService)