"""Tests for character-based chunking and metadata preservation."""

import pytest

from src.app.chunker import chunk_pages
from src.app.loader import ExtractedPage


def test_chunks_respect_size_and_preserve_metadata() -> None:
    page = ExtractedPage(
        source="guide.txt",
        page=3,
        text=" ".join(f"word{i}" for i in range(80)),
    )

    chunks = chunk_pages([page], chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 100 for chunk in chunks)
    assert all(chunk.source == "guide.txt" for chunk in chunks)
    assert all(chunk.page == 3 for chunk in chunks)
    assert all(chunk.content for chunk in chunks)
    assert all(chunk.id == chunk.chunk_id for chunk in chunks)


def test_adjacent_chunks_have_overlapping_content() -> None:
    page = ExtractedPage(
        source="overlap.txt",
        page=1,
        text=" ".join(f"word{i}" for i in range(50)),
    )

    chunks = chunk_pages([page], chunk_size=60, overlap=15)

    assert len(chunks) > 1
    assert any(
        left.content.split()[-1] in right.content.split()
        for left, right in zip(chunks, chunks[1:], strict=False)
    )


def test_chunk_ids_are_unique_and_deterministic() -> None:
    pages = [
        ExtractedPage(
            source="guide.txt",
            page=1,
            text=" ".join(f"first{i}" for i in range(30)),
        ),
        ExtractedPage(
            source="guide.txt",
            page=2,
            text=" ".join(f"second{i}" for i in range(30)),
        ),
    ]

    first_run = chunk_pages(pages, chunk_size=80, overlap=10)
    second_run = chunk_pages(pages, chunk_size=80, overlap=10)

    first_ids = [chunk.chunk_id for chunk in first_run]
    second_ids = [chunk.chunk_id for chunk in second_run]

    assert len(first_ids) == len(set(first_ids))
    assert first_ids == second_ids
    assert {chunk.page for chunk in first_run} == {1, 2}


def test_blank_pages_are_skipped() -> None:
    pages = [
        ExtractedPage(source="guide.txt", page=1, text="   \n "),
        ExtractedPage(source="guide.txt", page=2, text="Useful content here."),
    ]

    chunks = chunk_pages(pages, chunk_size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0].page == 2
    assert chunks[0].content == "Useful content here."


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (20_001, 0),
        (100, -1),
        (100, 100),
    ],
)
def test_invalid_chunk_settings_are_rejected(
    chunk_size: int,
    overlap: int,
) -> None:
    page = ExtractedPage(
        source="guide.txt",
        page=1,
        text="Example content.",
    )

    with pytest.raises(ValueError):
        chunk_pages([page], chunk_size=chunk_size, overlap=overlap)
