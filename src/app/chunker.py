"""Split extracted document pages into overlapping, traceable chunks."""

from collections.abc import Sequence
from hashlib import sha256

from src.app.loader import ExtractedPage
from src.app.models import DocumentChunk

MAX_CHUNK_SIZE = 20_000


def chunk_pages(
    pages: Sequence[ExtractedPage],
    *,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[DocumentChunk]:
    """Split pages into chunks while preserving source and page metadata.

    Chunk size and overlap are measured in characters. Whitespace is
    normalized before splitting. Chunks prefer word boundaries where
    practical, and overlapping content helps preserve local context.

    IDs are deterministic for a given ordered input and configuration.
    """
    if not 1 <= chunk_size <= MAX_CHUNK_SIZE:
        raise ValueError(
            f"chunk_size must be between 1 and {MAX_CHUNK_SIZE}."
        )
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1.")

    chunks: list[DocumentChunk] = []

    for page in pages:
        text = " ".join(page.text.split())
        if not text:
            continue

        start = 0
        while start < len(text):
            proposed_end = min(start + chunk_size, len(text))
            end = proposed_end

            if proposed_end < len(text):
                minimum_boundary = start + max(1, (chunk_size * 3) // 4)
                boundary = text.rfind(
                    " ", minimum_boundary, proposed_end
                )
                if boundary > start:
                    end = boundary

            content = text[start:end].strip()
            if content:
                # Include the global chunk position so repeated passages
                # in the same batch receive different IDs.
                identity = (
                    f"{page.source}\0{page.page}\0{len(chunks)}\0{content}"
                )
                digest = sha256(identity.encode("utf-8")).hexdigest()[:24]
                chunk_id = f"chunk-{digest}"

                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        chunk_id=chunk_id,
                        content=content,
                        source=page.source,
                        page=page.page,
                    )
                )

            if end >= len(text):
                break

            # Start near the requested overlap. Align to a preceding word
            # boundary where possible, without skipping unseen text.
            next_start = max(start + 1, end - overlap)
            previous_space = text.rfind(
                " ", start + 1, next_start + 1
            )
            if previous_space > start:
                next_start = previous_space + 1

            if next_start <= start:
                next_start = end

            start = next_start

    return chunks
