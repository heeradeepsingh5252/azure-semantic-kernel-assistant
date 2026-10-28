"""Live Azure integration check. Calls Azure and may consume credits."""

from pathlib import Path

from src.app.azure_embeddings import AzureOpenAIEmbedder
from src.app.azure_search_store import AzureSearchStore
from src.app.chunker import chunk_pages
from src.app.loader import extract_document


def main() -> None:
    sample_path = Path("data/sample.txt")
    if not sample_path.is_file():
        raise FileNotFoundError("Expected integration document: data/sample.txt")

    pages = extract_document(sample_path.name, sample_path.read_bytes())
    chunks = chunk_pages(pages)

    with AzureOpenAIEmbedder.from_environment() as embedder:
        store = AzureSearchStore.from_environment(embedder)
        try:
            removed_count = store.delete_by_source(sample_path.name)
            print(
                f"Removed {removed_count} previous chunk(s) "
                f"for {sample_path.name}."
            )

            uploaded_count = store.upsert_chunks(chunks)
            print(
                f"Uploaded {uploaded_count} current chunk(s) "
                f"from {sample_path.name}."
            )

            print(f"Indexed sources: {store.list_sources()}")

            query = "What does hybrid retrieval combine?"
            results = store.hybrid_search(query, top_k=3)

            print(f"\nQuery: {query}")
            if not results:
                print("No matching chunks were returned.")
            for rank, result in enumerate(results, start=1):
                page = result.page if result.page is not None else "N/A"
                print(f"\n[{rank}] {result.source}, page {page}")
                print(result.content[:500])
        finally:
            store.close()


if __name__ == "__main__":
    main()
