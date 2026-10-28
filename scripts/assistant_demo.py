"""Live Semantic Kernel demo. Calls Azure and may consume credits."""

import asyncio

from src.app.azure_embeddings import AzureOpenAIEmbedder
from src.app.azure_search_store import AzureSearchStore
from src.app.models import ChatRequest
from src.app.sk_assistant import SemanticKernelAssistant


async def main() -> None:
    embedder = AzureOpenAIEmbedder.from_environment()
    store = None
    assistant = None

    try:
        store = AzureSearchStore.from_environment(embedder)
        assistant = SemanticKernelAssistant.from_environment(store)

        request = ChatRequest(
            session_id="phase-7-demo",
            question="What does hybrid retrieval combine?",
            top_k=3,
        )
        response = await assistant.answer(request)

        print(f"Question: {request.question}\n")
        print(f"Answer: {response.answer}")

        if response.pii_detected:
            print(f"PII types masked: {', '.join(response.pii_detected)}")

        if response.citations:
            print("\nCitations:")
            for citation in response.citations:
                page = (
                    str(citation.page)
                    if citation.page is not None
                    else "unknown"
                )
                print(
                    f"[{citation.number}] {citation.source}, "
                    f"page {page}: {citation.text[:250]}"
                )
        else:
            print("\nNo citations were returned.")
    finally:
        if assistant is not None:
            await assistant.aclose()
        if store is not None:
            store.close()
        embedder.close()


if __name__ == "__main__":
    asyncio.run(main())
