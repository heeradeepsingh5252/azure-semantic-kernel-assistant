"""Azure OpenAI embedding adapter with batching and retry handling."""

import os
from collections.abc import Sequence

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import APIError, OpenAI

from src.app.interfaces import Embedder

EMBEDDING_DIMENSIONS = 1536
EMBEDDING_TOKEN_SCOPE = "https://ai.azure.com/.default"
DEFAULT_BATCH_SIZE = 64


class EmbeddingServiceError(RuntimeError):
    """Raised when the embedding service request fails."""


class AzureOpenAIEmbedder(Embedder):
    """Generate document and query embeddings using Azure OpenAI."""

    def __init__(
        self,
        *,
        client: OpenAI,
        deployment: str,
        credential: DefaultAzureCredential | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if not deployment.strip():
            raise ValueError("Embedding deployment name must not be empty.")
        if not 1 <= batch_size <= 2048:
            raise ValueError("batch_size must be between 1 and 2048.")

        self._client = client
        self._deployment = deployment.strip()
        self._credential = credential
        self._batch_size = batch_size

    @classmethod
    def from_environment(cls) -> "AzureOpenAIEmbedder":
        """Create an Entra-authenticated client from environment settings."""
        load_dotenv()

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "").strip()

        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT must be configured.")
        if not deployment:
            raise ValueError(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT must be configured."
            )

        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential,
            EMBEDDING_TOKEN_SCOPE,
        )

        client = OpenAI(
            base_url=f"{endpoint.rstrip('/')}/",
            api_key=token_provider,
            max_retries=3,
            timeout=30.0,
        )

        return cls(
            client=client,
            deployment=deployment,
            credential=credential,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches while preserving their original order."""
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not one string.")

        inputs = list(texts)
        if not inputs:
            return []

        if any(not isinstance(text, str) or not text.strip() for text in inputs):
            raise ValueError("Every embedding input must be a non-empty string.")

        all_vectors: list[list[float]] = []

        for start in range(0, len(inputs), self._batch_size):
            batch = inputs[start : start + self._batch_size]

            try:
                response = self._client.embeddings.create(
                    model=self._deployment,
                    input=batch,
                )
            except APIError as exc:
                status_code = getattr(exc, "status_code", None)
                request_id = getattr(exc, "request_id", None)
                details = []
                if status_code is not None:
                    details.append(f"HTTP status {status_code}")
                if request_id:
                    details.append(f"request ID {request_id}")
                suffix = f" ({', '.join(details)})" if details else ""
                raise EmbeddingServiceError(
                    f"Azure embedding request failed{suffix}."
                ) from exc

            ordered: list[list[float] | None] = [None] * len(batch)

            for item in response.data:
                index = item.index
                if index < 0 or index >= len(batch) or ordered[index] is not None:
                    raise EmbeddingServiceError(
                        "Embedding response contained an invalid item index."
                    )

                vector = [float(value) for value in item.embedding]
                if len(vector) != EMBEDDING_DIMENSIONS:
                    raise EmbeddingServiceError(
                        "Embedding dimension mismatch: "
                        f"expected {EMBEDDING_DIMENSIONS}, got {len(vector)}."
                    )

                ordered[index] = vector

            if any(vector is None for vector in ordered):
                raise EmbeddingServiceError(
                    "Embedding response did not include every input."
                )

            all_vectors.extend(
                vector for vector in ordered if vector is not None
            )

        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query."""
        return self.embed_documents([text])[0]

    def close(self) -> None:
        """Release the client and its credential."""
        self._client.close()
        if self._credential is not None:
            self._credential.close()

    def __enter__(self) -> "AzureOpenAIEmbedder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
