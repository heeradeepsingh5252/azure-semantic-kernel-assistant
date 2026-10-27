"""Generate one embedding using Azure text-embedding-3-small."""

import os
from pathlib import Path

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    """Embed one generic sentence using Microsoft Entra ID."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "").strip()

    if not endpoint:
        raise SystemExit("AZURE_OPENAI_ENDPOINT is missing from .env.")
    if not deployment:
        raise SystemExit(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT is missing from .env."
        )
    if "/openai/v1" not in endpoint:
        raise SystemExit(
            "AZURE_OPENAI_ENDPOINT should end with /openai/v1/ "
            "for this script."
        )

    credential = DefaultAzureCredential()
    try:
        token_provider = get_bearer_token_provider(
            credential,
            "https://ai.azure.com/.default",
        )

        with OpenAI(
            base_url=endpoint,
            api_key=token_provider,
        ) as client:
            result = client.embeddings.create(
                model=deployment,
                input=(
                    "Document retrieval helps an AI assistant find "
                    "relevant passages before answering a question."
                ),
            )

        if not result.data or not result.data[0].embedding:
            raise SystemExit("The Azure embedding deployment returned no vector.")

        vector = result.data[0].embedding

        print("Provider: Azure OpenAI")
        print(f"Deployment: {deployment}")
        print(f"Embedding dimensions: {len(vector)}")
        print(f"First 5 values: {[round(value, 6) for value in vector[:5]]}")

    except Exception as exc:
        raise SystemExit(
            f"Azure embedding request failed ({type(exc).__name__}): {exc}\n"
            "Check az login, deployment name, endpoint, and Azure role access."
        ) from exc
    finally:
        credential.close()


if __name__ == "__main__":
    main()