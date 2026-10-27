"""Send one test prompt to the Azure GPT-5-mini deployment."""

import os
from pathlib import Path

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    """Send a small chat request using Microsoft Entra ID."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip()

    if not endpoint:
        raise SystemExit("AZURE_OPENAI_ENDPOINT is missing from .env.")
    if not deployment:
        raise SystemExit("AZURE_OPENAI_CHAT_DEPLOYMENT is missing from .env.")
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
            response = client.responses.create(
                model=deployment,
                input=(
                    "Reply in one short sentence: "
                    "Explain what Microsoft Semantic Kernel does."
                ),
                text={"format": {"type": "text"}},
                reasoning={"effort": "low"},
                max_output_tokens=500,
            )

        answer = (response.output_text or "").strip()
        if not answer:
            raise SystemExit("The Azure chat deployment returned no text.")

        print("Provider: Azure OpenAI")
        print(f"Deployment: {deployment}")
        print("\nResponse:\n")
        print(answer)

    except Exception as exc:
        raise SystemExit(
            f"Azure chat request failed ({type(exc).__name__}): {exc}\n"
            "Check az login, deployment name, endpoint, and Azure role access."
        ) from exc
    finally:
        credential.close()


if __name__ == "__main__":
    main()