
"""Environment-based configuration for the assistant."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class ConfigurationError(ValueError):
    """Raised when application configuration is invalid."""


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    app_mode: str
    azure_openai_endpoint: str | None
    azure_openai_api_key: str | None
    azure_openai_api_version: str | None
    azure_openai_chat_deployment: str | None
    azure_openai_embedding_deployment: str | None
    azure_search_endpoint: str | None
    azure_search_api_key: str | None
    azure_search_index_name: str | None
    pii_engine: str
    azure_language_endpoint: str | None
    azure_language_api_key: str | None
    allowed_origins: tuple[str, ...]
    max_upload_mb: int


# All supported Azure settings, including optional settings.
AZURE_SETTING_NAMES = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_API_KEY",
    "AZURE_SEARCH_INDEX_NAME",
)

# Required settings for the current Azure service implementation.
# Entra ID authentication does not require an OpenAI API key.
# The OpenAI v1 endpoint does not require a dated API version.
AZURE_REQUIRED_SETTING_NAMES = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_API_KEY",
    "AZURE_SEARCH_INDEX_NAME",
)


def get_settings(
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Validate and return application settings."""
    values = os.environ if environ is None else environ

    app_mode = values.get("APP_MODE", "local").strip().lower()
    if app_mode not in {"local", "azure"}:
        raise ConfigurationError("APP_MODE must be 'local' or 'azure'.")

    pii_engine = values.get("PII_ENGINE", "presidio").strip().lower()
    if pii_engine not in {"presidio", "azure_language"}:
        raise ConfigurationError(
            "PII_ENGINE must be 'presidio' or 'azure_language'."
        )

    azure_values = {
        name: values.get(name, "").strip() or None
        for name in AZURE_SETTING_NAMES
    }

    missing = (
        [
            name
            for name in AZURE_REQUIRED_SETTING_NAMES
            if azure_values[name] is None
        ]
        if app_mode == "azure"
        else []
    )

    if pii_engine == "azure_language":
        for name in ("AZURE_LANGUAGE_ENDPOINT", "AZURE_LANGUAGE_API_KEY"):
            if not values.get(name, "").strip():
                missing.append(name)

    if missing:
        names = ", ".join(missing)
        raise ConfigurationError(
            f"Missing required environment settings: {names}"
        )

    raw_max_upload = values.get("MAX_UPLOAD_MB", "10").strip()
    try:
        max_upload_mb = int(raw_max_upload)
    except ValueError as exc:
        raise ConfigurationError(
            "MAX_UPLOAD_MB must be a positive integer."
        ) from exc

    if max_upload_mb < 1:
        raise ConfigurationError(
            "MAX_UPLOAD_MB must be a positive integer."
        )

    raw_origins = values.get(
        "ALLOWED_ORIGINS", "http://localhost:5173"
    )
    allowed_origins = tuple(
        origin.strip()
        for origin in raw_origins.split(",")
        if origin.strip()
    )

    return Settings(
        app_mode=app_mode,
        azure_openai_endpoint=azure_values["AZURE_OPENAI_ENDPOINT"],
        azure_openai_api_key=azure_values["AZURE_OPENAI_API_KEY"],
        azure_openai_api_version=azure_values["AZURE_OPENAI_API_VERSION"],
        azure_openai_chat_deployment=azure_values[
            "AZURE_OPENAI_CHAT_DEPLOYMENT"
        ],
        azure_openai_embedding_deployment=azure_values[
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        ],
        azure_search_endpoint=azure_values["AZURE_SEARCH_ENDPOINT"],
        azure_search_api_key=azure_values["AZURE_SEARCH_API_KEY"],
        azure_search_index_name=azure_values["AZURE_SEARCH_INDEX_NAME"],
        pii_engine=pii_engine,
        azure_language_endpoint=(
            values.get("AZURE_LANGUAGE_ENDPOINT", "").strip() or None
        ),
        azure_language_api_key=(
            values.get("AZURE_LANGUAGE_API_KEY", "").strip() or None
        ),
        allowed_origins=allowed_origins,
        max_upload_mb=max_upload_mb,
    )
