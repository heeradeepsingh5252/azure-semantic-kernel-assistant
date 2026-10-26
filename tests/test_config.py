import pytest

from src.app.config import ConfigurationError, get_settings


def valid_azure_environment() -> dict[str, str]:
    return {
        "APP_MODE": "azure",
        "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_API_VERSION": "test-version",
        "AZURE_OPENAI_CHAT_DEPLOYMENT": "test-chat",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "test-embedding",
        "AZURE_SEARCH_ENDPOINT": "https://example.search.windows.net",
        "AZURE_SEARCH_API_KEY": "test-search-key",
        "AZURE_SEARCH_INDEX_NAME": "test-index",
        "PII_ENGINE": "presidio",
        "ALLOWED_ORIGINS": "http://localhost:5173",
        "MAX_UPLOAD_MB": "10",
    }


def test_local_mode_needs_no_azure_credentials() -> None:
    settings = get_settings({"APP_MODE": "local"})

    assert settings.app_mode == "local"
    assert settings.azure_openai_endpoint is None
    assert settings.pii_engine == "presidio"


def test_default_values_and_multiple_origins() -> None:
    settings = get_settings({
        "ALLOWED_ORIGINS": "http://localhost:5173, http://localhost:4173"
    })

    assert settings.max_upload_mb == 10
    assert settings.allowed_origins == (
        "http://localhost:5173",
        "http://localhost:4173",
    )


def test_azure_mode_accepts_complete_configuration() -> None:
    settings = get_settings(valid_azure_environment())

    assert settings.app_mode == "azure"
    assert settings.azure_openai_chat_deployment == "test-chat"
    assert settings.azure_search_index_name == "test-index"


def test_azure_mode_rejects_missing_configuration() -> None:
    with pytest.raises(
        ConfigurationError,
        match="AZURE_OPENAI_ENDPOINT",
    ):
        get_settings({"APP_MODE": "azure"})


def test_invalid_app_mode_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="APP_MODE"):
        get_settings({"APP_MODE": "production"})


def test_invalid_pii_engine_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="PII_ENGINE"):
        get_settings({"PII_ENGINE": "unknown"})


def test_azure_language_requires_its_credentials() -> None:
    with pytest.raises(
        ConfigurationError,
        match="AZURE_LANGUAGE_ENDPOINT",
    ):
        get_settings({"PII_ENGINE": "azure_language"})


@pytest.mark.parametrize("value", ["zero", "0", "-1"])
def test_invalid_upload_size_is_rejected(value: str) -> None:
    with pytest.raises(
        ConfigurationError,
        match="MAX_UPLOAD_MB",
    ):
        get_settings({"MAX_UPLOAD_MB": value})
