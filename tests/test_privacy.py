"""Offline tests for PII detection, masking, and safe logging."""

import logging
from types import SimpleNamespace

import pytest

from src.app.privacy import (
    MaskedResult,
    PrivacyConfigurationError,
    PrivacyProcessingError,
    mask_pii,
)


class FakeAnalyzer:
    """Return known person-name spans without loading an NLP model."""

    def __init__(self, names: list[str] | None = None) -> None:
        self.names = names or []
        self.calls = []

    def analyze(self, *, text: str, language: str, entities: list[str]):
        self.calls.append({"text": text, "language": language, "entities": entities})
        results = []
        for name in self.names:
            start = text.find(name)
            if start >= 0:
                results.append(
                    SimpleNamespace(
                        start=start,
                        end=start + len(name),
                        entity_type="PERSON",
                    )
                )
        return results


class FakeAzureClient:
    """Stand-in for the Azure Language SDK."""

    def __init__(self, entities, *, is_error: bool = False) -> None:
        self.entities = entities
        self.is_error = is_error
        self.calls = []

    def recognize_pii_entities(self, documents, **kwargs):
        self.calls.append({"documents": documents, **kwargs})
        if self.is_error:
            result = SimpleNamespace(is_error=True, entities=[])
        else:
            result = SimpleNamespace(is_error=False, entities=self.entities)
        return [result]


def test_masks_email_indian_phone_aadhaar_pan_and_credit_card(caplog) -> None:
    text = (
        "Email priya@example.com; phone +91 98765 43210; "
        "Aadhaar 2345 6789 0123; PAN ABCDE1234F; "
        "card 4539 1488 0343 6467."
    )

    with caplog.at_level(logging.INFO, logger="src.app.privacy"):
        result = mask_pii(text, engine="presidio", analyzer=FakeAnalyzer())

    assert isinstance(result, MaskedResult)
    assert result.has_pii is True
    assert "priya@example.com" not in result.masked_text
    assert "+91 98765 43210" not in result.masked_text
    assert "2345 6789 0123" not in result.masked_text
    assert "ABCDE1234F" not in result.masked_text
    assert "4539 1488 0343 6467" not in result.masked_text
    assert result.masked_text.count("[REDACTED]") == 5
    assert result.detected_entity_types == [
        "CREDIT_CARD",
        "EMAIL_ADDRESS",
        "IN_AADHAAR",
        "IN_PAN",
        "PHONE_NUMBER",
    ]
    assert "priya@example.com" not in caplog.text
    assert "+91 98765 43210" not in caplog.text
    assert "input_length=" in caplog.text
    assert "EMAIL_ADDRESS" in caplog.text


def test_person_name_detection_uses_presidio_analyzer() -> None:
    text = "My name is Priya Sharma and I need help."
    analyzer = FakeAnalyzer(names=["Priya Sharma"])

    result = mask_pii(text, engine="presidio", analyzer=analyzer)

    assert "Priya Sharma" not in result.masked_text
    assert "PERSON" in result.detected_entity_types
    assert analyzer.calls[0]["language"] == "en"
    assert analyzer.calls[0]["entities"] == ["PERSON"]


def test_false_positive_checks_leave_unlabelled_identifiers_alone() -> None:
    text = (
        "Reference numbers: ABCDE1234F, 2345 6789 0123, "
        "98765 43210 and 4539 1488 0343 6468."
    )

    result = mask_pii(text, engine="presidio", analyzer=FakeAnalyzer())

    assert result.masked_text == text
    assert result.detected_entity_types == []
    assert result.has_pii is False


def test_empty_text_returns_empty_result() -> None:
    result = mask_pii("", engine="presidio", analyzer=FakeAnalyzer())

    assert result.masked_text == ""
    assert result.detected_entity_types == []
    assert result.has_pii is False


def test_non_string_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        mask_pii(None)  # type: ignore[arg-type]


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(PrivacyConfigurationError, match="PII_ENGINE"):
        mask_pii("Hello", engine="unknown")


def test_presidio_failure_does_not_return_unmasked_text() -> None:
    class BrokenAnalyzer:
        def analyze(self, **kwargs):
            raise RuntimeError("sensitive content must not appear in errors")

    with pytest.raises(PrivacyProcessingError) as exc_info:
        mask_pii(
            "private details",
            engine="presidio",
            analyzer=BrokenAnalyzer(),
        )

    assert "sensitive content" not in str(exc_info.value)
    assert "private details" not in str(exc_info.value)


def test_azure_language_masks_entities_and_disables_service_logs() -> None:
    text = "Contact alex@example.com for help."
    start = text.index("alex@example.com")
    entity = SimpleNamespace(
        offset=start,
        length=len("alex@example.com"),
        category="Email",
    )
    client = FakeAzureClient([entity])

    result = mask_pii(text, engine="azure_language", azure_client=client)

    assert "alex@example.com" not in result.masked_text
    assert "EMAIL_ADDRESS" in result.detected_entity_types
    assert client.calls[0]["documents"] == [text]
    assert client.calls[0]["disable_service_logs"] is True
    assert client.calls[0]["string_index_type"] == "UnicodeCodePoint"


def test_azure_language_failure_does_not_return_unmasked_text() -> None:
    client = FakeAzureClient([], is_error=True)
    text = "private content"

    with pytest.raises(PrivacyProcessingError):
        mask_pii(text, engine="azure_language", azure_client=client)


def test_azure_language_requires_configuration_without_a_fake_client(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AZURE_LANGUAGE_ENDPOINT", "")
    monkeypatch.setenv("AZURE_LANGUAGE_API_KEY", "")

    with pytest.raises(
        PrivacyConfigurationError,
        match="AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_API_KEY",
    ):
        mask_pii("hello", engine="azure_language")


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("Person", "PERSON"),
        ("Email", "EMAIL_ADDRESS"),
        ("PhoneNumber", "PHONE_NUMBER"),
        ("CreditCard", "CREDIT_CARD"),
    ],
)
def test_azure_entity_categories_are_normalized(category, expected) -> None:
    from src.app.privacy import _canonical_entity_type

    assert _canonical_entity_type(category) == expected
