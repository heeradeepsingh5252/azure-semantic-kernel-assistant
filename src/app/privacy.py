"""PII detection and masking for user-provided text."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)
REDACTION = "[REDACTED]"

EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
    r"(?:\.[A-Z]{2,})+",
    flags=re.IGNORECASE,
)

INDIAN_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?P<country>\+91[\s-]?)?"
    r"(?P<number>[6-9]\d{4}[\s-]?\d{5})(?!\d)"
)

AADHAAR_PATTERN = re.compile(
    r"(?<!\d)[2-9]\d{3}(?:[\s-]?\d{4}){2}(?!\d)"
)

PAN_PATTERN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", flags=re.IGNORECASE)

CREDIT_CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){11,18}\d(?!\d)")

PHONE_CONTEXT = re.compile(
    r"\b(phone|mobile|call|contact|whatsapp|telephone|tel)\b",
    flags=re.IGNORECASE,
)
AADHAAR_CONTEXT = re.compile(
    r"\b(aadhaar|aadhar|uidai|unique identification number)\b",
    flags=re.IGNORECASE,
)
PAN_CONTEXT = re.compile(
    r"\b(pan|permanent account number|tax id|tax identifier)\b",
    flags=re.IGNORECASE,
)


class PrivacyConfigurationError(ValueError):
    """Raised when the selected PII engine is not configured."""


class PrivacyProcessingError(RuntimeError):
    """Raised when PII detection fails; the original text is not included."""


@dataclass(frozen=True)
class MaskedResult:
    """Masked text plus entity types only, never detected raw values."""

    masked_text: str
    detected_entity_types: list[str]

    @property
    def has_pii(self) -> bool:
        """Return whether one or more entity types were detected."""
        return bool(self.detected_entity_types)


@dataclass(frozen=True)
class _PiiSpan:
    """Internal character range for a detected entity."""

    start: int
    end: int
    entity_types: frozenset[str]


def log_masking_result(result: MaskedResult, *, input_length: int) -> None:
    """Log only the input length and detected entity types, never its text."""
    entity_types = ",".join(result.detected_entity_types) or "none"
    LOGGER.info(
        "PII masking completed input_length=%d detected_entity_types=%s",
        input_length,
        entity_types,
    )


def _luhn_is_valid(value: str) -> bool:
    """Check a payment-card candidate using the Luhn checksum."""
    digits = [int(char) for char in value if char.isdigit()]
    if not 12 <= len(digits) <= 19:
        return False

    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit

    return checksum % 10 == 0


def _has_context(
    text: str,
    start: int,
    end: int,
    context_pattern: re.Pattern[str],
    *,
    radius: int = 40,
) -> bool:
    """Check for a useful label around a candidate identifier."""
    context = text[max(0, start - radius) : min(len(text), end + radius)]
    return bool(context_pattern.search(context))


def _regex_spans(text: str) -> list[_PiiSpan]:
    """Recognize common formats, including Indian identifiers and phones."""
    spans: list[_PiiSpan] = []

    for match in EMAIL_PATTERN.finditer(text):
        spans.append(
            _PiiSpan(match.start(), match.end(), frozenset({"EMAIL_ADDRESS"}))
        )

    for match in INDIAN_PHONE_PATTERN.finditer(text):
        has_country_prefix = bool(match.group("country"))
        has_phone_context = _has_context(
            text, match.start(), match.end(), PHONE_CONTEXT
        )
        if has_country_prefix or has_phone_context:
            spans.append(
                _PiiSpan(
                    match.start(),
                    match.end(),
                    frozenset({"PHONE_NUMBER"}),
                )
            )

    for match in AADHAAR_PATTERN.finditer(text):
        if _has_context(
            text, match.start(), match.end(), AADHAAR_CONTEXT
        ):
            spans.append(
                _PiiSpan(match.start(), match.end(), frozenset({"IN_AADHAAR"}))
            )

    for match in PAN_PATTERN.finditer(text):
        if _has_context(text, match.start(), match.end(), PAN_CONTEXT):
            spans.append(
                _PiiSpan(match.start(), match.end(), frozenset({"IN_PAN"}))
            )

    for match in CREDIT_CARD_PATTERN.finditer(text):
        if _luhn_is_valid(match.group()):
            spans.append(
                _PiiSpan(match.start(), match.end(), frozenset({"CREDIT_CARD"}))
            )

    return spans


def _create_presidio_analyzer() -> Any:
    """Create a local English Presidio analyzer with spaCy's small model."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }
    nlp_engine = NlpEngineProvider(
        nlp_configuration=configuration
    ).create_engine()

    return AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en"],
    )


@lru_cache(maxsize=1)
def _get_presidio_analyzer() -> Any:
    """Cache the local analyzer so the NLP model is not reloaded per request."""
    return _create_presidio_analyzer()


def _presidio_spans(text: str, analyzer: Any | None = None) -> list[_PiiSpan]:
    """Find person names using Presidio, without exposing recognizer values."""
    try:
        active_analyzer = analyzer or _get_presidio_analyzer()
        results = active_analyzer.analyze(
            text=text,
            language="en",
            entities=["PERSON"],
        )
    except Exception:
        raise PrivacyProcessingError(
            "Local Presidio detection failed. Check the Presidio and spaCy setup."
        ) from None

    spans = []
    for result in results:
        start = int(result.start)
        end = int(result.end)
        if 0 <= start < end <= len(text):
            spans.append(
                _PiiSpan(
                    start,
                    end,
                    frozenset({str(result.entity_type).upper()}),
                )
            )
    return spans


def _canonical_entity_type(category: Any) -> str:
    """Normalize Azure Language category names to stable entity labels."""
    value = getattr(category, "value", category)
    normalized = re.sub(r"[^a-z0-9]", "", str(value).lower())
    known = {
        "person": "PERSON",
        "email": "EMAIL_ADDRESS",
        "emailaddress": "EMAIL_ADDRESS",
        "phonenumber": "PHONE_NUMBER",
        "creditcard": "CREDIT_CARD",
        "bankaccount": "BANK_ACCOUNT",
        "address": "ADDRESS",
        "ipaddress": "IP_ADDRESS",
        "socialsecuritynumber": "SOCIAL_SECURITY_NUMBER",
    }
    if normalized in known:
        return known[normalized]

    return re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")


def _azure_language_spans(
    text: str,
    client: Any | None = None,
) -> list[_PiiSpan]:
    """Use Azure AI Language's PII endpoint and return only source spans."""
    owns_client = client is None
    active_client = client

    if active_client is None:
        endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT", "").strip()
        api_key = os.getenv("AZURE_LANGUAGE_API_KEY", "").strip()
        if not endpoint or not api_key:
            raise PrivacyConfigurationError(
                "AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_API_KEY "
                "must be configured when PII_ENGINE=azure_language."
            )

        try:
            from azure.ai.textanalytics import TextAnalyticsClient
            from azure.core.credentials import AzureKeyCredential

            active_client = TextAnalyticsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key),
            )
        except Exception:
            raise PrivacyProcessingError(
                "Azure AI Language client could not be initialized."
            ) from None

    try:
        response = active_client.recognize_pii_entities(
            [text],
            language="en",
            disable_service_logs=True,
            string_index_type="UnicodeCodePoint",
        )
        documents = list(response)
        if not documents or documents[0].is_error:
            raise PrivacyProcessingError(
                "Azure AI Language could not process the text for PII."
            )

        spans = []
        for entity in documents[0].entities:
            start = int(entity.offset)
            end = start + int(entity.length)
            if 0 <= start < end <= len(text):
                spans.append(
                    _PiiSpan(
                        start,
                        end,
                        frozenset({_canonical_entity_type(entity.category)}),
                    )
                )
        return spans
    except PrivacyProcessingError:
        raise
    except Exception:
        raise PrivacyProcessingError(
            "Azure AI Language PII detection failed."
        ) from None
    finally:
        if owns_client and active_client is not None:
            try:
                active_client.close()
            except Exception:
                pass


def _merge_overlapping_spans(spans: list[_PiiSpan]) -> list[_PiiSpan]:
    """Combine overlapping detections before replacing text."""
    merged: list[_PiiSpan] = []

    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if not merged or span.start >= merged[-1].end:
            merged.append(span)
            continue

        previous = merged[-1]
        merged[-1] = _PiiSpan(
            start=previous.start,
            end=max(previous.end, span.end),
            entity_types=previous.entity_types | span.entity_types,
        )

    return merged


def _build_result(text: str, spans: list[_PiiSpan]) -> MaskedResult:
    """Replace detected ranges and return entity labels only."""
    merged = _merge_overlapping_spans(spans)
    parts: list[str] = []
    cursor = 0
    detected_types: set[str] = set()

    for span in merged:
        parts.append(text[cursor : span.start])
        parts.append(REDACTION)
        cursor = span.end
        detected_types.update(span.entity_types)

    parts.append(text[cursor:])

    return MaskedResult(
        masked_text="".join(parts),
        detected_entity_types=sorted(detected_types),
    )


def mask_pii(
    text: str,
    *,
    engine: str | None = None,
    analyzer: Any | None = None,
    azure_client: Any | None = None,
) -> MaskedResult:
    """Mask detected PII and return detected entity types without raw values.

    The optional analyzer and Azure client arguments allow offline unit tests
    to supply fakes. Production callers should normally use the defaults.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    load_dotenv()
    selected_engine = (engine or os.getenv("PII_ENGINE", "presidio")).strip().lower()

    if selected_engine not in {"presidio", "azure_language"}:
        raise PrivacyConfigurationError(
            "PII_ENGINE must be 'presidio' or 'azure_language'."
        )

    spans = _regex_spans(text)

    if text:
        if selected_engine == "presidio":
            spans.extend(_presidio_spans(text, analyzer=analyzer))
        else:
            spans.extend(_azure_language_spans(text, client=azure_client))

    result = _build_result(text, spans)
    log_masking_result(result, input_length=len(text))
    return result
