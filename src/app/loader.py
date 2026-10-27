"""Load PDF and TXT documents while preserving source page metadata."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


class DocumentLoadError(ValueError):
    """Raised when a document cannot be loaded or contains no usable text."""


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text extracted from one source document page."""

    source: str
    page: int
    text: str


def extract_document(filename: str, content: bytes) -> list[ExtractedPage]:
    """Extract text from a TXT or PDF byte stream.

    TXT files are represented as page 1. PDF page numbers are preserved
    from the original document. Pages without text are skipped.

    This function does not perform OCR on scanned PDFs.
    """
    if not filename or not filename.strip():
        raise DocumentLoadError("A filename is required.")

    source_path = Path(filename.strip())
    source = source_path.name
    suffix = source_path.suffix.lower()

    if suffix not in {".txt", ".pdf"}:
        raise DocumentLoadError(
            f"Unsupported file type '{suffix or '(none)'}'. "
            "Only PDF and TXT files are supported."
        )

    if not content:
        raise DocumentLoadError("The uploaded document is empty.")

    if suffix == ".txt":
        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise DocumentLoadError(
                "The TXT file must contain valid UTF-8 text."
            ) from exc

        if not text:
            raise DocumentLoadError("The TXT file contains no usable text.")

        return [ExtractedPage(source=source, page=1, text=text)]

    return _extract_pdf(source, content)


def _extract_pdf(source: str, content: bytes) -> list[ExtractedPage]:
    """Extract non-empty PDF pages and preserve original page numbers."""
    try:
        reader = PdfReader(BytesIO(content), strict=False)
    except Exception as exc:
        raise DocumentLoadError(
            f"Could not read PDF '{source}'. The file may be malformed."
        ) from exc

    if reader.is_encrypted:
        raise DocumentLoadError(
            f"PDF '{source}' is encrypted. Password-protected PDFs "
            "are not supported."
        )

    if not reader.pages:
        raise DocumentLoadError(f"PDF '{source}' contains no pages.")

    extracted_pages: list[ExtractedPage] = []

    for page_number, pdf_page in enumerate(reader.pages, start=1):
        try:
            text = (pdf_page.extract_text() or "").strip()
        except Exception as exc:
            raise DocumentLoadError(
                f"Could not extract text from page {page_number} "
                f"of PDF '{source}'."
            ) from exc

        if text:
            extracted_pages.append(
                ExtractedPage(source=source, page=page_number, text=text)
            )

    if not extracted_pages:
        raise DocumentLoadError(
            f"PDF '{source}' contains no extractable text. "
            "It may be scanned or image-only; OCR is not implemented."
        )

    return extracted_pages
