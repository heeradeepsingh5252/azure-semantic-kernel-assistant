"""Offline tests for TXT and PDF document loading."""

from pathlib import Path

import pytest

from src.app.loader import DocumentLoadError, ExtractedPage, extract_document

FIXTURES = Path(__file__).parent / "fixtures"


def make_text_pdf(page_texts: list[str]) -> bytes:
    """Build a tiny PDF fixture containing selectable text."""
    objects: dict[int, bytes] = {}
    page_ids = [3 + index * 2 for index in range(len(page_texts))]
    font_id = 3 + 2 * len(page_texts)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
    )

    for index, text in enumerate(page_texts):
        page_id = page_ids[index]
        content_id = page_id + 1

        escaped = (
            text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")

        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode()
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"\nendstream"
        )

    objects[font_id] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0] * (font_id + 1)

    for object_id in range(1, font_id + 1):
        offsets[object_id] = len(pdf)
        pdf.extend(f"{object_id} 0 obj\n".encode())
        pdf.extend(objects[object_id])
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {font_id + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")

    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())

    pdf.extend(
        f"trailer\n<< /Size {font_id + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(pdf)


def test_extract_txt_fixture() -> None:
    content = (FIXTURES / "sample.txt").read_bytes()

    pages = extract_document("sample.txt", content)

    assert len(pages) == 1
    assert isinstance(pages[0], ExtractedPage)
    assert pages[0].source == "sample.txt"
    assert pages[0].page == 1
    assert "Semantic Kernel" in pages[0].text


def test_extract_pdf_preserves_page_numbers() -> None:
    pdf_bytes = make_text_pdf(
        ["First page contains retrieval basics.", "Second page contains citations."]
    )

    pages = extract_document("sample.pdf", pdf_bytes)

    assert [(page.page, page.source) for page in pages] == [
        (1, "sample.pdf"),
        (2, "sample.pdf"),
    ]
    assert "retrieval basics" in pages[0].text
    assert "citations" in pages[1].text


@pytest.mark.parametrize("filename", ["empty.txt", "empty.pdf"])
def test_reject_empty_file(filename: str) -> None:
    with pytest.raises(DocumentLoadError, match="empty"):
        extract_document(filename, b"")


def test_reject_unsupported_file_type() -> None:
    with pytest.raises(DocumentLoadError, match="Unsupported file type"):
        extract_document("notes.docx", b"some content")


def test_reject_invalid_utf8_txt() -> None:
    with pytest.raises(DocumentLoadError, match="UTF-8"):
        extract_document("invalid.txt", b"\xff\xfe\xfa")


def test_reject_malformed_pdf() -> None:
    with pytest.raises(DocumentLoadError, match="malformed"):
        extract_document("broken.pdf", b"this is not a PDF")


def test_detect_pdf_without_extractable_text() -> None:
    blank_pdf = make_text_pdf([""])

    with pytest.raises(DocumentLoadError, match="OCR is not implemented"):
        extract_document("scan.pdf", blank_pdf)


def test_skip_blank_pdf_page_and_keep_original_number() -> None:
    pdf_bytes = make_text_pdf(
        ["First page has text.", "", "Third page has text."]
    )

    pages = extract_document("mixed.pdf", pdf_bytes)

    assert [page.page for page in pages] == [1, 3]
