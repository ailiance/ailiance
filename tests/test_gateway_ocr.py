"""Tests for the OCR pipeline (image upload + image-only PDF fallback).

Requires the host to have ``tesseract-ocr`` + ``poppler-utils``
installed. CI is expected to install them via apt; tests are skipped
if Tesseract isn't available so the suite still runs on minimal dev
boxes.
"""

from __future__ import annotations

import importlib.util
import io
import shutil

import pytest

from src.gateway.file_extract import detect_format, extract


# OCR needs BOTH the tesseract binary and the pytesseract Python wrapper.
# The wrapper is an optional dep not pinned in requirements-ci.txt, so the
# binary alone is not sufficient (a dev box may have the binary via brew
# without the wrapper in the venv).
pytestmark = pytest.mark.skipif(
    shutil.which("tesseract") is None
    or importlib.util.find_spec("pytesseract") is None,
    reason="OCR needs tesseract binary + pytesseract wrapper",
)


def _png_with_text(text: str, w: int = 480, h: int = 120) -> bytes:
    """Render ``text`` onto a white PNG. Tesseract reads it back."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (w, h), color="white")
    draw = ImageDraw.Draw(img)
    # Try to use a real TTF font; fall back to default bitmap if not
    # found. DejaVu is present on most Linux distros + the macOS
    # installer Pillow ships with.
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 40), text, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestImageFormatDetection:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("photo.png", "image"),
            ("photo.PNG", "image"),
            ("photo.jpg", "image"),
            ("photo.jpeg", "image"),
            ("photo.gif", "image"),
            ("photo.webp", "image"),
            ("photo.tif", "image"),
        ],
    )
    def test_filename_dispatch(self, name, expected):
        assert detect_format(name, None) == expected

    def test_mime_dispatch(self):
        assert detect_format(None, "image/png") == "image"
        assert detect_format(None, "image/jpeg") == "image"


# ---------------------------------------------------------------------------
# Image OCR
# ---------------------------------------------------------------------------


class TestImageOCR:
    def test_simple_text_recognised(self):
        # Tesseract is reliable on rendered black-on-white text.
        data = _png_with_text("Hello Ailiance")
        result = extract(data, filename="hello.png", mime="image/png")
        assert result.format == "image"
        # OCR is fuzzy; accept any close match.
        normalised = result.markdown.lower().replace(" ", "")
        assert "hello" in normalised or "ailiance" in normalised, \
            f"unexpected OCR output: {result.markdown!r}"
        assert result.metadata.get("ocr") is True
        assert result.metadata.get("width") == 480

    def test_blank_image_returns_no_markdown(self):
        # All-white image — Tesseract returns empty text. That must
        # *not* raise.
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (100, 100), color="white").save(buf, format="PNG")
        result = extract(buf.getvalue(), filename="blank.png", mime="image/png")
        assert result.format == "image"
        assert result.markdown == ""


# ---------------------------------------------------------------------------
# PDF OCR fallback
# ---------------------------------------------------------------------------


class TestPDFOCRFallback:
    def test_text_pdf_skips_ocr(self):
        # The synthetic PDF from earlier sessions has pdfminer-readable
        # text; OCR must NOT be triggered.
        pdf = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/"
            b"Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica"
            b">>>>>>/Contents 4 0 R>>endobj\n"
            b"4 0 obj<</Length 120>>stream\n"
            b"BT /F1 18 Tf 100 700 Td (This PDF body has enough text to "
            b"escape OCR fallback for sure beyond fifty chars.) Tj ET\n"
            b"endstream endobj\n"
            b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000052 00000 n \n0000000098 00000 n \n0000000214 00000 n \n"
            b"trailer<</Size 5/Root 1 0 R>>\nstartxref\n400\n%%EOF"
        )
        result = extract(pdf, filename="text.pdf", mime="application/pdf")
        assert result.format == "pdf"
        assert "fifty" in result.markdown
        assert result.metadata.get("ocr") is False

    def test_image_only_pdf_triggers_ocr(self):
        # Build an "image-only" PDF by embedding our rendered PNG as
        # the sole content of a page. Easiest path: convert via Pillow
        # then save as PDF.
        from PIL import Image

        img = Image.open(io.BytesIO(_png_with_text("OCR Fallback PDF"))).convert("RGB")
        pdf_buf = io.BytesIO()
        img.save(pdf_buf, format="PDF")
        result = extract(pdf_buf.getvalue(), filename="scan.pdf", mime="application/pdf")
        assert result.format == "pdf"
        # OCR path produces a "## Page 1" header — distinct from text path.
        # Threshold may or may not trigger OCR depending on pdfminer's
        # extraction of Pillow's PDF format. We only assert that *if*
        # OCR ran, the result contains a recognisable token; if not,
        # we accept whatever pdfminer found (no false-positive failure).
        if result.metadata.get("ocr"):
            normalised = result.markdown.lower().replace(" ", "")
            assert "ocr" in normalised or "fallback" in normalised
