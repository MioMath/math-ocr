"""Tests for PDF processing and figure extraction."""

import json
import struct
import tempfile
from pathlib import Path

import pytest

from math_ocr.pdf import (
    FigureRegion,
    ImageIndexEntry,
    _is_logo,
    _png_dimensions,
    load_image_index,
    normalize_bbox,
    prescan_pdfs,
)


# ── normalize_bbox ───────────────────────────────────────────────────


class TestNormalizeBbox:
    def test_pixel_to_normalized(self):
        result = normalize_bbox([100, 200, 300, 400], page_width=1000, page_height=1000)
        assert result == [0.1, 0.2, 0.3, 0.4]

    def test_already_normalized(self):
        # Values <= 1.0 are treated as already normalized
        result = normalize_bbox([0.1, 0.2, 0.5, 0.8], page_width=1000, page_height=1000)
        assert result == [0.1, 0.2, 0.5, 0.8]

    def test_clamp_to_01(self):
        result = normalize_bbox([-10, -20, 1100, 1200], page_width=1000, page_height=1000)
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 1.0
        assert result[3] == 1.0

    def test_zero_page_size(self):
        # Zero page dimensions → return raw values
        result = normalize_bbox([100, 200, 300, 400], page_width=0, page_height=0)
        assert result == [100.0, 200.0, 300.0, 400.0]

    def test_tuple_input(self):
        result = normalize_bbox((50, 50, 150, 150), page_width=500, page_height=500)
        assert result == [0.1, 0.1, 0.3, 0.3]


# ── _png_dimensions ──────────────────────────────────────────────────


class TestPngDimensions:
    def test_valid_png_header(self):
        # Build a minimal valid PNG header with dimensions
        png_sig = b"\x89PNG\r\n\x1a\n"
        ihdr_start = b"\x00\x00\x00\r"  # length=13
        ihdr_type = b"IHDR"
        width = struct.pack(">I", 640)
        height = struct.pack(">I", 480)
        rest = b"\x08" * 5  # bit depth, color type, compression, filter, interlace
        data = png_sig + ihdr_start + ihdr_type + width + height + rest
        w, h = _png_dimensions(data)
        assert w == 640
        assert h == 480

    def test_too_short(self):
        assert _png_dimensions(b"short") == (0, 0)

    def test_empty(self):
        assert _png_dimensions(b"") == (0, 0)


# ── _is_logo ─────────────────────────────────────────────────────────


class TestIsLogo:
    def test_page0_bottom(self):
        entry = {"page_index": 0, "bbox": [0.1, 0.92, 0.9, 0.99]}
        assert _is_logo(entry) is True

    def test_page0_top(self):
        entry = {"page_index": 0, "bbox": [0.1, 0.1, 0.9, 0.3]}
        assert _is_logo(entry) is False

    def test_page1_bottom(self):
        # Only page 0 logos are filtered
        entry = {"page_index": 1, "bbox": [0.1, 0.92, 0.9, 0.99]}
        assert _is_logo(entry) is False

    def test_no_bbox(self):
        assert _is_logo({"page_index": 0}) is False

    def test_short_bbox(self):
        assert _is_logo({"page_index": 0, "bbox": [0.1, 0.9]}) is False


# ── load_image_index ─────────────────────────────────────────────────


class TestLoadImageIndex:
    def test_load_valid(self, tmp_path):
        index_file = tmp_path / "index.jsonl"
        entries = [
            {"pdf_path": "/test/a.pdf", "page_index": 0, "xref": 5,
             "bbox": [0.1, 0.2, 0.5, 0.8], "width": 800, "height": 600, "image_type": "bitmap"},
            {"pdf_path": "/test/a.pdf", "page_index": 1, "xref": 8,
             "bbox": [0.0, 0.0, 0.9, 0.9], "width": 1200, "height": 900, "image_type": "bitmap"},
            {"pdf_path": "/test/b.pdf", "page_index": 0, "xref": 3,
             "bbox": [0.2, 0.2, 0.8, 0.8], "width": 600, "height": 400, "image_type": "vector_path"},
        ]
        index_file.write_text("\n".join(json.dumps(e) for e in entries))

        result = load_image_index(index_file)
        assert len(result) == 2
        assert len(result["/test/a.pdf"]) == 2
        assert len(result["/test/b.pdf"]) == 1
        assert result["/test/b.pdf"][0]["image_type"] == "vector_path"

    def test_load_nonexistent(self, tmp_path):
        result = load_image_index(tmp_path / "nonexistent.jsonl")
        assert result == {}

    def test_load_with_bad_lines(self, tmp_path):
        index_file = tmp_path / "index.jsonl"
        index_file.write_text('{"valid": true}\nbad json\n{"also": "valid"}\n')

        result = load_image_index(index_file)
        # Should have 2 entries under "" key (no pdf_path)
        assert len(result) == 1
        assert len(result[""]) == 2


# ── prescan_pdfs (with nonexistent file) ─────────────────────────────


class TestPrescanPdfs:
    def test_nonexistent_pdf(self, tmp_path):
        """Nonexistent PDFs should be skipped gracefully."""
        entries = prescan_pdfs([tmp_path / "nonexistent.pdf"])
        assert entries == []

    def test_empty_list(self):
        entries = prescan_pdfs([])
        assert entries == []

    def test_save_index(self, tmp_path):
        """Test index file saving with no real PDFs (empty result)."""
        index_path = tmp_path / "output.jsonl"
        entries = prescan_pdfs([tmp_path / "fake.pdf"], index_path=index_path)
        assert index_path.exists()
        content = index_path.read_text()
        assert content == ""  # No entries for nonexistent PDF


# ── Data Types ───────────────────────────────────────────────────────


class TestFigureRegion:
    def test_creation(self):
        fig = FigureRegion(
            page_index=0,
            bbox=(0.1, 0.2, 0.5, 0.8),
            width=800,
            height=600,
            source="embedded_bitmap",
        )
        assert fig.page_index == 0
        assert fig.bbox == (0.1, 0.2, 0.5, 0.8)
        assert fig.source == "embedded_bitmap"

    def test_default_source(self):
        fig = FigureRegion(page_index=0, bbox=(0, 0, 1, 1), width=100, height=100)
        assert fig.source == "unknown"


# ── build_figure_hints ───────────────────────────────────────────────


class TestBuildFigureHints:
    def test_no_entries(self, tmp_path):
        from math_ocr.pdf import build_figure_hints
        # Empty index file → empty string
        index_file = tmp_path / "empty.jsonl"
        index_file.write_text("")
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        result = build_figure_hints(pdf_path, [0], index_path=index_file)
        assert result == ""

    def test_from_index_file(self, tmp_path):
        from math_ocr.pdf import build_figure_hints

        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        index_file = tmp_path / "index.jsonl"
        entry = {
            "pdf_path": str(pdf_path),
            "page_index": 0,
            "xref": 5,
            "bbox": [0.1, 0.2, 0.5, 0.8],
            "width": 800,
            "height": 600,
            "image_type": "bitmap",
        }
        index_file.write_text(json.dumps(entry) + "\n")

        result = build_figure_hints(pdf_path, [0], index_path=index_file)
        assert "hint_id=0" in result
        assert "batch_local_page_index=0" in result
        assert "800x600" in result
        assert "bitmap" in result

    def test_logo_filtered(self, tmp_path):
        from math_ocr.pdf import build_figure_hints

        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        index_file = tmp_path / "index.jsonl"
        entry = {
            "pdf_path": str(pdf_path),
            "page_index": 0,
            "xref": 1,
            "bbox": [0.1, 0.92, 0.9, 0.99],
            "width": 100,
            "height": 30,
            "image_type": "bitmap",
        }
        index_file.write_text(json.dumps(entry) + "\n")

        result = build_figure_hints(pdf_path, [0], index_path=index_file)
        assert result == ""  # Logo filtered out


# ── scan_page_signals ────────────────────────────────────────────────


class TestScanPageSignals:
    def test_nonexistent_file(self, tmp_path):
        from math_ocr.pdf import scan_page_signals
        result = scan_page_signals(tmp_path / "nonexistent.pdf")
        assert result == ""

    def test_real_pdf_with_questions(self, tmp_path):
        """Test with a real PDF containing English exam-style text."""
        from math_ocr.pdf import scan_page_signals
        import fitz

        # Create a minimal PDF with question text
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        text_point = fitz.Point(72, 72)
        page.insert_text(text_point, "Question 1\n(a) Solve x^2 + 3x - 1 = 0  [5 marks]\n(b) Find the derivative  [3 marks]\n\nQuestion 2\nFigure 1 shows the graph of y = x^2\nDiagram not drawn to scale")
        doc.save(str(pdf_path))
        doc.close()

        result = scan_page_signals(pdf_path)
        # Should detect question numbering
        assert "Question 1, Question 2" in result
        # Should detect marks
        assert "[3]" in result or "[5]" in result
        # Should detect figure/diagram labels
        assert "Figure 1" in result or "Diagram" in result

    def test_real_pdf_chinese(self, tmp_path):
        """Test language detection with Chinese text via Unicode content stream."""
        from math_ocr.pdf import scan_page_signals
        import fitz

        # Use a font that supports CJK (fitz built-in CJK font)
        pdf_path = tmp_path / "chinese.pdf"
        doc = fitz.open()
        page = doc.new_page()
        # Use insert_text with CJK-capable fontname
        try:
            page.insert_text(
                fitz.Point(72, 72),
                "第一题 求解方程 （10分） 第二题 证明不等式 （15分）",
                fontname="china-s",  # PyMuPDF built-in CJK serif font
            )
        except Exception:
            # If CJK font not available, use a text block directly
            tw = fitz.TextWriter(page.rect)
            # Fallback: just test the signal scanner logic with the raw text function
            doc.close()
            # Skip this test if we can't create Chinese PDF
            pytest.skip("CJK font not available in this PyMuPDF build")
        doc.save(str(pdf_path))
        doc.close()

        result = scan_page_signals(pdf_path)
        # Should detect Chinese text or at least Chinese mark patterns
        assert "Chinese" in result or "分" in result

    def test_real_pdf_plain(self, tmp_path):
        """Test with a PDF that has no distinctive signals."""
        from math_ocr.pdf import scan_page_signals
        import fitz

        pdf_path = tmp_path / "plain.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(fitz.Point(72, 72), "The quick brown fox jumps over the lazy dog. This is a test of plain text with no mathematical content or question structures.")
        doc.save(str(pdf_path))
        doc.close()

        result = scan_page_signals(pdf_path)
        # May just detect language, but no question/marks/figure signals
        # Result could be "" if only "Latin/English" is detected
        # or it could have just the language signal
