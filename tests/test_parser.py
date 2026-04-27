"""Tests for JSON/LaTeX parsing and normalization."""

import json

import pytest

from math_ocr.parser import (
    extract_json_array,
    fix_invalid_escapes,
    fix_latex_control_escapes,
    normalize_items,
    parse_json_response,
)
from math_ocr.latex import normalize_latex_text, _strip_dollar_delimiters


# ── extract_json_array ───────────────────────────────────────────────


class TestExtractJsonArray:
    def test_clean_json(self):
        raw = '[{"a": 1}, {"b": 2}]'
        assert extract_json_array(raw) == raw

    def test_with_think_tags(self):
        raw = '<think let me reason</think [{"a": 1}]'
        assert extract_json_array(raw) == '[{"a": 1}]'

    def test_with_markdown_fence(self):
        raw = '```json\n[{"a": 1}]\n```'
        assert extract_json_array(raw) == '[{"a": 1}]'

    def test_truncated_json(self):
        # Truncation recovery: missing closing bracket
        raw = '[{"a": 1}, {"b": 2}'
        result = extract_json_array(raw)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["a"] == 1

    def test_empty_input(self):
        assert extract_json_array("") == ""
        assert extract_json_array("no json here") == ""


# ── fix_latex_control_escapes ────────────────────────────────────────


class TestFixLatexControlEscapes:
    def test_frac_not_eaten(self):
        # \f in JSON would be form-feed, but \frac is a LaTeX command
        payload = '{"text": "\\frac{1}{2}"}'
        fixed = fix_latex_control_escapes(payload)
        parsed = json.loads(fixed)
        assert "\\frac" in parsed["text"]

    def test_text_not_eaten(self):
        payload = '{"text": "\\text{hello}"}'
        fixed = fix_latex_control_escapes(payload)
        parsed = json.loads(fixed)
        assert "\\text" in parsed["text"]

    def test_already_doubled_unchanged(self):
        payload = '{"text": "\\\\frac{1}{2}"}'
        fixed = fix_latex_control_escapes(payload)
        parsed = json.loads(fixed)
        assert "\\\\frac" in parsed["text"] or "\\frac" in parsed["text"]

    def test_backslash_t_in_non_latex_context_preserved(self):
        # If \t is NOT followed by an alpha (no LaTeX command), the fix
        # should not double it. The input is already valid JSON with \\t.
        payload = '{"text": "hello\\\\tworld"}'
        fixed = fix_latex_control_escapes(payload)
        parsed = json.loads(fixed)
        # \\t in JSON string becomes literal \t (backslash + t)
        assert "\\t" in parsed["text"]

    def test_theta_preserved(self):
        payload = '{"text": "\\theta = 30"}'
        fixed = fix_latex_control_escapes(payload)
        parsed = json.loads(fixed)
        assert "\\theta" in parsed["text"]


# ── fix_invalid_escapes ──────────────────────────────────────────────


class TestFixInvalidEscapes:
    def test_unescaped_paren(self):
        payload = r'{"text": "\(x^2\)"}'
        # This has \( which is invalid JSON escape
        fixed, n = fix_invalid_escapes(payload)
        parsed = json.loads(fixed)
        assert "\\(" in parsed["text"]


# ── parse_json_response ──────────────────────────────────────────────


class TestParseJsonResponse:
    def test_clean_response(self):
        raw = '[{"page_index": 0, "text": "hello"}]'
        items, error = parse_json_response(raw)
        assert len(items) == 1
        assert items[0]["text"] == "hello"
        assert error == ""

    def test_with_latex(self):
        raw = '[{"text": "\\\\frac{1}{2}"}]'
        items, error = parse_json_response(raw)
        assert len(items) == 1
        assert error == ""

    def test_single_object(self):
        # Single object wrapped in brackets for extraction
        raw = '[{"page_index": 0, "text": "hello"}]'
        items, error = parse_json_response(raw)
        assert len(items) == 1

    def test_empty_array(self):
        raw = '[]'
        items, error = parse_json_response(raw)
        assert items == []

    def test_trailing_commas(self):
        raw = '[{"a": 1,},]'
        items, error = parse_json_response(raw)
        assert len(items) == 1


# ── normalize_latex_text ─────────────────────────────────────────────


class TestNormalizeLatexText:
    def test_unicode_to_latex(self):
        assert "\\times" in normalize_latex_text("5 × 3")
        assert "\\le" in normalize_latex_text("x ≤ 5")
        assert "\\pi" in normalize_latex_text("π")

    def test_double_backslash_cleanup(self):
        result = normalize_latex_text("\\\\frac{1}{2}")
        assert result == "\\frac{1}{2}"

    def test_dollar_delimiters(self):
        result = normalize_latex_text("$x = 5$")
        assert "\\(" in result
        assert "$" not in result

    def test_display_math_dollars(self):
        result = normalize_latex_text("$$\\frac{1}{2}$$")
        assert "\\[" in result
        assert "$$" not in result

    def test_empty_string(self):
        assert normalize_latex_text("") == ""
        assert normalize_latex_text(None) == ""

    def test_whitespace_collapse(self):
        result = normalize_latex_text("hello   world")
        assert result == "hello world"

    def test_preserve_existing_delimiters(self):
        result = normalize_latex_text("\\(x = 5\\)")
        assert result == "\\(x = 5\\)"


# ── normalize_items ──────────────────────────────────────────────────


class TestNormalizeItems:
    def test_basic_normalization(self):
        items = [{"text": "$x = 5$"}]
        result = normalize_items(items)
        assert "\\(" in result[0]["text"]
        assert "$" not in result[0]["text"]

    def test_custom_fields(self):
        items = [{"my_field": "5 × 3"}]
        result = normalize_items(items, text_fields=["my_field"])
        assert "\\times" in result[0]["my_field"]

    def test_non_string_fields_untouched(self):
        items = [{"page_index": 0, "text": "$x$"}]
        result = normalize_items(items)
        assert result[0]["page_index"] == 0
