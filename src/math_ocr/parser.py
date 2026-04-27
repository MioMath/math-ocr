"""JSON repair and LaTeX normalization for math OCR outputs.

Vision LLMs frequently produce malformed JSON when LaTeX math is involved,
because backslash commands like \\frac, \\text, \\theta collide with JSON
escape sequences (\\f, \\t, \\n, \\r). This module provides battle-tested
repair logic extracted from production OCR pipelines.
"""

from __future__ import annotations

import json
import re
from typing import Any

from math_ocr.latex import normalize_latex_text


# ── JSON Extraction ──────────────────────────────────────────────────


def extract_json_array(raw: str) -> str:
    """Extract a JSON array blob from raw LLM output.

    Handles:
    - <think/thinking> tags (model reasoning traces)
    - Markdown code fences
    - Truncated JSON (attempts bracket recovery)
    - Split arrays ("][" at top level)

    Returns the extracted blob, or "" if nothing found.
    """
    # Strip thinking tags
    text = re.sub(r"<(?:think|thinking)>[\s\S]*?</(?:think|thinking)>", "", raw or "")
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\n?", "", text).strip().rstrip("`")

    s, e = text.find("["), text.rfind("]")
    if s == -1:
        return ""
    if e == -1 or s >= e:
        recovered = _try_close_truncated_json(text)
        return recovered
    blob = text[s : e + 1]
    # Handle double-]] wrapper
    blob = re.sub(r"\]\s*\]$", "]", blob)
    # Merge top-level ][ boundaries
    blob = re.sub(r"\](\s*)\[(?![^{]*\})", lambda m: "," + m.group(1), blob)
    return blob


def _try_close_truncated_json(text: str) -> str:
    """Recover a truncated JSON array by closing open brackets."""
    s = text.find("[")
    if s == -1:
        return ""
    blob = text[s:].rstrip()
    blob = re.sub(r',\s*"[^"]*$', "", blob)
    blob = re.sub(r",\s*$", "", blob)

    stack: list[str] = []
    in_string = False
    escape = False
    for ch in blob:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    close_map = {"{": "}", "[": "]"}
    blob += "".join(close_map[c] for c in reversed(stack))
    return blob


# ── JSON Repair ──────────────────────────────────────────────────────


def fix_latex_control_escapes(payload: str) -> str:
    """Fix single-backslash LaTeX commands that collide with JSON escapes.

    JSON parsers silently interpret \\f as form-feed, \\t as tab, etc.,
    corrupting LaTeX commands like \\frac, \\text, \\theta, \\nu, \\rho,
    \\beta, \\alpha, \\nabla.

    Only touches backslashes inside JSON string values.
    """
    result = []
    in_string = False
    escape = False
    i = 0
    while i < len(payload):
        ch = payload[i]
        if escape:
            escape = False
            result.append(ch)
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue
        if in_string and ch == '\\' and i + 1 < len(payload):
            next_ch = payload[i + 1]
            if next_ch == '\\':
                result.append(ch)
                result.append(next_ch)
                i += 2
            elif next_ch == '"':
                result.append(ch)
                result.append(next_ch)
                i += 2
            elif (
                next_ch in ('f', 't', 'n', 'r', 'b')
                and i + 2 < len(payload)
                and (payload[i + 2].isalpha() or payload[i + 2] == '{')
            ):
                # Looks like a LaTeX command — double the backslash
                result.append('\\')
                result.append('\\')
                i += 1
            else:
                result.append(ch)
                i += 1
        else:
            result.append(ch)
            i += 1
    return ''.join(result)


def fix_invalid_escapes(payload: str, max_rounds: int = 200) -> tuple[str, int]:
    """Repair invalid JSON escapes (e.g. \\(, \\frac) by inserting backslashes."""
    repaired = payload
    fix_count = 0

    for _ in range(max_rounds):
        try:
            json.loads(repaired)
            return repaired, fix_count
        except json.JSONDecodeError as e:
            if "escape" not in str(e).lower():
                raise
            pos = e.pos
            if pos < 0 or pos >= len(repaired) or repaired[pos] != "\\":
                raise
            repaired = repaired[:pos] + "\\" + repaired[pos:]
            fix_count += 1

    raise json.JSONDecodeError("too many invalid escape repairs", repaired, 0)


# ── Main Parse Function ──────────────────────────────────────────────


def parse_json_response(raw: str) -> tuple[list[dict[str, Any]], str]:
    """Parse an LLM response into a list of dicts with robust repair.

    Returns (items, error_str). error_str is "" on success.
    """
    blob = extract_json_array(raw)
    if not blob:
        return [], "no JSON array found"

    # Pre-fix LaTeX/JSON collisions
    blob = fix_latex_control_escapes(blob)

    candidates = [
        blob,
        re.sub(r",\s*([}\]])", r"\1", blob),  # trailing commas
    ]

    last_error = ""
    seen: set[str] = set()
    for idx, candidate in enumerate(candidates, 1):
        if candidate in seen:
            continue
        seen.add(candidate)

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                repaired, n = fix_invalid_escapes(candidate)
                parsed = json.loads(repaired)
            except Exception as ex:
                last_error = str(ex)
                continue

        if isinstance(parsed, list):
            return parsed, ""
        if isinstance(parsed, dict):
            return [parsed], ""
        return [], f"unexpected JSON type: {type(parsed).__name__}"

    return [], last_error


def normalize_items(items: list[dict[str, Any]], *, text_fields: list[str] | None = None) -> list[dict[str, Any]]:
    """Apply LaTeX normalization to text fields in extracted items.

    Parameters
    ----------
    items : list[dict]
        Parsed OCR output items.
    text_fields : list[str], optional
        Fields to normalize. Default: common field names used in math OCR.
    """
    if text_fields is None:
        text_fields = [
            "text", "stem_text", "raw_text",
            "answer_latex", "answer", "guidance_latex", "guidance",
            "question_text", "solution",
        ]

    for item in items:
        for fname in text_fields:
            val = item.get(fname)
            if isinstance(val, str):
                item[fname] = normalize_latex_text(val)
    return items
