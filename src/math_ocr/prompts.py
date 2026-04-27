"""Prompt templates for math OCR.

Modular prompt builder: each section is a named function that returns a string
section. Users can override any section via ``prompt_overrides`` or add extra
rules via ``custom_rules``. The ``doc_type`` parameter adds context hints to
help the LLM adapt its output format without hardcoding schemas.
"""

from __future__ import annotations


# ── Doc Type Presets ──────────────────────────────────────────────────

_DOC_TYPE_HINTS: dict[str, str] = {
    "exam_qp": (
        "This document appears to be an exam question paper. "
        "Include question numbers and sub-part labels (e.g. Q1, 1(a), (ii)) in the output. "
        "Preserve the full question text verbatim."
    ),
    "exam_ms": (
        "This document appears to be a marking scheme or answer key. "
        "Include question numbers, mark allocations, answer content, and any method/guidance notes. "
        "Preserve alternative answer paths where visible."
    ),
    "textbook": (
        "This document appears to be a textbook or reference material. "
        "Preserve the hierarchical structure (sections, examples, exercises). "
        "Keep prose and explanation text alongside formulas."
    ),
}


# ── Section Builders ──────────────────────────────────────────────────


def _build_role(**_) -> str:
    return (
        "You are a precision math content extraction engine. Your job is to extract "
        "all mathematical content from page images with absolute fidelity — every "
        "symbol, every subscript, every fraction must be captured exactly as it "
        "appears in the image."
    )


def _build_latex_rules(**_) -> str:
    return """\
## LATEX RULES (MANDATORY)

1. Wrap ALL mathematical expressions in \\(...\\) for inline or \\[...\\] for display.
   NEVER use $ or $$ as math delimiters.
2. Keep ordinary prose as plain text OUTSIDE of \\(...\\). Do NOT convert entire \
sentences into LaTeX.
3. Replace Unicode math symbols with LaTeX commands:
   × → \\times,  ÷ → \\div,  ≤ → \\le,  ≥ → \\ge,  ≈ → \\approx,
   π → \\pi,  θ → \\theta,  ∞ → \\infty,  → → \\to.
4. Use \\frac{}{} only for actual fractions. Leave plain words plain.
5. If a math span is uncertain, prefer a short plain-text approximation \
over a long malformed LaTeX expression.
6. Output MUST be valid JSON strings. In JSON, every backslash must be doubled \
(e.g. \\\\frac, \\\\times, \\\\mathrm{H}_0).

### Examples

BAD:  "text": "$a = 2$"
GOOD: "text": "\\\\(a = 2\\\\)"

BAD:  "text": "\\\\frac{dy}{dx} = 3x"
GOOD: "text": "Find \\\\(\\\\frac{dy}{dx}\\\\) when \\\\(x = 5\\\\)"

BAD:  "text": "\\\\(The curve C passes through\\\\)"
GOOD: "text": "The curve C passes through \\\\(A(2, 3)\\\\)"

GOOD: "\\\\(\\\\frac{0-(-10)}{\\\\sqrt{30}}\\\\)"
GOOD: "\\\\(x^2 + 3x - 1\\\\)"
GOOD: "\\\\(\\\\int_{0}^{2} (3x^2 - 2x)\\\\,dx\\\\)"\
"""


def _build_output_budget(*, max_items: int = 25, content_only: bool = False, **_) -> str:
    if content_only:
        skip_rule = (
            "3. ONLY extract actual content: questions, answers, explanations, formulas, "
            "or figures. Cover pages, instruction pages, blank pages, administrative text "
            "(exam board info, instructions, \"turn over\" notices, blank page markers) "
            "should output []."
        )
    else:
        skip_rule = (
            "3. Cover pages and administrative pages should output []. "
            "Instruction pages may be extracted if they contain useful information."
        )
    return f"""\
## OUTPUT BUDGET (MANDATORY)

1. Return ONLY the minimum JSON needed. No explanations, no commentary, \
no repetition, no markdown fences.
2. Do not repeat decorative elements (repeated headers, watermarks, page numbers \
that appear on every page). Do include content-bearing text even if it looks \
like a header or label.
{skip_rule}
4. If the page is noisy or ambiguous, output fewer objects rather than \
speculative extra objects.
5. Never emit duplicate objects for the same visible item.
6. If there is no real extractable content, output [].
7. At most {max_items} objects per page.\
"""


def _build_figure_rules(**_) -> str:
    return """\
## FIGURE RULES

1. figure_bbox uses normalized 0-1 image coordinates: [x0, y0, x1, y1] \
where 0.0 is top-left and 1.0 is bottom-right. Do NOT output pixel coordinates.
2. page_index is 0-based within the provided batch of page images.
3. If a row/item contains a figure/diagram/graph with no accompanying text, \
still extract it: set has_figure=true, provide figure_bbox, and use "" for \
empty text fields. A figure-only row is NOT an empty row.\
"""


def _build_json_format(**_) -> str:
    return """\
## OUTPUT FORMAT

Output a JSON array. Each element is an object with these fields:
{
  "page_index": 0,
  "text": "the extracted content with \\\\(\\\\ LaTeX \\\\)\\\\)",
  "has_figure": false,
  "figure_bbox": null,
  "figure_description": null
}

For figure-containing items:
{
  "page_index": 0,
  "text": "description or accompanying text",
  "has_figure": true,
  "figure_bbox": [0.1, 0.2, 0.5, 0.8],
  "figure_description": "brief description of the figure"
}\
"""


def _build_markdown_format(**_) -> str:
    return """\
## OUTPUT FORMAT

Output a markdown document with extracted math content.
Use \\(...\\) for inline math and \\[...\\] for display math.
Separate distinct items with horizontal rules (---).\
"""


def _build_doc_type_hint(*, doc_type: str | None = None, **_) -> str:
    hint = _DOC_TYPE_HINTS.get(doc_type or "", "")
    if not hint:
        return ""
    return f"## Document Context\n{hint}"


def _build_page_signals(*, page_signals: str | None = None, **_) -> str:
    if not page_signals:
        return ""
    return f"## Page Analysis\n{page_signals}"


# ── Section Registry ──────────────────────────────────────────────────

# Order matters — sections are joined in this order.
SECTION_BUILDERS: list[tuple[str, object]] = [
    ("role", _build_role),
    ("latex_rules", _build_latex_rules),
    ("output_budget", _build_output_budget),
    ("figure_rules", _build_figure_rules),
    ("json_format", _build_json_format),
    ("markdown_format", _build_markdown_format),
    ("doc_type_hint", _build_doc_type_hint),
    ("page_signals", _build_page_signals),
]


# ── Public API ────────────────────────────────────────────────────────


def math_extraction_prompt(
    *,
    detect_figures: bool = False,
    max_items_per_page: int = 25,
    output_format: str = "json_array",
    doc_type: str | None = None,
    page_signals: str | None = None,
    content_only: bool = False,
    custom_rules: list[str] | None = None,
    prompt_overrides: dict[str, str] | None = None,
) -> str:
    """Build a system prompt for math OCR extraction.

    Parameters
    ----------
    detect_figures : bool
        Whether to include figure detection rules.
    max_items_per_page : int
        Hard cap on extracted items per page.
    output_format : str
        "json_array" (default) or "markdown".
    doc_type : str, optional
        Document type hint: "exam_qp", "exam_ms", "textbook", or None for generic.
    page_signals : str, optional
        Neutral page analysis signals from scan_page_signals().
    content_only : bool
        If True, strictly skip cover/instruction/blank pages. If False, extract
        all content but still skip purely decorative elements.
    custom_rules : list[str], optional
        Additional rules appended to the prompt.
    prompt_overrides : dict[str, str], optional
        Replace entire sections by name. Keys are section names from the registry
        (e.g. "role", "latex_rules", "json_format"). Values are replacement strings.

    Returns
    -------
    str
        Complete system prompt.
    """
    overrides = prompt_overrides or {}
    kwargs = {
        "max_items": max_items_per_page,
        "doc_type": doc_type,
        "page_signals": page_signals,
        "content_only": content_only,
    }

    parts: list[str] = []

    for section_name, builder in SECTION_BUILDERS:
        # Skip figure rules if not detecting figures
        if section_name == "figure_rules" and not detect_figures:
            continue
        # Only include one format section
        if section_name == "json_format" and output_format != "json_array":
            continue
        if section_name == "markdown_format" and output_format != "markdown":
            continue

        # Check for override
        if section_name in overrides:
            content = overrides[section_name]
        else:
            content = builder(**kwargs)

        if content:
            parts.append(content)

    if custom_rules:
        parts.append("\n## Additional Rules\n" + "\n".join(f"- {r}" for r in custom_rules))

    return "\n\n".join(parts)


def user_extraction_prompt(
    *,
    page_count: int = 1,
    hint_text: str = "",
) -> str:
    """Build the user-level extraction prompt.

    Parameters
    ----------
    page_count : int
        Number of page images provided.
    hint_text : str, optional
        Pre-detected figure hints to inject.
    """
    parts = [
        f"Extract all mathematical content from the {page_count} page image(s) below.",
        "Output a JSON array where each element represents one distinct mathematical item.",
    ]

    if hint_text:
        parts.append("\n" + hint_text)

    return "\n".join(parts)
