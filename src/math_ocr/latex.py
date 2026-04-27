"""LaTeX normalization utilities.

Handles the messy reality of LLM-generated LaTeX:
- Unicode math symbols → LaTeX commands
- Double-backslash from JSON → single-backslash
- Dollar delimiters → \\(...\\) / \\[...\\]
- Whitespace cleanup
"""

from __future__ import annotations

import re

# Unicode math symbols that LLMs sometimes output instead of LaTeX
UNICODE_TO_LATEX = {
    "×": r"\times",
    "÷": r"\div",
    "−": "-",  # minus sign → hyphen-minus
    "≤": r"\le",
    "≥": r"\ge",
    "≈": r"\approx",
    "∞": r"\infty",
    "π": r"\pi",
    "θ": r"\theta",
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "λ": r"\lambda",
    "μ": r"\mu",
    "σ": r"\sigma",
    "τ": r"\tau",
    "φ": r"\phi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Σ": r"\Sigma",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
    "→": r"\to",
    "←": r"\leftarrow",
    "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow",
    "⇐": r"\Leftarrow",
    "±": r"\pm",
    "∓": r"\mp",
    "≠": r"\ne",
    "∝": r"\propto",
    "√": r"\sqrt",
    "∂": r"\partial",
    "∇": r"\nabla",
    "∈": r"\in",
    "∉": r"\notin",
    "⊂": r"\subset",
    "⊃": r"\supset",
    "∪": r"\cup",
    "∩": r"\cap",
    "∅": r"\emptyset",
    "∀": r"\forall",
    "∃": r"\exists",
    "…": r"\ldots",
    "⋯": r"\cdots",
    "⋮": r"\vdots",
    "·": r"\cdot",
}


def normalize_latex_text(s: str | None) -> str:
    """Normalize LaTeX in a text string to single-backslash runtime semantics.

    Pipeline:
    1. Unicode math symbols → LaTeX commands
    2. Double-backslash (JSON artifact) → single-backslash
    3. Dollar delimiters → \\(...\\) / \\[...\\]
    4. Whitespace cleanup
    """
    if not s:
        return ""

    # 1. Unicode → LaTeX
    for char, latex in UNICODE_TO_LATEX.items():
        s = s.replace(char, latex)

    # 2. Double-backslash → single-backslash (JSON artifact)
    s = re.sub(r"\\\\+([A-Za-z]+)", r"\\\1", s)
    s = s.replace("\\\\(", "\\(").replace("\\\\)", "\\)")
    s = s.replace("\\\\[", "\\[").replace("\\\\]", "\\]")

    # 3. Fix dollar delimiters
    s = _strip_dollar_delimiters(s)

    # 4. Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_dollar_delimiters(s: str) -> str:
    """Remove $/$$ math delimiters, replacing with \\(...\\) / \\[...\\]."""
    if not s or "$" not in s:
        return s

    def _strip_line(line: str) -> str:
        stripped = line.strip()
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            inner = stripped[2:-2]
            if "$" not in inner:
                return f"\\[{inner.strip()}\\]"
        if stripped.startswith("$") and stripped.endswith("$") and len(stripped) > 2:
            inner = stripped[1:-1]
            if "$" not in inner:
                return f"\\({inner.strip()}\\)"
        return re.sub(r"\$\$?([^$]+?)\$\$?", r"\\(\1\\)", line)

    if "\n" in s:
        return "\n".join(_strip_line(line) for line in s.split("\n"))
    return _strip_line(s)


