"""Example: Marking scheme extraction with custom rules.

    python examples/marking_scheme.py
"""

from math_ocr import MathOCR

ocr = MathOCR()

results = ocr.extract_pdf(
    "marking_scheme.pdf",
    doc_type="exam_ms",
    content_only=True,
    custom_rules=[
        "Preserve M1, A1, B1 method mark annotations",
        "Include alternative answer paths where visible",
    ],
)

for item in results:
    print(f"  p{item['page_index']}: {item['text'][:100]}...")
