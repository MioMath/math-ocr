"""Example: Basic PDF extraction.

    python examples/basic_extraction.py
"""

from math_ocr import MathOCR

ocr = MathOCR()

# Extract text and figures from a math exam paper
results = ocr.extract_pdf(
    "exam.pdf",
    doc_type="exam_qp",       # hint: question paper format
    content_only=True,         # skip cover/instruction pages
    figures_dir="./figures",   # save cropped PNG+SVG to disk
)

for item in results:
    page = item["page_index"]
    text = item["text"][:80]
    fig = " [FIGURE]" if item.get("has_figure") else ""
    print(f"  p{page}: {text}...{fig}")

print(f"\nExtracted {len(results)} items total")
