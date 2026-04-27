# math-ocr

[![PyPI](https://img.shields.io/pypi/v/math-ocr.svg?color=blue)](https://pypi.org/project/math-ocr/)
[![Python](https://img.shields.io/pypi/pyversions/math-ocr.svg)](https://pypi.org/project/math-ocr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/MioMath/math-ocr/actions/workflows/ci.yml/badge.svg)](https://github.com/MioMath/math-ocr/actions/workflows/ci.yml)

Turn math PDFs into structured JSON with LaTeX text **and cropped figure images**.

**What makes this different:** Every other tool (MinerU, Docling, Marker) re-renders figures as low-res screenshots. math-ocr uses IoU matching to extract the **original embedded images** from PDFs — full resolution, no quality loss. It also detects vector-path drawings (geometry, coordinate systems) that have no bitmap at all.

## Highlights

- **Original-quality figure extraction** — IoU-matched embedded images, not re-rendered screenshots
- **Vector path detection** — finds geometry diagrams, coordinate systems, and drawings with no bitmap
- **Robust JSON repair** — handles truncated output, broken escapes, LaTeX/JSON collisions
- **LaTeX normalization** — Unicode → LaTeX, dollar delimiters → `\(...\)`, backslash cleanup
- **Provider-agnostic** — works with Gemini, GPT-4o, Ollama, any OpenAI-compatible API
- **Pre-scan index** — catalog figures once, reuse across extractions
- **SVG + PNG output** — every cropped figure in both formats

## Table of Contents

- [What it does](#what-it-does)
- [Figure extraction approach](#figure-extraction-approach)
- [Install](#install)
- [Quick Start](#quick-start)
- [CLI Commands](#cli-commands)
- [Output Format](#output-format)
- [LaTeX Normalization](#latex-normalization)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Comparison with alternatives](#comparison-with-alternatives)
- [Examples](examples/)
- [License](#license)

## What it does

Given a math exam paper or textbook PDF:

1. **Extracts text** — structured JSON with properly normalized LaTeX
2. **Locates figures** — normalized bounding boxes `[x0, y0, x1, y1]` in 0-1 coordinates
3. **Crops figures** — extracts embedded originals at full resolution, or renders page regions as fallback
4. **Outputs PNG + SVG** — each cropped figure saved with both formats

```json
{
  "page_index": 0,
  "text": "Find \\(\\frac{dy}{dx}\\) when \\(x = 5\\)",
  "has_figure": true,
  "figure_bbox": [0.1, 0.2, 0.5, 0.8],
  "figure_description": "Graph of y = x² + 3x - 1",
  "figure_source": "embedded"
}
```

## Figure extraction approach

| Step | What happens |
|------|-------------|
| **Pre-scan** | Scan PDF for embedded bitmaps (via PyMuPDF xrefs) and vector paths (drawing commands) |
| **LLM detection** | Vision model identifies which page regions contain figures and returns bounding boxes |
| **IoU matching** | Match LLM-detected regions against the pre-scan index using Intersection-over-Union |
| **Extract original** | Pull the embedded PNG/JPEG at its native resolution — not a re-rendered screenshot |
| **Fallback render** | If no embedded image matches, render the page region at high DPI |

Why this matters: A figure that's a 1200x900 embedded JPEG in the PDF gets extracted as exactly that 1200x900 image. Competitors render it at whatever DPI they're using (typically 72-150), producing a blurry 400x300 screenshot.

## Install

```bash
pip install math-ocr
```

Requires Python 3.10+. Uses PyMuPDF for PDF processing and any OpenAI-compatible API for vision (Gemini, GPT-4o, Ollama, etc.).

## Quick Start

### CLI

```bash
# Set your API key
export MATH_OCR_API_KEY="your-key"

# Extract text + figures from a PDF
math-ocr extract exam.pdf -o result.json

# Pre-scan to build a figure index (faster subsequent extractions)
math-ocr prescan exam.pdf -o index.jsonl

# Extract using the pre-scan index for better figure accuracy
math-ocr extract exam.pdf --index index.jsonl --figures-dir ./figures

# Extract only content pages (skip cover, instructions, blank pages)
math-ocr extract exam.pdf --content-only

# Tell the model what kind of document this is
math-ocr extract exam.pdf --doc-type exam_qp
math-ocr extract marking_scheme.pdf --doc-type exam_ms
math-ocr extract textbook.pdf --doc-type textbook

# Extract Chinese math exam papers
math-ocr extract 高考数学.pdf --lang zh

# Custom extraction rules
math-ocr extract exam.pdf --custom-rules "Include difficulty level,Include topic tags"
math-ocr extract exam.pdf --custom-rules @my_rules.txt

# Use a different model
math-ocr extract exam.pdf --model gpt-4o --base-url https://api.openai.com/v1
```

### Python API

```python
from math_ocr import MathOCR, OCRConfig
from pathlib import Path

ocr = MathOCR()

# Basic extraction
results = ocr.extract_pdf("exam.pdf")
for item in results:
    print(item["text"])

# Only extract content pages (skip covers, instructions, blank pages)
results = ocr.extract_pdf("exam.pdf", content_only=True)

# Document type hint for better output
results = ocr.extract_pdf("exam.pdf", doc_type="exam_qp")

# With figure export to files
results = ocr.extract_pdf(
    "exam.pdf",
    figures_dir=Path("./figures"),  # saves PNG + SVG per figure
)

# With pre-scan index for better figure accuracy
results = ocr.extract_pdf(
    "exam.pdf",
    index_path=Path("index.jsonl"),
    figures_dir=Path("./figures"),
)

# From an image
results = ocr.extract_image("formula_photo.png")

# Custom provider
config = OCRConfig(
    api_key="your-key",
    base_url="https://api.openai.com/v1",
    model="gpt-4o",
    dpi=200,
)
ocr = MathOCR(config=config)
```

## CLI Commands

### `math-ocr extract`

Extract math content from a PDF or image.

```
math-ocr extract INPUT_PATH [OPTIONS]

Options:
  -o, --output       Output JSON file path
  --pages            Page indices (comma-separated, 0-based)
  --no-figures       Skip figure detection
  --preprocess       Apply contrast enhancement and resize
  --index            Pre-scan JSONL index file for figure hints
  --figures-dir      Directory to save cropped figure files (PNG + SVG)
  --doc-type         Document type hint: generic, exam_qp, exam_ms, textbook
  --lang             Content language hint (e.g. en, zh, ja)
  --content-only     Only extract content pages (skip cover, instructions, blanks)
  --custom-rules     Extra prompt rules (comma-separated, or @file.txt)
  --model            LLM model name
  --dpi              PDF render DPI (default: 150)
  --api-key          API key
  --base-url         API base URL
  --max-tokens       Max output tokens
  --timeout          Hard timeout in seconds
  -v, --verbose      Debug logging
```

### `math-ocr prescan`

Pre-scan PDFs to build a figure index without calling any LLM.

```
math-ocr prescan INPUT_PATHS... [OPTIONS]

Options:
  -o, --output       Output JSONL index file path
  -v, --verbose      Debug logging
```

### `math-ocr config`

Show or save configuration.

```
math-ocr config [--set-api-key KEY] [--set-base-url URL] [--set-model MODEL]
```

## Output Format

Each extracted item is a dict:

| Field | Type | Description |
|-------|------|-------------|
| `page_index` | int | 0-based page number |
| `text` | string | Extracted content with normalized LaTeX |
| `has_figure` | bool | Whether a figure was detected |
| `figure_bbox` | list[float] | Normalized 0-1 bounding box `[x0, y0, x1, y1]` |
| `figure_description` | string | Brief description of the figure |
| `figure_b64` | string | Base64-encoded cropped figure (when not using `figures_dir`) |
| `figure_source` | string | `"embedded"` (original) or `"page_crop"` (rendered) |
| `figure_png` | string | Path to saved PNG file (when using `figures_dir`) |
| `figure_svg` | string | Path to saved SVG file (when using `figures_dir`) |

## LaTeX Normalization

The pipeline normalizes raw LLM output:

- Unicode symbols → LaTeX: `×` → `\times`, `≤` → `\le`, `π` → `\pi`
- Dollar delimiters → parentheses: `$x = 5$` → `\(x = 5\)`
- Double backslash cleanup: `\\frac` → `\frac`
- JSON escape repair: `\frac` not eaten by `\f` form-feed, `\theta` preserved

## Configuration

Config priority (later wins): **defaults < config file < env vars < CLI flags / constructor args**.

### Config File

Save your API key and defaults so you don't need to set them every time:

```bash
# Save API key (stored in ~/.math-ocr.json)
math-ocr config --set-api-key "your-key"

# Switch to OpenAI
math-ocr config --set-base-url "https://api.openai.com/v1" --set-model "gpt-4o"

# Show current config
math-ocr config
```

The config file is `~/.math-ocr.json`:

```json
{
  "api_key": "your-key",
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4o"
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MATH_OCR_API_KEY` | API key (falls back to `OPENAI_API_KEY`) | — |
| `MATH_OCR_BASE_URL` | API endpoint (falls back to `OPENAI_BASE_URL`) | Gemini |
| `MATH_OCR_MODEL` | Model to use | `gemini-2.5-flash-preview-05-20` |

### Python Config

```python
from math_ocr import OCRConfig

config = OCRConfig(
    api_key="your-key",
    base_url="https://api.openai.com/v1",
    model="gpt-4o",
    dpi=200,
    figure_crop_dpi=300,
    figure_padding=0.01,
    extract_embedded_images=True,
    max_tokens=32768,
)
```

## How It Works

```
PDF → Pre-scan (bitmap + vector index)
    → Render pages to PNG
    → Send to Vision LLM with figure hints
    → Parse JSON (repair truncated output, fix LaTeX escapes)
    → Normalize LaTeX (unicode, delimiters, backslashes)
    → Crop figures (IoU match → extract embedded original → fallback render)
    → Output JSON + PNG + SVG
```

1. **Pre-scan** — Catalog all embedded bitmaps (xref, bbox, dimensions) and vector drawings (path regions)
2. **Render** — PDF pages converted to base64 PNG at configurable DPI
3. **LLM call** — Streaming API call with figure hints from the pre-scan index
4. **JSON repair** — Extracts JSON array from messy LLM output, fixes broken escapes, handles truncation
5. **LaTeX normalize** — Unicode → LaTeX, dollar → `\(...\)`, backslash cleanup
6. **Figure crop** — IoU match against embedded originals, extract at full resolution, save PNG + SVG

## Comparison with alternatives

| Feature | math-ocr | MinerU | Docling | Marker | pix2tex |
|---------|----------|--------|---------|--------|---------|
| LaTeX text extraction | Yes | Yes | Yes | Yes | Yes |
| Figure detection + bbox | Yes | Yes | Yes | Partial | No |
| **Extract embedded originals** | **Yes** | No | No | No | No |
| Vector path detection | Yes | No | No | No | No |
| SVG output | Yes | No | No | No | No |
| Pre-scan index | Yes | No | No | No | No |
| Provider-agnostic | Yes | N/A | N/A | N/A | N/A |
| PDF-only (no LLM needed) | Yes (prescan) | Yes | Yes | Yes | No |

The key difference: math-ocr extracts the **original embedded image** from the PDF, not a re-rendered screenshot. This preserves full resolution and quality.

## License

MIT
