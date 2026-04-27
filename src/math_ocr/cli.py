"""CLI interface for math-ocr.

Usage:
    math-ocr extract input.pdf
    math-ocr extract photo.png -o result.json
    math-ocr extract input.pdf --pages 0,1,2 --no-figures
    math-ocr prescan input.pdf -o index.jsonl
    math-ocr prescan dir/*.pdf -o index.jsonl
    math-ocr config
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import click

from math_ocr import __version__


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(name)s %(levelname)s: %(message)s" if verbose else "%(message)s"
    logging.basicConfig(level=level, format=fmt)


@click.group()
@click.version_option(__version__, prog_name="math-ocr")
def main():
    """math-ocr: Turn math PDFs and images into structured, LaTeX-ready data."""
    pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output JSON file path.")
@click.option(
    "--pages", type=str, default=None,
    help="Page indices to extract (comma-separated, 0-based). Default: all pages.",
)
@click.option(
    "--no-figures", is_flag=True, default=False,
    help="Skip figure detection and cropping.",
)
@click.option(
    "--preprocess", is_flag=True, default=False,
    help="Apply image preprocessing (contrast enhancement, resize).",
)
@click.option(
    "--model", type=str, default=None,
    help="LLM model to use (default from config).",
)
@click.option(
    "--dpi", type=int, default=None,
    help="PDF render DPI (default: 150).",
)
@click.option(
    "--api-key", type=str, default=None,
    help="API key (or set MATH_OCR_API_KEY / OPENAI_API_KEY env var).",
)
@click.option(
    "--base-url", type=str, default=None,
    help="API base URL (or set MATH_OCR_BASE_URL / OPENAI_BASE_URL env var).",
)
@click.option(
    "--max-tokens", type=int, default=None,
    help="Max output tokens from LLM.",
)
@click.option(
    "--timeout", type=int, default=None,
    help="Hard timeout in seconds.",
)
@click.option(
    "--index", type=click.Path(exists=True), default=None,
    help="Pre-scan JSONL index file for figure hints.",
)
@click.option(
    "--figures-dir", type=click.Path(), default=None,
    help="Directory to save cropped figure PNG+SVG files.",
)
@click.option(
    "--doc-type", type=click.Choice(["generic", "exam_qp", "exam_ms", "textbook"]),
    default="generic",
    help="Document type hint for better extraction.",
)
@click.option(
    "--lang", type=str, default=None,
    help="Content language hint (e.g. en, zh, ja).",
)
@click.option(
    "--custom-rules", type=str, default=None,
    help="Extra prompt rules. Comma-separated, or @file.txt to load from file.",
)
@click.option(
    "--content-only", is_flag=True, default=False,
    help="Only extract content pages. Skip cover, instructions, and blank pages.",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose logging.")
def extract(
    input_path: str,
    output: str | None,
    pages: str | None,
    no_figures: bool,
    preprocess: bool,
    model: str | None,
    dpi: int | None,
    api_key: str | None,
    base_url: str | None,
    max_tokens: int | None,
    timeout: int | None,
    index: str | None,
    figures_dir: str | None,
    doc_type: str,
    lang: str | None,
    custom_rules: str | None,
    content_only: bool,
    verbose: bool,
):
    """Extract math content from a PDF or image file."""
    from math_ocr.config import OCRConfig
    from math_ocr.pipeline import MathOCR

    _setup_logging(verbose)

    # Build config with CLI overrides
    config = OCRConfig()
    if api_key:
        config.api_key = api_key
    if base_url:
        config.base_url = base_url
    if model:
        config.model = model
    if dpi:
        config.dpi = dpi
    if max_tokens:
        config.max_tokens = max_tokens
    if timeout:
        config.hard_timeout_s = timeout

    if not config.api_key:
        click.echo("Error: API key required. Set MATH_OCR_API_KEY or use --api-key.", err=True)
        sys.exit(1)

    # Parse page indices
    page_indices = None
    if pages:
        page_indices = [int(p.strip()) for p in pages.split(",")]

    input_p = Path(input_path)
    click.echo(f"Extracting: {input_p.name}")
    click.echo(f"Model: {config.model}")
    click.echo(f"Figures: {'disabled' if no_figures else 'enabled'}")
    if doc_type != "generic":
        click.echo(f"Doc type: {doc_type}")
    if lang:
        click.echo(f"Language: {lang}")
    if index:
        click.echo(f"Index: {index}")

    # Resolve index and figures_dir
    index_path = Path(index) if index else None
    fig_dir = Path(figures_dir) if figures_dir else None

    # Parse custom rules
    rules_list: list[str] | None = None
    if custom_rules:
        if custom_rules.startswith("@"):
            rules_file = Path(custom_rules[1:])
            if not rules_file.exists():
                click.echo(f"Error: rules file not found: {rules_file}", err=True)
                sys.exit(1)
            rules_list = [line.strip() for line in rules_file.read_text().splitlines() if line.strip()]
        else:
            rules_list = [r.strip() for r in custom_rules.split(",") if r.strip()]

    # Resolve doc_type
    effective_doc_type = doc_type if doc_type != "generic" else None
    effective_lang = lang if lang else None

    start = time.time()
    ocr = MathOCR(config=config)

    suffix = input_p.suffix.lower()
    if suffix == ".pdf":
        results = ocr.extract_pdf(
            input_p,
            page_indices=page_indices,
            detect_figures=not no_figures,
            preprocess=preprocess,
            index_path=index_path,
            figures_dir=fig_dir,
            doc_type=effective_doc_type,
            lang=effective_lang,
            content_only=content_only,
            custom_rules=rules_list,
        )
    elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
        results = ocr.extract_image(
            input_p,
            detect_figures=not no_figures,
            preprocess=preprocess,
        )
    else:
        click.echo(f"Error: unsupported file type: {suffix}", err=True)
        sys.exit(1)

    elapsed = time.time() - start
    click.echo(f"Extracted {len(results)} items in {elapsed:.1f}s")

    # Output
    output_json = json.dumps(results, indent=2, ensure_ascii=False)
    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(output_json, encoding="utf-8")
        click.echo(f"Saved to: {out_p}")
    else:
        click.echo(output_json)


@main.command()
@click.argument("input_paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output JSONL index file path.")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose logging.")
def prescan(
    input_paths: tuple[str, ...],
    output: str | None,
    verbose: bool,
):
    """Pre-scan PDFs to build an image/figure index.

    Scans all embedded bitmaps and vector paths in the given PDFs.
    Outputs a JSONL index with normalized bounding boxes for each figure.

    This index can be used to speed up subsequent extractions with --index.
    """
    from math_ocr.pdf import prescan_pdfs

    _setup_logging(verbose)

    pdf_paths = [Path(p) for p in input_paths]
    index_path = Path(output) if output else None

    click.echo(f"Scanning {len(pdf_paths)} file(s)...")
    start = time.time()

    entries = prescan_pdfs(pdf_paths, index_path=index_path)

    elapsed = time.time() - start
    click.echo(f"Found {len(entries)} figure(s) in {elapsed:.1f}s")

    if not output:
        for entry in entries:
            click.echo(json.dumps(entry, ensure_ascii=False))
    else:
        click.echo(f"Saved to: {output}")


@main.command()
@click.option("--set-api-key", type=str, default=None, help="Save API key to config file.")
@click.option("--set-base-url", type=str, default=None, help="Save API base URL to config file.")
@click.option("--set-model", type=str, default=None, help="Save default model to config file.")
def config(set_api_key: str | None, set_base_url: str | None, set_model: str | None):
    """Show or save configuration.

    Run without options to display current config.
    Use --set-api-key, --set-base-url, --set-model to persist values to ~/.math-ocr.json.

    Config priority (later wins): defaults < ~/.math-ocr.json < env vars < CLI flags.
    """
    from math_ocr.config import CONFIG_FILE, OCRConfig, save_config_file

    # Save mode
    if any([set_api_key, set_base_url, set_model]):
        updates: dict = {}
        if set_api_key:
            updates["api_key"] = set_api_key
        if set_base_url:
            updates["base_url"] = set_base_url
        if set_model:
            updates["model"] = set_model
        path = save_config_file(updates)
        click.echo(f"Saved to {path}")
        return

    # Display mode
    c = OCRConfig()
    masked_key = "..." + c.api_key[-4:] if len(c.api_key) > 8 else "(set)" if c.api_key else "(not set)"
    click.echo(f"Config file: {CONFIG_FILE}")
    click.echo(f"API Key:     {masked_key}")
    click.echo(f"Base URL:    {c.base_url}")
    click.echo(f"Model:       {c.model}")
    click.echo(f"DPI:         {c.dpi}")
    click.echo(f"Timeout:     {c.hard_timeout_s}s")


if __name__ == "__main__":
    main()
