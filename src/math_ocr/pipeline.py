"""Main MathOCR pipeline: image in, structured math out.

Usage:
    from math_ocr import MathOCR, OCRConfig

    ocr = MathOCR()                          # uses env vars for API key
    results = ocr.extract_pdf("exam.pdf")    # returns list of dicts
    results = ocr.extract_image("photo.png") # single image
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from openai import OpenAI

from math_ocr.client import LlmResponse, build_client, complete
from math_ocr.config import OCRConfig
from math_ocr.parser import normalize_items, parse_json_response
from math_ocr.pdf import (
    FigureRegion,
    PageImage,
    build_figure_hints,
    crop_all_figures,
    crop_figure,
    image_to_b64,
    pdf_to_images,
    preprocess_image,
    scan_page_signals,
)
from math_ocr.prompts import math_extraction_prompt, user_extraction_prompt

logger = logging.getLogger("math_ocr")


class MathOCR:
    """High-level API for math OCR.

    Parameters
    ----------
    config : OCRConfig, optional
        Configuration. Uses env vars for API key/base_url if not provided.
    client : OpenAI, optional
        Pre-built OpenAI client. Built from config if not provided.
    """

    def __init__(
        self,
        config: OCRConfig | None = None,
        client: OpenAI | None = None,
    ):
        self.config = config or OCRConfig()
        self.client = client or build_client(self.config)

    # ── Public API ───────────────────────────────────────────────────

    def extract_pdf(
        self,
        pdf_path: str | Path,
        *,
        page_indices: list[int] | None = None,
        content_only: bool = False,
        detect_figures: bool = True,
        preprocess: bool = False,
        text_fields: list[str] | None = None,
        index_path: Path | None = None,
        figures_dir: Path | None = None,
        doc_type: str | None = None,
        lang: str | None = None,
        custom_rules: list[str] | None = None,
        output_format: str = "json_array",
        prompt_overrides: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract structured math content from a PDF.

        Parameters
        ----------
        pdf_path : str or Path
            Path to the PDF file.
        page_indices : list[int], optional
            Specific pages (0-based) to extract. None = all pages.
        content_only : bool
            If True, skip cover pages, instruction pages, and blank pages.
            Only extract actual content (questions, answers, formulas, figures).
        detect_figures : bool
            Whether to detect and crop figures.
        preprocess : bool
            Apply image preprocessing (contrast, resize).
        text_fields : list[str], optional
            Fields to apply LaTeX normalization.
        index_path : Path, optional
            Pre-scan JSONL index for figure hints.
        figures_dir : Path, optional
            If set, save cropped figure PNG+SVG files to this directory.
        doc_type : str, optional
            Document type hint: "exam_qp", "exam_ms", "textbook".
        lang : str, optional
            Content language hint (e.g. "en", "zh").
        custom_rules : list[str], optional
            Additional rules appended to the prompt.
        output_format : str
            "json_array" or "markdown".
        prompt_overrides : dict[str, str], optional
            Replace entire prompt sections by name.

        Returns
        -------
        list[dict]
            Extracted items with normalized LaTeX.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Render pages to images
        pages = pdf_to_images(pdf_path, page_indices=page_indices, config=self.config)
        if not pages:
            return []

        if preprocess:
            pages = [
                PageImage(
                    page_index=p.page_index,
                    b64=preprocess_image(p.b64),
                    width=p.width,
                    height=p.height,
                )
                for p in pages
            ]

        return self._extract_from_pages(
            pages=pages,
            pdf_path=pdf_path,
            detect_figures=detect_figures,
            text_fields=text_fields,
            index_path=index_path,
            figures_dir=figures_dir,
            doc_type=doc_type,
            lang=lang,
            content_only=content_only,
            custom_rules=custom_rules,
            output_format=output_format,
            prompt_overrides=prompt_overrides,
        )

    def extract_image(
        self,
        image_path: str | Path,
        *,
        detect_figures: bool = False,
        preprocess: bool = False,
        text_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract structured math content from an image.

        Parameters
        ----------
        image_path : str or Path
            Path to an image file (PNG, JPEG, etc.).
        detect_figures : bool
            Whether to detect figures in the image.
        preprocess : bool
            Apply image preprocessing.
        text_fields : list[str], optional
            Fields to apply LaTeX normalization.

        Returns
        -------
        list[dict]
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        b64 = image_to_b64(image_path)
        if preprocess:
            b64 = preprocess_image(b64)

        pages = [PageImage(page_index=0, b64=b64, width=0, height=0)]
        return self._extract_from_pages(
            pages=pages,
            pdf_path=None,
            detect_figures=detect_figures,
            text_fields=text_fields,
        )

    def extract_b64(
        self,
        images: list[str],
        *,
        detect_figures: bool = False,
        text_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract from pre-encoded base64 images.

        Parameters
        ----------
        images : list[str]
            Base64-encoded PNG images.
        detect_figures : bool
            Whether to include figure detection rules.
        text_fields : list[str], optional
            Fields to normalize.

        Returns
        -------
        list[dict]
        """
        pages = [
            PageImage(page_index=i, b64=b64, width=0, height=0)
            for i, b64 in enumerate(images)
        ]
        return self._extract_from_pages(
            pages=pages,
            pdf_path=None,
            detect_figures=detect_figures,
            text_fields=text_fields,
        )

    # ── Internal ─────────────────────────────────────────────────────

    def _extract_from_pages(
        self,
        pages: list[PageImage],
        pdf_path: Path | None,
        detect_figures: bool,
        text_fields: list[str] | None,
        index_path: Path | None = None,
        figures_dir: Path | None = None,
        doc_type: str | None = None,
        lang: str | None = None,
        content_only: bool = False,
        custom_rules: list[str] | None = None,
        output_format: str = "json_array",
        prompt_overrides: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Core extraction: pages → LLM call → parse → normalize."""
        # Auto-scan page signals from PDF
        page_signals = ""
        if pdf_path:
            page_signals = scan_page_signals(
                pdf_path,
                page_indices=[p.page_index for p in pages],
            )
            if lang:
                page_signals += f"\n- Language hint: {lang}"

        system_prompt = math_extraction_prompt(
            detect_figures=detect_figures,
            max_items_per_page=self.config.max_items_per_page,
            output_format=output_format,
            doc_type=doc_type,
            page_signals=page_signals,
            content_only=content_only,
            custom_rules=custom_rules,
            prompt_overrides=prompt_overrides,
        )

        # Build figure hints from pre-scan index
        hint_text = ""
        if detect_figures and pdf_path and index_path:
            page_indices = [p.page_index for p in pages]
            hint_text = build_figure_hints(pdf_path, page_indices, index_path)

        user_prompt = user_extraction_prompt(
            page_count=len(pages),
            hint_text=hint_text,
        )

        images = [p.b64 for p in pages]

        result: LlmResponse = complete(
            self.client,
            images=images,
            system_prompt=system_prompt,
            user_text=user_prompt,
            config=self.config,
        )

        if result.failed:
            logger.error("[extract] API request failed after retries")
            raise RuntimeError("API request failed after all retries")

        items, error = parse_json_response(result.text)
        if error:
            logger.warning("[extract] JSON parse error: %s", error)
            if not items:
                raise RuntimeError(f"Failed to parse LLM response: {error}")

        # Tag with page indices if not present
        for item in items:
            if "page_index" not in item:
                item["page_index"] = 0

        # Normalize LaTeX in text fields
        items = normalize_items(items, text_fields=text_fields)

        # Crop figures if we have a PDF source
        if detect_figures and pdf_path:
            items = self._crop_figures(items, pdf_path, figures_dir=figures_dir)

        return items

    def _crop_figures(
        self,
        items: list[dict[str, Any]],
        pdf_path: Path,
        *,
        figures_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Crop detected figures from the source PDF.

        If figures_dir is provided, also save PNG+SVG files to disk.
        """
        # Collect figure regions for batch processing
        figures = []
        figure_indices = []
        for i, item in enumerate(items):
            if not item.get("has_figure") or not item.get("figure_bbox"):
                continue
            bbox = item["figure_bbox"]
            if len(bbox) != 4:
                continue

            page_idx = item.get("page_index", 0)
            figure = FigureRegion(
                page_index=int(page_idx),
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                width=0,
                height=0,
            )
            figures.append(figure)
            figure_indices.append(i)

        if not figures:
            return items

        # If figures_dir specified, use batch crop with file output
        if figures_dir:
            crop_results = crop_all_figures(
                pdf_path, figures, figures_dir, config=self.config,
            )
            # Map results back to items
            for crop_info in crop_results:
                idx = crop_info["index"]
                if idx < len(figure_indices):
                    item_idx = figure_indices[idx]
                    items[item_idx]["figure_b64"] = None  # saved to file instead
                    items[item_idx]["figure_source"] = crop_info["source"]
                    items[item_idx]["figure_png"] = crop_info["png_path"]
                    items[item_idx]["figure_svg"] = crop_info["svg_path"]
                    items[item_idx]["figure_width"] = crop_info["width"]
                    items[item_idx]["figure_height"] = crop_info["height"]
            return items

        # Inline crop (b64 only)
        for i, figure in zip(figure_indices, figures):
            cropped = crop_figure(pdf_path, figure, config=self.config)
            if cropped:
                items[i]["figure_b64"] = cropped.b64
                items[i]["figure_source"] = cropped.source
            else:
                items[i]["has_figure"] = False

        return items

    def extract_pdf_to_file(
        self,
        pdf_path: str | Path,
        output_path: str | Path,
        **kwargs,
    ) -> Path:
        """Extract and save results to a JSON file.

        Parameters
        ----------
        pdf_path : str or Path
            Input PDF.
        output_path : str or Path
            Output JSON file path.
        **kwargs
            Passed to extract_pdf.

        Returns
        -------
        Path
            Path to the output file.
        """
        items = self.extract_pdf(pdf_path, **kwargs)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(items, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path
