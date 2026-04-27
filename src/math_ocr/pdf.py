"""PDF and image processing for math OCR.

Core capabilities:
  - PDF → base64 page images (configurable DPI)
  - Pre-scan: build an index of all embedded images + vector paths in a PDF
  - Figure cropping: extract embedded originals via IoU matching (not screenshots!)
  - Vector path detection: find geometry/coordinate-system drawings
  - SVG generation alongside PNG exports
  - Image preprocessing (contrast, resize)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from math_ocr.config import OCRConfig

logger = logging.getLogger("math_ocr")

# ── Data Types ────────────────────────────────────────────────────────


@dataclass
class FigureRegion:
    """A detected figure region on a page."""
    page_index: int
    bbox: tuple[float, float, float, float]  # normalized 0-1: (x0, y0, x1, y1)
    width: int
    height: int
    source: str = "unknown"  # "embedded_bitmap", "vector_path", "llm"


@dataclass
class PageImage:
    """A rendered page image."""
    page_index: int
    b64: str
    width: int
    height: int


@dataclass
class CroppedFigure:
    """A cropped figure image."""
    figure: FigureRegion
    b64: str
    source: str  # "embedded" or "page_crop"
    width: int = 0
    height: int = 0


class ImageIndexEntry(TypedDict):
    """One entry in the pre-scan image index."""
    pdf_path: str
    page_index: int
    xref: int
    bbox: list[float]  # normalized 0-1
    width: int
    height: int
    image_type: str  # "bitmap" or "vector_path"


# ── Noise Filtering ──────────────────────────────────────────────────

# Only filter pixel area < 100 (real figures are 6000+ px).
MIN_NOISE_PIXEL_AREA = 100
# Vector paths: minimum path count and area ratio to count as a figure
MIN_PATH_COUNT = 10
MIN_PATH_AREA_RATIO = 0.02


def _is_logo(entry: dict) -> bool:
    """Detect known logo bitmaps: page=0 and at bottom of page (y0 > 0.9)."""
    if entry.get("page_index") != 0:
        return False
    bbox = entry.get("bbox") or []
    return len(bbox) == 4 and float(bbox[1]) > 0.9


# ── PDF → Base64 Images ──────────────────────────────────────────────


def pdf_to_images(
    pdf_path: Path,
    *,
    dpi: int | None = None,
    page_indices: list[int] | None = None,
    config: OCRConfig | None = None,
) -> list[PageImage]:
    """Render PDF pages to base64 PNG images."""
    import fitz

    if config is None:
        config = OCRConfig()
    dpi = dpi or config.dpi

    doc = fitz.open(str(pdf_path))
    try:
        indices = page_indices if page_indices is not None else list(range(len(doc)))
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        results = []
        for i in indices:
            page = doc[i]
            pix = page.get_pixmap(matrix=mat)
            b64 = base64.b64encode(pix.tobytes("png")).decode()
            results.append(PageImage(
                page_index=i,
                b64=b64,
                width=pix.width,
                height=pix.height,
            ))
        return results
    finally:
        doc.close()


def image_to_b64(image_path: Path) -> str:
    """Read an image file and return base64-encoded PNG."""
    from PIL import Image

    img = Image.open(image_path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Image Preprocessing ──────────────────────────────────────────────


def preprocess_image(
    b64: str,
    *,
    max_size: int = 2048,
    enhance_contrast: bool = False,
) -> str:
    """Optional preprocessing: resize, enhance contrast."""
    from PIL import Image, ImageEnhance

    img = Image.open(io.BytesIO(base64.b64decode(b64)))

    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    if enhance_contrast:
        img = ImageEnhance.Contrast(img).enhance(1.5)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Pre-Scan: Build Image Index ──────────────────────────────────────


def prescan_pdf(pdf_path: Path) -> list[ImageIndexEntry]:
    """Scan a PDF and build an index of all embedded images + vector paths.

    This is the foundation of precise figure extraction. It finds:
    1. Embedded bitmaps with normalized bounding boxes
    2. Pages with significant vector drawings (geometry, coordinate systems)

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF.

    Returns
    -------
    list[ImageIndexEntry]
    """
    import fitz

    results: list[ImageIndexEntry] = []
    pdf_str = str(pdf_path)

    doc = fitz.open(str(pdf_path))
    try:
        for pg_i in range(len(doc)):
            page = doc[pg_i]

            # 1. Scan embedded bitmaps
            for img in page.get_images(full=True):
                xref = img[0]
                bbox = _get_image_norm_bbox(page, xref)
                if bbox is None:
                    continue

                # Filter tiny images (icons, lines)
                w_norm = bbox[2] - bbox[0]
                h_norm = bbox[3] - bbox[1]
                if w_norm < 0.03 or h_norm < 0.03:
                    continue

                try:
                    img_data = doc.extract_image(xref)
                    results.append(ImageIndexEntry(
                        pdf_path=pdf_str,
                        page_index=pg_i,
                        xref=xref,
                        bbox=bbox,
                        width=img_data["width"],
                        height=img_data["height"],
                        image_type="bitmap",
                    ))
                except Exception:
                    pass

            # 2. Scan vector paths (drawings)
            has_paths, area_ratio = _page_has_figure_paths(page)
            if has_paths:
                path_bbox = _get_path_bbox(page)
                if path_bbox:
                    results.append(ImageIndexEntry(
                        pdf_path=pdf_str,
                        page_index=pg_i,
                        xref=-1,  # vector paths don't have xrefs
                        bbox=path_bbox,
                        width=int(page.rect.width),
                        height=int(page.rect.height),
                        image_type="vector_path",
                    ))
    finally:
        doc.close()

    return results


def prescan_pdfs(pdf_paths: list[Path], index_path: Path | None = None) -> list[ImageIndexEntry]:
    """Scan multiple PDFs and optionally save index to JSONL."""
    all_entries: list[ImageIndexEntry] = []
    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            logger.warning("[prescan] not found: %s", pdf_path)
            continue
        entries = prescan_pdf(pdf_path)
        all_entries.extend(entries)
        logger.info("[prescan] %s: %d items", pdf_path.name, len(entries))

    if index_path is not None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "w") as f:
            for entry in all_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("[prescan] saved %d entries to %s", len(all_entries), index_path)

    return all_entries


def load_image_index(index_path: Path) -> dict[str, list[ImageIndexEntry]]:
    """Load a pre-scan index, grouped by pdf_path."""
    if not index_path.exists():
        return {}
    index: dict[str, list[ImageIndexEntry]] = {}
    with open(index_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                key = entry.get("pdf_path", "")
                if key not in index:
                    index[key] = []
                index[key].append(entry)
            except Exception:
                continue
    return index


def _get_image_norm_bbox(page: Any, xref: int) -> list[float] | None:
    """Get normalized 0-1 bbox for an embedded image on a page."""
    import fitz

    ctm = page.transformation_matrix
    probe = page.get_pixmap(matrix=fitz.Matrix(1, 1))
    pw, ph = probe.width, probe.height

    infos = [i for i in page.get_image_info(hashes=False, xrefs=True) if i.get("xref") == xref]
    if not infos:
        return None

    # Pick the largest transform (image may be referenced multiple times)
    info = max(infos, key=lambda x: abs(x["transform"][0] * x["transform"][3]))
    xs, ys = _transform_to_page_coords(info["transform"], ctm)

    return [
        round(min(xs) / pw, 4),
        round(min(ys) / ph, 4),
        round(max(xs) / pw, 4),
        round(max(ys) / ph, 4),
    ]


def _transform_to_page_coords(
    transform: tuple[float, ...], ctm
) -> tuple[list[float], list[float]]:
    """Expand an image transform matrix to page coordinates.

    Takes a 6-element transform (a, b, c, d, e, f) and a page CTM,
    returns (xs, ys) for the four corners of the unit square.
    """
    a, b, c, d, e, f = transform
    xs, ys = [], []
    for ux, uy in [(0, 0), (1, 0), (1, 1), (0, 1)]:
        px = a * ux + c * uy + e
        py = b * ux + d * uy + f
        xs.append(ctm.a * px + ctm.c * py + ctm.e)
        ys.append(ctm.b * px + ctm.d * py + ctm.f)
    return xs, ys


def _paths_union_bbox(page) -> tuple[float, float, float, float] | None:
    """Compute the union bounding box of all drawing paths on a page."""
    paths = page.get_drawings()
    if len(paths) < MIN_PATH_COUNT:
        return None
    bboxes = [p["rect"] for p in paths if p.get("rect")]
    if not bboxes:
        return None
    return (
        min(b.x0 for b in bboxes),
        min(b.y0 for b in bboxes),
        max(b.x1 for b in bboxes),
        max(b.y1 for b in bboxes),
    )


def _page_has_figure_paths(page: Any) -> tuple[bool, float]:
    """Check if a page has significant vector drawings."""
    union = _paths_union_bbox(page)
    if union is None:
        return False, 0.0
    x0, y0, x1, y1 = union
    area_ratio = (x1 - x0) * (y1 - y0) / (page.rect.width * page.rect.height)
    return area_ratio >= MIN_PATH_AREA_RATIO, area_ratio


def _get_path_bbox(page: Any) -> list[float] | None:
    """Get the overall bounding box of all vector paths on a page, normalized."""
    union = _paths_union_bbox(page)
    if union is None:
        return None
    x0, y0, x1, y1 = union
    pw, ph = page.rect.width, page.rect.height
    if pw <= 0 or ph <= 0:
        return None
    return [
        max(0, round(x0 / pw, 4)),
        max(0, round(y0 / ph, 4)),
        min(1, round(x1 / pw, 4)),
        min(1, round(y1 / ph, 4)),
    ]


# ── Figure Cropping ──────────────────────────────────────────────────


def crop_figure(
    pdf_path: Path,
    figure: FigureRegion,
    *,
    config: OCRConfig | None = None,
) -> CroppedFigure | None:
    """Crop a figure region from a PDF page.

    Strategy:
      1. Try to extract the original embedded image via IoU matching (best quality)
      2. Fall back to rendering a page-region crop

    Parameters
    ----------
    pdf_path : Path
    figure : FigureRegion
    config : OCRConfig, optional

    Returns
    -------
    CroppedFigure or None
    """
    import fitz

    if config is None:
        config = OCRConfig()

    doc = fitz.open(str(pdf_path))
    try:
        if figure.page_index < 0 or figure.page_index >= len(doc):
            return None

        page = doc[figure.page_index]
        rect = page.rect
        w, h = int(rect.width), int(rect.height)

        x0, y0, x1, y1 = figure.bbox
        pad = config.figure_padding
        x0 = max(0.0, x0 - pad)
        y0 = max(0.0, y0 - pad)
        x1 = min(1.0, x1 + pad)
        y1 = min(1.0, y1 + pad)

        if x1 <= x0 or y1 <= y0:
            return None

        clip = fitz.Rect(x0 * w, y0 * h, x1 * w, y1 * h)

        # Try embedded image extraction first
        if config.extract_embedded_images:
            embedded = _extract_embedded_image(page, clip, iou_threshold=config.figure_iou_threshold)
            if embedded is not None:
                fw, fh = _png_dimensions(embedded)
                return CroppedFigure(
                    figure=figure,
                    b64=base64.b64encode(embedded).decode(),
                    source="embedded",
                    width=fw,
                    height=fh,
                )

        # Fallback: render page crop
        dpi = config.figure_crop_dpi
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=clip)
        png_bytes = pix.tobytes("png")
        return CroppedFigure(
            figure=figure,
            b64=base64.b64encode(png_bytes).decode(),
            source="page_crop",
            width=pix.width,
            height=pix.height,
        )
    except Exception as e:
        logger.warning("[crop] failed: %s", e)
        return None
    finally:
        doc.close()


def crop_all_figures(
    pdf_path: Path,
    figures: list[FigureRegion],
    output_dir: Path,
    *,
    config: OCRConfig | None = None,
    prefix: str = "",
) -> list[dict]:
    """Crop all figures from a PDF and save as PNG + SVG.

    Returns a list of dicts with file paths and metadata.
    """
    if config is None:
        config = OCRConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, figure in enumerate(figures):
        cropped = crop_figure(pdf_path, figure, config=config)
        if cropped is None:
            continue

        fname = f"{prefix}fig_{i:03d}_p{figure.page_index}"
        png_path = output_dir / f"{fname}.png"
        svg_path = output_dir / f"{fname}.svg"

        # Save PNG
        png_bytes = base64.b64decode(cropped.b64)
        png_path.write_bytes(png_bytes)

        # Generate SVG wrapper
        fw, fh = _png_dimensions(png_bytes)
        if fw > 0 and fh > 0:
            svg_content = (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'xmlns:xlink="http://www.w3.org/1999/xlink" '
                f'width="{fw}" height="{fh}" '
                f'viewBox="0 0 {fw} {fh}">'
                f'<image width="{fw}" height="{fh}" '
                f'xlink:href="data:image/png;base64,{cropped.b64}"/>'
                f'</svg>'
            )
            svg_path.write_text(svg_content)
        else:
            svg_path = None

        results.append({
            "index": i,
            "page_index": figure.page_index,
            "bbox": list(figure.bbox),
            "source": cropped.source,
            "width": cropped.width or fw,
            "height": cropped.height or fh,
            "png_path": str(png_path),
            "svg_path": str(svg_path) if svg_path else None,
            "figure_type": figure.source,
        })

    return results


# ── Embedded Image Extraction (IoU Matching) ─────────────────────────


def _extract_embedded_image(page: Any, clip_rect: Any, *, iou_threshold: float = 0.05) -> bytes | None:
    """Try to extract the original embedded image overlapping with clip_rect.

    Uses IoU (Intersection over Union) matching to find the best embedded
    image that corresponds to the requested region. This preserves the
    original image quality rather than re-rendering at lower resolution.
    """
    import fitz

    try:
        ctm = page.transformation_matrix
        all_infos = [i for i in page.get_image_info(hashes=False, xrefs=True) if i.get("xref")]
        if not all_infos:
            return None

        candidates: list[tuple[float, int]] = []
        for info in all_infos:
            xref = info["xref"]
            xs, ys = _transform_to_page_coords(info["transform"], ctm)
            img_rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
            inter = clip_rect & img_rect
            union = clip_rect | img_rect
            iou = _rect_area(inter) / _rect_area(union) if not union.is_empty else 0.0
            candidates.append((iou, xref))

        best: tuple[float, int] | None = None
        if len(candidates) == 1:
            # Single candidate: use a very lenient threshold
            if candidates[0][0] > iou_threshold * 0.2:
                best = candidates[0]
        elif candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            if candidates[0][0] > iou_threshold:
                best = candidates[0]

        if best is None:
            return None

        doc = page.parent
        if doc is None:
            return None
        img_data = doc.extract_image(best[1])
        ext = img_data["ext"]
        raw = img_data["image"]
        if ext == "png":
            return raw
        # Convert other formats to PNG
        if ext in ("jpeg", "jpg"):
            pix = fitz.Pixmap(io.BytesIO(raw))
        else:
            pix = fitz.Pixmap(doc, best[1])
        if pix.n > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        return pix.tobytes("png")
    except Exception as e:
        logger.warning("[extract_embedded] failed: %s", e)
        return None


# ── Utilities ─────────────────────────────────────────────────────────


def _rect_area(rect: Any) -> float:
    width = max(0.0, float(rect.x1) - float(rect.x0))
    height = max(0.0, float(rect.y1) - float(rect.y0))
    return width * height


def _png_dimensions(data: bytes) -> tuple[int, int]:
    """Read PNG dimensions from header bytes."""
    if len(data) < 24:
        return 0, 0
    try:
        w = struct.unpack(">I", data[16:20])[0]
        h = struct.unpack(">I", data[20:24])[0]
        return w, h
    except Exception:
        return 0, 0


def normalize_bbox(bbox: list | tuple, page_width: int, page_height: int) -> list[float]:
    """Convert pixel coordinates to normalized 0-1 coordinates."""
    vals = [float(v) for v in bbox]
    if len(vals) != 4 or max(vals) <= 1.0:
        return vals
    if page_width <= 0 or page_height <= 0:
        return vals
    return [
        max(0.0, min(1.0, vals[0] / page_width)),
        max(0.0, min(1.0, vals[1] / page_height)),
        max(0.0, min(1.0, vals[2] / page_width)),
        max(0.0, min(1.0, vals[3] / page_height)),
    ]


def scan_page_signals(
    pdf_path: Path,
    page_indices: list[int] | None = None,
) -> str:
    """Quick text scan of PDF pages to detect neutral content signals.

    Uses PyMuPDF's page.get_text() to extract text (no LLM needed, ~ms per page).
    Detects patterns like question numbers, mark annotations, language, and figure
    labels. Returns a neutral description string for prompt injection.
    """
    import fitz

    if not pdf_path.exists():
        return ""

    doc = fitz.open(str(pdf_path))
    try:
        indices = page_indices if page_indices is not None else list(range(len(doc)))
        all_text = ""
        for i in indices:
            if 0 <= i < len(doc):
                all_text += doc[i].get_text() + "\n"
    finally:
        doc.close()

    if not all_text.strip():
        return ""

    signals: list[str] = ["Detected page signals:"]

    _detect_language(all_text, signals)
    _detect_question_numbers(all_text, signals)
    _detect_marks(all_text, signals)
    _detect_figure_labels(all_text, signals)

    if len(signals) == 1:
        return ""

    return "\n".join(signals)


def _detect_language(text: str, signals: list[str]) -> None:
    cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', text))
    if cjk_chars > 5:
        if re.search(r'[\u4e00-\u9fff]', text):
            signals.append("- Text contains Chinese characters")
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            signals.append("- Text contains Japanese characters")
    elif len(re.findall(r'[a-zA-Z]', text)) > 50:
        signals.append("- Text is primarily Latin/English")


def _detect_question_numbers(text: str, signals: list[str]) -> None:
    patterns = []
    if re.search(r'\bQ\.?\s*\d+', text, re.IGNORECASE):
        patterns.append("Q1, Q2...")
    if re.search(r'\bQuestion\s+\d+', text, re.IGNORECASE):
        patterns.append("Question 1, Question 2...")
    if re.search(r'第\d+题', text):
        patterns.append("第1题, 第2题...")
    if _has_sub_part_labels(text):
        patterns.append("(a), (b), (c) sub-parts")
    if re.search(r'\b\d+\s*[\.。、]\s*\(', text):
        patterns.append("numbered sub-parts (1., 2., etc.)")
    if patterns:
        signals.append(f"- Numbered items detected: {', '.join(patterns)}")


def _has_sub_part_labels(text: str) -> bool:
    """Check if text contains at least two (a)/(b)-style sub-part labels."""
    matches = list(re.finditer(r'\(\s*[a-z]\s*\)', text))
    if len(matches) < 2:
        return False
    # Verify they are distinct labels (not the same one repeated)
    labels = {m.group() for m in matches}
    return len(labels) >= 2


def _detect_marks(text: str, signals: list[str]) -> None:
    patterns = []
    if re.search(r'\[\d+[\]\s]', text):
        patterns.append("[3], [5], etc.")
    if re.search(r'\(\d+\s*marks?\)', text, re.IGNORECASE):
        patterns.append("(5 marks), (3 marks)")
    if re.search(r'[\(（]\d+\s*分[\)）]', text):
        patterns.append("(5分), (3分)")
    if re.search(r'\bM[1-9]\b|\bA[1-9]\b|\bB[1-9]\b|\bE[1-9]\b', text):
        patterns.append("method/accuracy marks (M1, A1, etc.)")
    if patterns:
        signals.append(f"- Score/mark annotations detected: {', '.join(patterns)}")


def _detect_figure_labels(text: str, signals: list[str]) -> None:
    patterns = []
    if re.search(r'\bFigure\s+\d+', text, re.IGNORECASE):
        patterns.append("Figure 1, Figure 2...")
    if re.search(r'\bDiagram\b', text, re.IGNORECASE):
        patterns.append("Diagram")
    if re.search(r'图\s*\d+', text):
        patterns.append("图1, 图2...")
    if patterns:
        signals.append(f"- Figure/diagram labels detected: {', '.join(patterns)}")


def build_figure_hints(
    pdf_path: Path,
    page_indices: list[int],
    index_path: Path | None = None,
) -> str:
    """Build text hints from pre-scan index to inject into LLM prompts.

    Helps the LLM know where figures already exist in the PDF, so it can
    produce accurate bboxes without guessing.
    """
    if index_path is not None:
        index = load_image_index(index_path)
    else:
        entries = prescan_pdf(pdf_path)
        index = {str(pdf_path): entries}

    items = index.get(str(pdf_path), [])
    if not items:
        return ""

    page_to_local = {page_idx: local_idx for local_idx, page_idx in enumerate(page_indices)}
    page_set = set(page_indices)

    rows = [
        img for img in items
        if img.get("page_index") in page_set
        and not _is_logo(img)
        and int(img.get("width", 0)) * int(img.get("height", 0)) >= MIN_NOISE_PIXEL_AREA
    ]
    if not rows:
        return ""

    lines = [
        "Confirmed figure-region checklist for the provided batch page images.",
        "Every listed hint_id/bbox is a known true figure region from the PDF image index.",
        "Each hint uses hint_id and batch_local_page_index, matching the page image order starting at 0.",
        "You must explicitly inspect every hinted bbox on the matching page image.",
        "If a real extracted row is tied to that confirmed figure/graph, set has_figure=true.",
        "Do not create rows from hints alone.",
    ]

    hint_id = 0
    for img in rows:
        bbox = img.get("bbox") or []
        if len(bbox) != 4:
            continue
        w = img.get("width", "?")
        h = img.get("height", "?")
        try:
            if int(w) * int(h) < MIN_NOISE_PIXEL_AREA:
                continue
        except (ValueError, TypeError):
            pass
        local_page_index = page_to_local.get(img["page_index"])
        if local_page_index is None:
            continue
        lines.append(
            f"- hint_id={hint_id} batch_local_page_index={local_page_index} "
            f"source_pdf_page_index={img['page_index']} "
            f"bbox=[{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}] "
            f"size={w}x{h} type={img.get('image_type','?')}"
        )
        hint_id += 1

    return "\n".join(lines)
