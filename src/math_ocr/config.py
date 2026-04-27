"""Configuration for math-ocr.

Four layers of config (later wins):
  1. Built-in defaults (dataclass defaults)
  2. Config file: ~/.math-ocr.json
  3. Environment variables: MATH_OCR_API_KEY, MATH_OCR_BASE_URL, MATH_OCR_MODEL
  4. Explicit constructor args or CLI flags

Constructor args that differ from defaults are preserved (not overwritten by file or env).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_FILE = Path.home() / ".math-ocr.json"


@dataclass
class OCRConfig:
    """All tuneable knobs for the OCR pipeline.

    Every field has a sensible default. Override via config file, env vars,
    constructor args, or CLI flags (later overrides earlier).
    """

    # ── LLM Provider ──────────────────────────────────────────────
    api_key: str = ""
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    model: str = "gemini-2.5-flash-preview-05-20"

    # ── Streaming / Budget ────────────────────────────────────────
    max_tokens: int = 16384
    max_output_chars: int = 140_000
    hard_timeout_s: int = 1800
    request_timeout_s: int = 1815
    max_retries: int = 2

    # ── PDF Rendering ─────────────────────────────────────────────
    dpi: int = 150
    figure_crop_dpi: int = 220
    figure_padding: float = 0.01  # normalized padding around detected figures

    # ── Output ────────────────────────────────────────────────────
    max_items_per_page: int = 25

    # ── Figure Detection ──────────────────────────────────────────
    min_figure_pixel_area: int = 100  # ignore noise below this threshold
    figure_iou_threshold: float = 0.3
    extract_embedded_images: bool = True  # prefer extracting embedded PNGs over page crops

    def __post_init__(self):
        # Snapshot which fields the caller explicitly set (non-default).
        # These must survive file/env overrides.
        defaults = {f.name: f.default for f in self.__dataclass_fields__.values()}
        explicit = {k for k, v in defaults.items() if getattr(self, k) != v}

        # Layer 2: config file — only touch fields still at default
        file_cfg = _load_config_file()
        for key, val in file_cfg.items():
            if key in explicit:
                continue
            if hasattr(self, key):
                setattr(self, key, val)

        # Layer 3: env vars (only if field is still default/empty)
        if "api_key" not in explicit and not self.api_key:
            self.api_key = os.getenv("MATH_OCR_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if "base_url" not in explicit and (not self.base_url or self.base_url == defaults["base_url"]):
            env_url = os.getenv("MATH_OCR_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))
            if env_url:
                self.base_url = env_url
        if "model" not in explicit:
            env_model = os.getenv("MATH_OCR_MODEL")
            if env_model and self.model == defaults["model"]:
                self.model = env_model


def _load_config_file() -> dict:
    """Read ~/.math-ocr.json if it exists."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_config_file(updates: dict) -> Path:
    """Merge updates into ~/.math-ocr.json and return the path."""
    current = _load_config_file()
    current.update({k: v for k, v in updates.items() if v is not None})
    CONFIG_FILE.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(CONFIG_FILE, 0o600)
    return CONFIG_FILE
