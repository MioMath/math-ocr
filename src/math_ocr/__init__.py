"""math-ocr: Turn math PDFs and images into structured, LaTeX-ready data."""

__version__ = "0.1.0"

from math_ocr.pipeline import MathOCR
from math_ocr.config import OCRConfig

__all__ = ["MathOCR", "OCRConfig"]
