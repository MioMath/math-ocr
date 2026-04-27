"""Tests for OCR config."""

import os

from math_ocr.config import OCRConfig


class TestOCRConfig:
    def test_defaults(self):
        config = OCRConfig()
        assert config.dpi == 150
        assert config.max_retries == 2
        assert config.model == "gemini-2.5-flash"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MATH_OCR_API_KEY", "test-key-123")
        monkeypatch.setenv("MATH_OCR_BASE_URL", "https://custom.api.com/v1")
        config = OCRConfig()
        assert config.api_key == "test-key-123"
        assert config.base_url == "https://custom.api.com/v1"

    def test_openai_fallback(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.delenv("MATH_OCR_API_KEY", raising=False)
        config = OCRConfig()
        assert config.api_key == "openai-key"

    def test_custom_values(self):
        config = OCRConfig(
            api_key="my-key",
            model="gpt-4o",
            dpi=300,
        )
        assert config.api_key == "my-key"
        assert config.model == "gpt-4o"
        assert config.dpi == 300
