"""Provider-agnostic LLM client for math OCR.

Wraps the OpenAI SDK (which works with any OpenAI-compatible endpoint,
including Gemini, Azure, local models via Ollama, etc.) with streaming,
timeout, retry, and truncation detection tailored for long OCR outputs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from math_ocr.config import OCRConfig

logger = logging.getLogger("math_ocr")


@dataclass
class LlmResponse:
    """Result of a single LLM call."""
    text: str
    truncated: bool
    failed: bool
    elapsed_s: float


def build_client(config: OCRConfig) -> OpenAI:
    """Create an OpenAI client from OCRConfig."""
    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )


def complete(
    client: OpenAI,
    images: list[str],
    system_prompt: str,
    user_text: str,
    *,
    extra_images: list[str] | None = None,
    temperature: float = 0.0,
    config: OCRConfig | None = None,
    **overrides: Any,
) -> LlmResponse:
    """Send images + prompt to the vision LLM and collect the response.

    Parameters
    ----------
    client : OpenAI
        Pre-built OpenAI client.
    images : list[str]
        Base64-encoded page images.
    system_prompt : str
        System-level instruction.
    user_text : str
        User-level extraction prompt.
    extra_images : list[str], optional
        Additional base64 images appended after ``images``.
    temperature : float
        Sampling temperature.
    config : OCRConfig, optional
        Fallback for any override not provided.
    **overrides
        Named overrides that take priority over config:
        ``model``, ``max_tokens``, ``max_output_chars``,
        ``hard_timeout_s``, ``request_timeout_s``, ``max_retries``.
    """
    if config is None:
        config = OCRConfig()
    model = overrides.get("model") if overrides.get("model") is not None else config.model
    max_tokens = overrides.get("max_tokens") if overrides.get("max_tokens") is not None else config.max_tokens
    max_output_chars = overrides.get("max_output_chars") if overrides.get("max_output_chars") is not None else config.max_output_chars
    hard_timeout_s = overrides.get("hard_timeout_s") if overrides.get("hard_timeout_s") is not None else config.hard_timeout_s
    request_timeout_s = overrides.get("request_timeout_s") if overrides.get("request_timeout_s") is not None else config.request_timeout_s
    max_retries = overrides.get("max_retries") if overrides.get("max_retries") is not None else config.max_retries

    # Build multimodal content
    all_b64 = list(images) + (extra_images or [])
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        for b64 in all_b64
    ]
    content.append({"type": "text", "text": user_text})

    for attempt in range(1, max_retries + 1):
        started = time.time()
        stream = None
        try:
            logger.info(
                "[api] model=%s images=%d attempt=%d/%d budget=%d_tok timeout=%ds",
                model, len(images), attempt, max_retries, max_tokens, hard_timeout_s,
            )
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=request_timeout_s,
                stream=True,
            )

            chunks: list[str] = []
            char_count = 0
            finish_reason = None
            client_cut = False

            for event in stream:
                if time.time() - started > hard_timeout_s:
                    client_cut = True
                    logger.warning("[api] hard timeout %ds, forcing truncation", hard_timeout_s)
                    break
                if not event.choices:
                    continue
                choice = event.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                    break
                delta = choice.delta
                if delta and delta.content:
                    chunks.append(delta.content)
                    char_count += len(delta.content)
                    if char_count > max_output_chars:
                        client_cut = True
                        logger.warning("[api] output exceeded %d chars, forcing truncation", max_output_chars)
                        break

            raw_text = "".join(chunks)
            elapsed = time.time() - started
            truncated = client_cut or finish_reason in {"length", "max_tokens", "MAX_TOKENS"}

            logger.info(
                "[api] %s %.1fs chars=%d finish=%s",
                "truncated" if truncated else "ok", elapsed, char_count, finish_reason,
            )
            return LlmResponse(
                text=raw_text,
                truncated=truncated,
                failed=False,
                elapsed_s=elapsed,
            )

        except Exception as exc:
            elapsed = time.time() - started
            logger.warning("[api] attempt %d/%d failed (%.1fs): %s", attempt, max_retries, elapsed, exc)
            if attempt == max_retries:
                return LlmResponse(text="", truncated=False, failed=True, elapsed_s=elapsed)
            time.sleep(2 * attempt)
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    # Loop did not execute (max_retries=0)
    return LlmResponse(text="", truncated=False, failed=True, elapsed_s=0.0)
