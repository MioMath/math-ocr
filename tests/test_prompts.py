"""Tests for prompt templates."""

from math_ocr.prompts import math_extraction_prompt, user_extraction_prompt


class TestMathExtractionPrompt:
    def test_basic_prompt(self):
        prompt = math_extraction_prompt()
        assert "LATEX" in prompt
        assert "MANDATORY" in prompt
        assert "BUDGET" in prompt

    def test_with_figures(self):
        prompt = math_extraction_prompt(detect_figures=True)
        assert "figure" in prompt.lower()

    def test_without_figures(self):
        prompt = math_extraction_prompt(detect_figures=False)
        assert "FIGURE RULES" not in prompt

    def test_custom_max_items(self):
        prompt = math_extraction_prompt(max_items_per_page=10)
        assert "10" in prompt

    def test_custom_rules(self):
        prompt = math_extraction_prompt(custom_rules=["My custom rule"])
        assert "My custom rule" in prompt

    def test_json_format(self):
        prompt = math_extraction_prompt(output_format="json_array")
        assert "JSON" in prompt

    def test_markdown_format(self):
        prompt = math_extraction_prompt(output_format="markdown")
        assert "markdown" in prompt.lower()

    def test_no_a_level_content(self):
        """Prompt should not contain A-Level/exam-specific assumptions."""
        prompt = math_extraction_prompt()
        assert "A-Level" not in prompt
        assert "CIE" not in prompt
        assert "9709" not in prompt

    def test_doc_type_exam_qp(self):
        prompt = math_extraction_prompt(doc_type="exam_qp")
        assert "question" in prompt.lower()
        assert "question paper" in prompt.lower()

    def test_doc_type_exam_ms(self):
        prompt = math_extraction_prompt(doc_type="exam_ms")
        assert "marking scheme" in prompt.lower()
        assert "mark allocations" in prompt.lower()

    def test_doc_type_textbook(self):
        prompt = math_extraction_prompt(doc_type="textbook")
        assert "textbook" in prompt.lower()
        assert "structure" in prompt.lower()

    def test_doc_type_generic_is_empty(self):
        """Generic doc_type should not add a section."""
        prompt_generic = math_extraction_prompt(doc_type="generic")
        prompt_none = math_extraction_prompt(doc_type=None)
        assert prompt_generic == prompt_none

    def test_page_signals_injected(self):
        signals = "Detected page signals:\n- Text is primarily Latin/English\n- Numbered items detected: Q1, Q2..."
        prompt = math_extraction_prompt(page_signals=signals)
        assert "Page Analysis" in prompt
        assert "Latin/English" in prompt

    def test_prompt_overrides_role(self):
        custom_role = "You are a Chinese math exam extractor."
        prompt = math_extraction_prompt(prompt_overrides={"role": custom_role})
        assert custom_role in prompt
        # Should NOT contain the default role text
        assert "precision math content extraction engine" not in prompt

    def test_prompt_overrides_json_format(self):
        custom_format = 'Output: {"question": "...", "answer": "..."}'
        prompt = math_extraction_prompt(
            output_format="json_array",
            prompt_overrides={"json_format": custom_format},
        )
        assert custom_format in prompt

    def test_multiple_overrides(self):
        overrides = {
            "role": "Custom role.",
            "latex_rules": "Custom latex rules.",
        }
        prompt = math_extraction_prompt(prompt_overrides=overrides)
        assert "Custom role." in prompt
        assert "Custom latex rules." in prompt

    def test_doc_type_with_custom_rules(self):
        """Both doc_type and custom_rules should work together."""
        prompt = math_extraction_prompt(
            doc_type="exam_qp",
            custom_rules=["Include difficulty level."],
        )
        assert "question paper" in prompt.lower()
        assert "Include difficulty level." in prompt


class TestUserExtractionPrompt:
    def test_single_page(self):
        prompt = user_extraction_prompt(page_count=1)
        assert "1" in prompt

    def test_multi_page(self):
        prompt = user_extraction_prompt(page_count=5)
        assert "5" in prompt

    def test_with_hints(self):
        prompt = user_extraction_prompt(hint_text="Figure at [0.1, 0.2, 0.5, 0.8]")
        assert "0.1" in prompt
