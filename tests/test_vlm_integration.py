"""Integration tests — real API calls to each VLM provider.

Each test class is gated on its own fixture (skipped when API key absent).
These tests validate:
  - call() returns non-empty text
  - detect() returns a well-formed DetectResult
  - Confidence values are in [0, 1]
  - bbox (if present) has exactly 4 non-negative integers

Run only Claude tests:
    pytest tests/test_vlm_integration.py::TestClaudeIntegration -v

Run all integration tests (requires all keys):
    pytest tests/test_vlm_integration.py -v
"""

import pytest
from llm.vlm_client import DetectResult


# ================================================================
# Shared assertion helpers
# ================================================================

def assert_detect_result(result: DetectResult):
    assert isinstance(result, DetectResult)
    assert isinstance(result.has_waldo, bool)
    assert 0.0 <= result.confidence <= 1.0
    if result.bbox is not None:
        assert len(result.bbox) == 4
        assert all(isinstance(v, (int, float)) and v >= 0 for v in result.bbox)
    assert isinstance(result.raw_response, str) and len(result.raw_response) > 0


# ================================================================
# Claude
# ================================================================

class TestClaudeIntegration:
    """Requires ANTHROPIC_API_KEY."""

    def test_call_returns_text(self, claude_client, test_image):
        response = claude_client.call(test_image, "What do you see in this image? Reply in one sentence.")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_detect_result_type(self, claude_client, test_image):
        result = claude_client.detect(test_image)
        assert_detect_result(result)

    def test_detect_confidence_range(self, claude_client, test_image):
        result = claude_client.detect(test_image)
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_custom_max_tokens(self, claude_client, test_image):
        response = claude_client.call(test_image, "Reply with the single word: OK", max_tokens=16)
        assert isinstance(response, str)
        assert len(response) > 0


# ================================================================
# GPT-4o
# ================================================================

class TestGPT4oIntegration:
    """Requires OPENAI_API_KEY."""

    def test_call_returns_text(self, gpt4o_client, test_image):
        response = gpt4o_client.call(test_image, "What do you see in this image? Reply in one sentence.")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_detect_result_type(self, gpt4o_client, test_image):
        result = gpt4o_client.detect(test_image)
        assert_detect_result(result)

    def test_detect_confidence_range(self, gpt4o_client, test_image):
        result = gpt4o_client.detect(test_image)
        assert 0.0 <= result.confidence <= 1.0


# ================================================================
# Gemini
# ================================================================

class TestGeminiIntegration:
    """Requires GOOGLE_API_KEY and google-generativeai package."""

    def test_call_returns_text(self, gemini_client, test_image):
        response = gemini_client.call(test_image, "What do you see in this image? Reply in one sentence.")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_detect_result_type(self, gemini_client, test_image):
        result = gemini_client.detect(test_image)
        assert_detect_result(result)

    def test_detect_confidence_range(self, gemini_client, test_image):
        result = gemini_client.detect(test_image)
        assert 0.0 <= result.confidence <= 1.0


# ================================================================
# Qwen
# ================================================================

class TestQwenIntegration:
    """Requires DASHSCOPE_API_KEY and openai package."""

    def test_call_returns_text(self, qwen_client, test_image):
        response = qwen_client.call(test_image, "What do you see in this image? Reply in one sentence.")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_detect_result_type(self, qwen_client, test_image):
        result = qwen_client.detect(test_image)
        assert_detect_result(result)

    def test_detect_confidence_range(self, qwen_client, test_image):
        result = qwen_client.detect(test_image)
        assert 0.0 <= result.confidence <= 1.0
