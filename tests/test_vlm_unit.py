"""Unit tests for llm/vlm_client.py — no API calls, no network, no keys needed.

Covers:
  - _extract_json: all edge cases for VLM response parsing
  - _parse_detect / _parse_verify: field extraction and defaults
  - get_vlm_client: factory routing and error on unknown provider
  - DetectResult / VerifyResult: dataclass defaults
  - BaseVLMClient: abstract method enforcement
"""

import json
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm.vlm_client import (
    DetectResult,
    VerifyResult,
    BaseVLMClient,
    ClaudeVLMClient,
    get_vlm_client,
    _extract_json,
)


# ================================================================
# _extract_json
# ================================================================

class TestExtractJson:
    def test_plain_json(self):
        text = '{"has_waldo": true, "confidence": 0.9}'
        assert _extract_json(text) == {"has_waldo": True, "confidence": 0.9}

    def test_markdown_json_block(self):
        text = '```json\n{"has_waldo": false, "confidence": 0.1}\n```'
        assert _extract_json(text) == {"has_waldo": False, "confidence": 0.1}

    def test_markdown_block_no_lang(self):
        text = '```\n{"is_waldo": true, "confidence": 0.8}\n```'
        assert _extract_json(text) == {"is_waldo": True, "confidence": 0.8}

    def test_json_embedded_in_prose(self):
        text = 'Here is my answer: {"has_waldo": true, "confidence": 0.75, "bbox": null} hope that helps!'
        result = _extract_json(text)
        assert result["has_waldo"] is True
        assert result["confidence"] == 0.75

    def test_json_with_bbox(self):
        text = '{"has_waldo": true, "confidence": 0.95, "bbox": [10, 20, 50, 80]}'
        result = _extract_json(text)
        assert result["bbox"] == [10, 20, 50, 80]

    def test_completely_invalid_returns_empty(self):
        assert _extract_json("no json here at all") == {}

    def test_empty_string_returns_empty(self):
        assert _extract_json("") == {}

    def test_nested_json(self):
        text = '{"grid_scores": [[0.9, 0.2], [0.5, 0.8]]}'
        result = _extract_json(text)
        assert result["grid_scores"] == [[0.9, 0.2], [0.5, 0.8]]

    def test_whitespace_stripped(self):
        text = '  \n  {"has_waldo": false, "confidence": 0.0}  \n  '
        assert _extract_json(text)["has_waldo"] is False

    def test_extra_text_after_markdown_fence(self):
        text = '```json\n{"confidence": 0.5}\n```\n\nSome trailing text'
        result = _extract_json(text)
        assert result.get("confidence") == 0.5


# ================================================================
# BaseVLMClient._parse_detect
# ================================================================

class TestParseDetect:
    def test_has_waldo_true(self):
        text = '{"has_waldo": true, "confidence": 0.88, "bbox": [0, 10, 100, 200]}'
        result = BaseVLMClient._parse_detect(text)
        assert isinstance(result, DetectResult)
        assert result.has_waldo is True
        assert result.confidence == pytest.approx(0.88)
        assert result.bbox == [0, 10, 100, 200]
        assert result.raw_response == text

    def test_has_waldo_false(self):
        text = '{"has_waldo": false, "confidence": 0.05, "bbox": null}'
        result = BaseVLMClient._parse_detect(text)
        assert result.has_waldo is False
        assert result.bbox is None

    def test_missing_fields_use_defaults(self):
        result = BaseVLMClient._parse_detect("{}")
        assert result.has_waldo is False
        assert result.confidence == 0.0
        assert result.bbox is None

    def test_invalid_json_uses_defaults(self):
        result = BaseVLMClient._parse_detect("this is not json")
        assert result.has_waldo is False
        assert result.confidence == 0.0

    def test_confidence_coerced_to_float(self):
        text = '{"has_waldo": true, "confidence": 1, "bbox": null}'
        result = BaseVLMClient._parse_detect(text)
        assert isinstance(result.confidence, float)
        assert result.confidence == 1.0

    # --- 新版 DETECT_PROMPT 格式（只返回 present 字段）---

    def test_new_format_present_true(self):
        # 新 prompt 只要求 {"present": true}，无 confidence/bbox
        result = BaseVLMClient._parse_detect('{"present": true}')
        assert result.has_waldo is True
        assert result.confidence == pytest.approx(0.8)  # 推断默认值
        assert result.bbox is None

    def test_new_format_present_false(self):
        result = BaseVLMClient._parse_detect('{"present": false}')
        assert result.has_waldo is False
        assert result.confidence == pytest.approx(0.0)  # 推断默认值

    def test_new_format_present_overridden_by_explicit_confidence(self):
        # 若模型同时返回了 confidence，以模型返回值为准
        result = BaseVLMClient._parse_detect('{"present": true, "confidence": 0.55}')
        assert result.has_waldo is True
        assert result.confidence == pytest.approx(0.55)

    def test_alias_present_false_not_skipped(self):
        # 确保 present: false 不会被 or 逻辑跳过（旧 bug 复现测试）
        result = BaseVLMClient._parse_detect('{"present": false}')
        assert result.has_waldo is False

    def test_alias_score_used_as_confidence(self):
        # gpt-5.5 有时用 score 替代 confidence
        result = BaseVLMClient._parse_detect('{"present": true, "score": 0.72}')
        assert result.confidence == pytest.approx(0.72)


# ================================================================
# BaseVLMClient._parse_verify
# ================================================================

class TestParseVerify:
    def test_is_waldo_true(self):
        text = '{"is_waldo": true, "confidence": 0.97, "reason": "red-white stripes visible"}'
        result = BaseVLMClient._parse_verify(text)
        assert isinstance(result, VerifyResult)
        assert result.is_waldo is True
        assert result.confidence == pytest.approx(0.97)

    def test_is_waldo_false(self):
        text = '{"is_waldo": false, "confidence": 0.02}'
        result = BaseVLMClient._parse_verify(text)
        assert result.is_waldo is False

    def test_missing_fields_use_defaults(self):
        result = BaseVLMClient._parse_verify("{}")
        assert result.is_waldo is False
        assert result.confidence == 0.0

    def test_raw_response_preserved(self):
        text = '{"is_waldo": false, "confidence": 0.1}'
        result = BaseVLMClient._parse_verify(text)
        assert result.raw_response == text


# ================================================================
# DetectResult / VerifyResult dataclass defaults
# ================================================================

class TestDataclasses:
    def test_detect_result_defaults(self):
        r = DetectResult(has_waldo=True, confidence=0.5)
        assert r.bbox is None
        assert r.raw_response == ""

    def test_detect_result_with_bbox(self):
        r = DetectResult(has_waldo=True, confidence=0.9, bbox=[1, 2, 3, 4])
        assert r.bbox == [1, 2, 3, 4]

    def test_verify_result_defaults(self):
        r = VerifyResult(is_waldo=False, confidence=0.0)
        assert r.raw_response == ""


# ================================================================
# get_vlm_client factory
# ================================================================

class TestGetVlmClient:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_vlm_client("nonexistent_provider")

    def test_returns_claude_client_type(self):
        # ClaudeVLMClient constructor calls anthropic.Anthropic() — but we only
        # check the type, and anthropic is installed, so constructor runs.
        # If ANTHROPIC_API_KEY is missing the constructor still returns the object
        # (API key is checked at call time, not init time).
        client = get_vlm_client("claude")
        assert isinstance(client, ClaudeVLMClient)

    def test_factory_passes_kwargs(self):
        client = get_vlm_client("claude", max_tokens=128)
        assert client._max_tokens == 128

    def test_invalid_provider_message_lists_valid(self):
        with pytest.raises(ValueError) as exc:
            get_vlm_client("badprovider")
        msg = str(exc.value)
        assert "claude" in msg
        assert "gpt4o" in msg
        assert "gemini" in msg
        assert "qwen" in msg


# ================================================================
# Abstract method enforcement
# ================================================================

class TestAbstractEnforcement:
    def test_cannot_instantiate_base_directly(self):
        with pytest.raises(TypeError):
            BaseVLMClient()

    def test_partial_implementation_raises(self):
        class Partial(BaseVLMClient):
            def call(self, image_path, prompt, max_tokens=None):
                return ""
            # missing detect and verify

        with pytest.raises(TypeError):
            Partial()
