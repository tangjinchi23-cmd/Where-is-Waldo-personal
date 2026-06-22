"""核心逻辑测试：VLM 响应解析（无 API、无网络、无 key）。

覆盖 _extract_json / _parse_detect / _parse_select 与 select 能力契约。
"""

import pytest

from llm.vlm_client import (
    DetectResult,
    SelectResult,
    BaseVLMClient,
    GeminiVLMClient,
    get_vlm_client,
    _extract_json,
)


# ── _extract_json ───────────────────────────────────────────────────

class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"present": true, "confidence": 0.9}') == {
            "present": True, "confidence": 0.9}

    def test_markdown_json_block(self):
        assert _extract_json('```json\n{"present": false}\n```') == {"present": False}

    def test_json_embedded_in_prose(self):
        r = _extract_json('answer: {"present": true, "confidence": 0.75} done')
        assert r["present"] is True and r["confidence"] == 0.75

    def test_invalid_returns_empty(self):
        assert _extract_json("no json here") == {}

    def test_empty_returns_empty(self):
        assert _extract_json("") == {}


# ── _parse_detect ───────────────────────────────────────────────────

class TestParseDetect:
    def test_present_true(self):
        r = BaseVLMClient._parse_detect('{"present": true, "confidence": 0.88, "bbox": [0,10,100,200]}')
        assert isinstance(r, DetectResult)
        assert r.has_waldo is True
        assert r.confidence == pytest.approx(0.88)
        assert r.bbox == [0, 10, 100, 200]

    def test_present_false(self):
        r = BaseVLMClient._parse_detect('{"present": false, "bbox": null}')
        assert r.has_waldo is False
        assert r.bbox is None

    def test_present_false_not_skipped(self):
        # present: false 不应被 or 逻辑跳过（旧 bug 复现）
        assert BaseVLMClient._parse_detect('{"present": false}').has_waldo is False

    def test_missing_fields_defaults(self):
        r = BaseVLMClient._parse_detect("{}")
        assert r.has_waldo is False and r.confidence == 0.0 and r.bbox is None

    def test_present_true_infers_confidence(self):
        r = BaseVLMClient._parse_detect('{"present": true}')
        assert r.has_waldo is True and r.confidence == pytest.approx(0.8)


# ── _parse_select（横向单选）────────────────────────────────────────

class TestParseSelect:
    def test_choice_and_per_image(self):
        r = BaseVLMClient._parse_select('{"choice": 0, "confidence": 0.95, "per_image": [true, false]}')
        assert isinstance(r, SelectResult)
        assert r.choice == 0 and r.per_image == [True, False]

    def test_choice_none_is_minus_one(self):
        assert BaseVLMClient._parse_select('{"choice": -1}').choice == -1

    def test_missing_defaults(self):
        r = BaseVLMClient._parse_select("{}")
        assert r.choice == -1 and r.per_image == []

    def test_choice_coerced_to_int(self):
        assert BaseVLMClient._parse_select('{"choice": "2"}').choice == 2

    def test_non_numeric_choice_defaults(self):
        assert BaseVLMClient._parse_select('{"choice": "none"}').choice == -1


# ── select 能力契约 ─────────────────────────────────────────────────

class TestSelectCapability:
    def test_subclass_without_override_raises(self):
        class Dummy(BaseVLMClient):
            def call(self, image_path, prompt, max_tokens=None):
                return ""
            def detect(self, image_path):
                return None

        with pytest.raises(NotImplementedError):
            Dummy().select(["a.jpg"])


# ── factory ─────────────────────────────────────────────────────────

class TestFactory:
    def test_returns_gemini(self):
        assert isinstance(get_vlm_client("gemini"), GeminiVLMClient)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_vlm_client("claude")
