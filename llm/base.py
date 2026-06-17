"""BaseVLMClient 抽象基类及 JSON 解析工具。"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from prompts import DETECT_PROMPT, VERIFY_PROMPT, SELECT_PROMPT
from llm.results import DetectResult, VerifyResult, SelectResult


class BaseVLMClient(ABC):

    DETECT_PROMPT = DETECT_PROMPT
    VERIFY_PROMPT = VERIFY_PROMPT
    SELECT_PROMPT = SELECT_PROMPT

    @abstractmethod
    def call(self, image_path: str, prompt: str, max_tokens: int | None = None) -> str:
        """发送图片 + 自定义 prompt，返回 VLM 原始文本响应。"""

    @abstractmethod
    def detect(self, image_path: str) -> DetectResult:
        """VLM 判断 patch 中是否有 Waldo，返回粗略 bbox 和置信度。"""

    @abstractmethod
    def verify(self, image_path: str) -> VerifyResult:
        """VLM 对裁剪区域二次确认是否是 Waldo。"""

    def select(self, image_paths: list[str]) -> SelectResult:
        """多张候选裁剪图横向单选哪张是 Waldo（默认不支持，需 provider 覆盖）。"""
        raise NotImplementedError(f"{type(self).__name__} does not support select()")

    @staticmethod
    def _parse_detect(text: str) -> DetectResult:
        data = _extract_json(text)
        # 按优先级查 key，用 `in` 判断避免 False 被 or 跳过
        has_waldo = _first(data, "has_waldo", "present", "found", "detected", default=False)
        has_waldo = bool(has_waldo)
        # DETECT_PROMPT now requests confidence; fall back to 0.8/0.0 if model omits it.
        raw_confidence = _first(data, "confidence", "score", "probability", default=None)
        confidence = float(raw_confidence) if raw_confidence is not None else (0.8 if has_waldo else 0.0)
        bbox = _first(data, "bbox", "bounding_box", default=None)
        return DetectResult(
            has_waldo=has_waldo,
            confidence=confidence,
            bbox=bbox,
            raw_response=text,
        )

    @staticmethod
    def _parse_verify(text: str) -> VerifyResult:
        data = _extract_json(text)
        is_waldo = _first(data, "is_waldo", "confirmed", "verified", "is_wally", default=False)
        confidence = _first(data, "confidence", "score", "probability", default=0.0)
        return VerifyResult(
            is_waldo=bool(is_waldo),
            confidence=float(confidence),
            raw_response=text,
        )

    @staticmethod
    def _parse_select(text: str) -> SelectResult:
        data = _extract_json(text)
        raw_choice = _first(data, "choice", "index", "selected", default=-1)
        try:
            choice = int(raw_choice)
        except (TypeError, ValueError):
            choice = -1
        raw_conf = _first(data, "confidence", "score", "probability", default=0.0)
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            confidence = 0.0
        per_image = _first(data, "per_image", "per_crop", default=[])
        per_image = [bool(x) for x in per_image] if isinstance(per_image, list) else []
        return SelectResult(
            choice=choice,
            confidence=confidence,
            per_image=per_image,
            raw_response=text,
        )


def _first(data: dict, *keys, default):
    """按优先级返回 dict 中第一个存在的 key 的值，找不到返回 default。"""
    for key in keys:
        if key in data:
            return data[key]
    return default


def _extract_json(text: str) -> dict:
    """从 VLM 返回文本中提取第一个 JSON 对象，容错处理 markdown 代码块。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}
