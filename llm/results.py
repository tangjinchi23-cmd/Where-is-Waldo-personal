"""VLM 调用结果数据类。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectResult:
    has_waldo: bool
    confidence: float              # 0.0 ~ 1.0
    bbox: list[int] | None = None  # [x, y, w, h]，相对 patch 坐标；无则 None
    raw_response: str = ""


@dataclass
class VerifyResult:
    is_waldo: bool
    confidence: float
    raw_response: str = ""
