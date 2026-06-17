"""VLM 调用结果数据类。"""

from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass
class SelectResult:
    """横向单选结果：在多张候选裁剪图中挑出唯一真 Waldo。"""
    choice: int                          # 选中候选的 0 基索引；-1 表示都不是
    confidence: float                    # 0.0 ~ 1.0
    per_image: list[bool] = field(default_factory=list)  # 每张候选是否像 Waldo（按入参顺序）
    raw_response: str = ""
