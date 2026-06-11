"""读取根目录 config.json，给 quick 测试脚本提供可调参数 + 构造 VLM 客户端。

把 provider / model / temperature / repeats / limit 等手调参数集中到 config.json，
改 json 即可，无需动脚本代码。缺失的键用内置默认值兜底。
"""

import json
import os

from llm.vlm_client import get_vlm_client
from llm.base import _extract_json
from prompts import DETECT_PROMPT

# prompt engineering 专用：在真 DETECT_PROMPT 后临时追加「附原因」要求，
# 只走测试路径；正式 agent 的 detect 节点仍用精简 DETECT_PROMPT，不产出 reason。
_REASON_SUFFIX = (
    "\n\nAdditionally, add a \"reason\" field: ONE short sentence explaining your "
    "decision — what you saw (or did not see) that drove present/confidence.\n"
    'Output JSON: {"present": true/false, "confidence": 0.0-1.0, "reason": "..."}'
)

ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.json")

_DEFAULTS = {
    "provider": "gpt4o",
    "model": "gpt-5.4-mini",
    "temperature": 0,
    "max_tokens": 1024,
    "repeats": 1,
    "limit": 0,
}


def load_config() -> dict:
    """加载 config.json（忽略下划线开头的注释键），缺键用默认值补齐。"""
    cfg = dict(_DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            user = json.load(f)
        cfg.update({k: v for k, v in user.items() if not k.startswith("_")})
    return cfg


def build_vlm(cfg: dict):
    """按 config 构造 VLM 客户端。

    temperature 只对 gpt4o 非推理模型有意义，故仅在 provider=gpt4o 时透传；
    其它 provider 忽略该参数，避免给不接受 temperature 的客户端传参报错。
    """
    kwargs = {"model": cfg["model"]}
    if cfg["provider"] == "gpt4o":
        if cfg.get("temperature") is not None:
            kwargs["temperature"] = cfg["temperature"]
        if cfg.get("max_tokens"):
            kwargs["max_tokens"] = cfg["max_tokens"]
    return get_vlm_client(cfg["provider"], **kwargs)


def run_repeats(vlm, image_path: str, repeats: int):
    """跑 repeats 次「带原因」detect，返回 (多数票是否present, 平均conf, 明细列表)。

    明细列表每项 = (present, confidence, reason)，供测试脚本逐次打印原因。
    用 DETECT_PROMPT + _REASON_SUFFIX，便于 prompt engineering 时看决策依据。
    """
    votes = 0
    conf_sum = 0.0
    details: list[tuple[bool, float, str]] = []
    for _ in range(repeats):
        raw = vlm.call(image_path, DETECT_PROMPT + _REASON_SUFFIX)
        data = _extract_json(raw)
        present = bool(data.get("present", data.get("has_waldo", False)))
        conf = data.get("confidence")
        conf = float(conf) if conf is not None else (0.8 if present else 0.0)
        reason = str(data.get("reason", "")).strip()
        votes += int(present)
        conf_sum += conf
        details.append((present, conf, reason))
    fired = votes * 2 >= repeats
    return fired, conf_sum / repeats, details
