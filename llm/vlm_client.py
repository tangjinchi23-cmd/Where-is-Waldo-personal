"""聚合 import 入口（Gemini-only）。

实现拆解至：
    llm/results.py            DetectResult / SelectResult
    llm/base.py               BaseVLMClient + _extract_json
    llm/factory.py            get_vlm_client
    llm/providers/gemini_client.py  GeminiVLMClient
"""

from llm.results import DetectResult, SelectResult
from llm.base import BaseVLMClient, _extract_json
from llm.factory import get_vlm_client, Provider
from llm.providers.gemini_client import GeminiVLMClient

__all__ = [
    "DetectResult",
    "SelectResult",
    "BaseVLMClient",
    "get_vlm_client",
    "Provider",
    "GeminiVLMClient",
    "_extract_json",
]
