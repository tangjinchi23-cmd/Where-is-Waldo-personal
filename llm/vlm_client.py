"""兼容垫片：保持旧 import 路径可用。

所有实现已拆解至：
    llm/results.py
    llm/base.py
    llm/factory.py
    llm/providers/claude_client.py
    llm/providers/gpt4o_client.py
    llm/providers/gemini_client.py
    llm/providers/qwen_client.py
"""

from llm.results import DetectResult, VerifyResult, SelectResult
from llm.base import BaseVLMClient, _extract_json
from llm.factory import get_vlm_client, Provider
from llm.providers.claude_client import ClaudeVLMClient
from llm.providers.gpt4o_client import GPT4oVLMClient
from llm.providers.gemini_client import GeminiVLMClient
from llm.providers.qwen_client import QwenVLMClient

__all__ = [
    "DetectResult",
    "VerifyResult",
    "SelectResult",
    "BaseVLMClient",
    "get_vlm_client",
    "Provider",
    "ClaudeVLMClient",
    "GPT4oVLMClient",
    "GeminiVLMClient",
    "QwenVLMClient",
    "_extract_json",
]
