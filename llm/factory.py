"""VLM 客户端工厂。"""

from __future__ import annotations

from typing import Literal

from llm.base import BaseVLMClient
from llm.providers.claude_client import ClaudeVLMClient
from llm.providers.gpt4o_client import GPT4oVLMClient
from llm.providers.gemini_client import GeminiVLMClient
from llm.providers.qwen_client import QwenVLMClient

Provider = Literal["claude", "gpt4o", "gemini", "qwen"]

_REGISTRY: dict[str, type[BaseVLMClient]] = {
    "claude": ClaudeVLMClient,
    "gpt4o":  GPT4oVLMClient,
    "gemini": GeminiVLMClient,
    "qwen":   QwenVLMClient,
}


def get_vlm_client(provider: Provider = "claude", **kwargs) -> BaseVLMClient:
    """根据 provider 名称返回对应的 VLM 客户端实例。

    Args:
        provider: "claude" | "gpt4o" | "gemini" | "qwen"
        **kwargs: 传递给对应客户端的额外参数（model, max_tokens 等）。
    """
    if provider not in _REGISTRY:
        raise ValueError(f"Unknown provider: {provider!r}. Choose from {list(_REGISTRY)}")
    return _REGISTRY[provider](**kwargs)
