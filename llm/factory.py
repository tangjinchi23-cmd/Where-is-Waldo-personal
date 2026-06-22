"""VLM 客户端工厂（Gemini-only）。"""

from __future__ import annotations

from typing import Literal

from llm.base import BaseVLMClient
from llm.providers.gemini_client import GeminiVLMClient

Provider = Literal["gemini"]

_REGISTRY: dict[str, type[BaseVLMClient]] = {
    "gemini": GeminiVLMClient,
}


def get_vlm_client(provider: Provider = "gemini", **kwargs) -> BaseVLMClient:
    """返回 VLM 客户端实例。当前只支持 gemini。

    Args:
        provider: 仅 "gemini"。
        **kwargs: 传给客户端的额外参数（model, max_tokens 等）。
    """
    if provider not in _REGISTRY:
        raise ValueError(f"Unknown provider: {provider!r}. Choose from {list(_REGISTRY)}")
    return _REGISTRY[provider](**kwargs)
