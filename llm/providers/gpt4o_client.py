"""OpenAI VLM 实现，支持 GPT-5.x 系列视觉模型。

支持的视觉模型（截至 2026-06）：
    gpt-5.5          — 当前旗舰，复杂视觉推理首选
    gpt-5.4-mini     — 轻量版，速度/成本平衡，适合批量评测
    gpt-5.4-nano     — 最小最快，适合流程验证
    注：当前 OpenAI 全系模型均原生支持图像输入（视觉）。
"""

from __future__ import annotations

import mimetypes

from llm.base import BaseVLMClient
from llm.results import DetectResult
from vision.image_utils import image_to_base64


class GPT4oVLMClient(BaseVLMClient):
    """使用 OpenAI 视觉模型进行 Waldo 检测（默认 gpt-5.5）。"""

    def __init__(
        self,
        model: str = "gpt-5.5",
        max_tokens: int = 1024,
        detail: str = "high",          # Waldo 细节密集，默认 high；可设 low/auto
        temperature: float = 1.0,      # 推理模型（gpt-5.5）只支持 1.0；切换到 gpt-5.4-mini 等非推理模型时可调低
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e
        self._client = OpenAI()
        self._model = model
        self._max_tokens = max_tokens
        self._detail = detail
        self._temperature = temperature

    def call(self, image_path: str, prompt: str, max_tokens: int | None = None) -> str:
        b64 = image_to_base64(image_path)
        mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"  # 动态 MIME

        # 推理模型（gpt-5.5 等）只支持 temperature=1，传其他值会报错，故跳过
        extra = {} if self._temperature == 1.0 else {"temperature": self._temperature}
        response = self._client.chat.completions.create(
            model=self._model,
            max_completion_tokens=max_tokens or self._max_tokens,
            response_format={"type": "json_object"},  # 强制 JSON 输出，防止模型自创 key
            **extra,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                            "detail": self._detail,   # OpenAI 标准参数，直接写在 image_url 内
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content or ""

    def detect(self, image_path: str) -> DetectResult:
        return self._parse_detect(self.call(image_path, self.DETECT_PROMPT))