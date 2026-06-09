"""Qwen-VL 视觉模型实现，用于 Waldo 检测。

通过百炼 OpenAI 兼容接口调用，默认 qwen-vl-max-latest。
注意：本类专用于 Qwen，base_url 指向百炼兼容端点。
"""

from __future__ import annotations

import mimetypes

from llm.base import BaseVLMClient
from llm.results import DetectResult, VerifyResult
from vision.image_utils import image_to_base64


class QwenVLMClient(BaseVLMClient):
    """使用通义千问 Qwen-VL 进行 Waldo 检测。"""

    # 百炼 OpenAI 兼容端点（国内地域）
    _BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        model: str = "qwen-vl-max",
        max_tokens: int = 100000,
        api_key: str | None = None,          # 不传则读 DASHSCOPE_API_KEY
        high_resolution: bool = True,         # 默认开高分辨率，Waldo 细节密集
        temperature: float = 0.1,             # 检测任务要稳定，调低
        seed: int | None = 42,                # 固定 seed 保证可复现（毕设答辩用）
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e
        import os

        self._client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=self._BASE_URL,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._high_resolution = high_resolution
        self._temperature = temperature
        self._seed = seed

    def call(self, image_path: str, prompt: str, max_tokens: int | None = None) -> str:
        b64 = image_to_base64(image_path)
        # 动态 MIME：不再写死 jpeg，按文件扩展名判断（Qwen 要求格式匹配）
        mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"

        # 强制 JSON 输出。注意：用 json_object 时提示词里必须出现“json”字样，否则报错。
        # 这里追加一句兜底指令，确保即使外部 prompt 没写也不会报错。
        json_prompt = prompt
        if "json" not in prompt.lower():
            json_prompt = f"{prompt}\n\n请严格按照 JSON 格式输出。"

        response = self._client.chat.completions.create(
            model=self._model,
            # max_tokens 即将废弃，改用 max_completion_tokens
            max_completion_tokens=max_tokens or self._max_tokens,
            temperature=self._temperature,
            seed=self._seed,
            response_format={"type": "json_object"},   # 强制结构化输出
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": json_prompt},
                ],
            }],
            # Qwen 特有参数必须放 extra_body，不能当关键字直接传
            extra_body={"vl_high_resolution_images": self._high_resolution},
        )

        content = response.choices[0].message.content
        if content is None:
            # 触发审查/返回空时，别让下游解析裸崩
            raise RuntimeError(
                f"模型返回空内容，finish_reason="
                f"{response.choices[0].finish_reason}"
            )
        return content

    def detect(self, image_path: str) -> DetectResult:
        return self._parse_detect(self.call(image_path, self.DETECT_PROMPT))

    def verify(self, image_path: str) -> VerifyResult:
        return self._parse_verify(self.call(image_path, self.VERIFY_PROMPT))
