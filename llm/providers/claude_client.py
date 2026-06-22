"""Claude (Anthropic) VLM 实现。"""

from __future__ import annotations

from llm.base import BaseVLMClient
from llm.results import DetectResult
from vision.image_utils import image_to_base64


class ClaudeVLMClient(BaseVLMClient):
    """使用 Anthropic Claude claude-sonnet-4-6 进行 Waldo 检测。"""

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 256):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("pip install anthropic") from e
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def call(self, image_path: str, prompt: str, max_tokens: int | None = None) -> str:
        b64 = image_to_base64(image_path)
        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens or self._max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return message.content[0].text

    def detect(self, image_path: str) -> DetectResult:
        return self._parse_detect(self.call(image_path, self.DETECT_PROMPT))
