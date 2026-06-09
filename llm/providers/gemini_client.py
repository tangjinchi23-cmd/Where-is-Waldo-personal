"""Google Gemini VLM 实现。"""

from __future__ import annotations

from llm.base import BaseVLMClient
from llm.results import DetectResult, VerifyResult


class GeminiVLMClient(BaseVLMClient):
    """使用 Google Gemini Vision 进行 Waldo 检测。"""

    def __init__(self, model: str = "gemini-1.5-flash", max_tokens: int = 256):
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError("pip install google-generativeai") from e
        self._genai = genai
        self._model_name = model
        self._max_tokens = max_tokens

    def call(self, image_path: str, prompt: str, max_tokens: int | None = None) -> str:
        import PIL.Image
        model = self._genai.GenerativeModel(self._model_name)
        img = PIL.Image.open(image_path)
        response = model.generate_content(
            [prompt, img],
            generation_config={"max_output_tokens": max_tokens or self._max_tokens},
        )
        return response.text

    def detect(self, image_path: str) -> DetectResult:
        return self._parse_detect(self.call(image_path, self.DETECT_PROMPT))

    def verify(self, image_path: str) -> VerifyResult:
        return self._parse_verify(self.call(image_path, self.VERIFY_PROMPT))
