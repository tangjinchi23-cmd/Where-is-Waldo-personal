"""校验 Google Gemini API Key 可用性，并做一次 detect 调用。

直接在 PyCharm 中右键 Run 即可。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

TEST_IMAGE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "original-images", "1.jpg")


def main():
    print("=" * 50)
    print("Gemini API Key 校验")
    print("=" * 50)

    # 1. 检查 key
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("[FAIL] GOOGLE_API_KEY 未设置，请检查 .env 文件")
        return
    print(f"[OK]   Key 已读取: ...{api_key[-6:]}")

    # 2. 导入 SDK
    try:
        import google.generativeai as genai
        print("[OK]   google-generativeai SDK 已安装")
    except ImportError:
        print("[FAIL] google-generativeai 未安装，请执行: pip install google-generativeai")
        return

    # 3. 纯文本 ping
    print("\n--- 纯文本 ping ---")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(
            "Reply with the single word: PONG",
            generation_config={"max_output_tokens": 16},
        )
        reply = resp.text.strip()
        print(f"[OK]   响应: {reply!r}")
    except Exception as e:
        print(f"[FAIL] 文本调用失败: {e}")
        return

    # 4. 图片 detect 调用
    print("\n--- 图片 detect 调用 ---")
    if not os.path.exists(TEST_IMAGE):
        print(f"[SKIP] 测试图片不存在: {TEST_IMAGE}")
    else:
        try:
            from llm.vlm_client import GeminiVLMClient
            vlm = GeminiVLMClient()
            result = vlm.detect(TEST_IMAGE)
            print(f"[OK]   has_waldo  : {result.has_waldo}")
            print(f"[OK]   confidence : {result.confidence:.3f}")
            print(f"[OK]   bbox       : {result.bbox}")
            print(f"[OK]   raw (前80字符): {result.raw_response[:80]!r}")
        except Exception as e:
            print(f"[FAIL] detect 调用失败: {e}")

    print("\n[DONE] Gemini 校验完成")


if __name__ == "__main__":
    main()
