"""一次性校验所有 VLM Provider 的 API Key 可用性。

有 key 的 provider 会调用 detect()，没有 key 的直接标记 SKIP。
运行结束后打印汇总表格。

直接在 PyCharm 中右键 Run 即可。
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

TEST_IMAGE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "original-images", "1.jpg")


# ----------------------------------------------------------------
# 每个 provider 的检查逻辑
# ----------------------------------------------------------------

def check_provider(name: str, make_client, image_path: str) -> dict:
    result = {"provider": name, "status": "", "has_waldo": "-", "confidence": "-",
              "latency_ms": "-", "note": ""}
    try:
        client = make_client()
        t0 = time.perf_counter()
        dr = client.detect(image_path)
        elapsed = (time.perf_counter() - t0) * 1000
        result.update({
            "status": "OK",
            "has_waldo": str(dr.has_waldo),
            "confidence": f"{dr.confidence:.3f}",
            "latency_ms": f"{elapsed:.0f}",
            "note": f"bbox={dr.bbox}",
        })
    except Exception as e:
        result.update({"status": "FAIL", "note": str(e)[:60]})
    return result


def make_claude():
    from llm.vlm_client import ClaudeVLMClient
    return ClaudeVLMClient()


def make_gpt4o():
    from llm.vlm_client import GPT4oVLMClient
    return GPT4oVLMClient()


def make_gemini():
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    from llm.vlm_client import GeminiVLMClient
    return GeminiVLMClient()


def make_qwen():
    from llm.vlm_client import QwenVLMClient
    return QwenVLMClient()


# ----------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------

PROVIDERS = [
    ("claude",  "ANTHROPIC_API_KEY",  make_claude),
    ("gpt4o",   "OPENAI_API_KEY",     make_gpt4o),
    ("gemini",  "GOOGLE_API_KEY",     make_gemini),
    ("qwen",    "DASHSCOPE_API_KEY",  make_qwen),
]


def main():
    if not os.path.exists(TEST_IMAGE):
        print(f"[ERROR] 测试图片不存在: {TEST_IMAGE}")
        return

    print("=" * 70)
    print("VLM Provider 全量校验")
    print(f"测试图片: {TEST_IMAGE}")
    print("=" * 70)

    rows = []
    for name, env_key, factory in PROVIDERS:
        api_key = os.environ.get(env_key, "")
        if not api_key:
            rows.append({
                "provider": name, "status": "SKIP",
                "has_waldo": "-", "confidence": "-", "latency_ms": "-",
                "note": f"{env_key} 未设置",
            })
            print(f"[SKIP]  {name:<10} — {env_key} 未设置")
            continue

        print(f"[RUN]   {name:<10} 调用中...", end=" ", flush=True)
        row = check_provider(name, factory, TEST_IMAGE)
        rows.append(row)
        if row["status"] == "OK":
            print(f"OK  ({row['latency_ms']} ms)")
        else:
            print(f"FAIL")

    # 汇总表格
    print("\n" + "=" * 70)
    print(f"{'Provider':<12} {'Status':<8} {'has_waldo':<12} {'confidence':<12} {'ms':<8} note")
    print("-" * 70)
    for r in rows:
        print(f"{r['provider']:<12} {r['status']:<8} {r['has_waldo']:<12} "
              f"{r['confidence']:<12} {r['latency_ms']:<8} {r['note']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
