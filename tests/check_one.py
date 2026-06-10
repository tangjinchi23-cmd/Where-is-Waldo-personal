"""对单张图片调用 detect 并打印结果。裁剪自备，本脚本只负责调模型。

    python tests/check_one.py path/to/crop.png
    python tests/check_one.py path/to/crop.png gpt-5.5
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from llm.vlm_client import GPT4oVLMClient


def main():
    if len(sys.argv) < 2:
        print("用法: python tests/check_one.py <图片路径> [模型名，默认 gpt-5.4-mini]")
        return
    image = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-5.4-mini"
    if not os.path.exists(image):
        print(f"[FAIL] 图片不存在: {image}")
        return

    vlm = GPT4oVLMClient(model=model)
    t0 = time.perf_counter()
    r = vlm.detect(image)
    dt = time.perf_counter() - t0

    print(f"image      : {image}")
    print(f"model      : {model}")
    print(f"has_waldo  : {r.has_waldo}")
    print(f"confidence : {r.confidence:.2f}")
    print(f"raw        : {r.raw_response!r}")
    print(f"elapsed    : {dt:.2f}s")


if __name__ == "__main__":
    main()
