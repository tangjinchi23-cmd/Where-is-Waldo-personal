"""测试新版 DETECT_PROMPT 在已知含 Waldo 图片上的召回率。

images_withWaldo/ 目录下的图片均含 Waldo（ground truth = present），
因此该测试直接衡量 false negative 率：
  召回率 = detected / total

直接在 PyCharm 中右键 Run 即可。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

ROOT = os.path.dirname(os.path.dirname(__file__))
IMAGE_DIR = os.path.join(ROOT, "images_withWaldo")
VLM_PROVIDER = "claude"   # 换成 "gpt4o" 可对比


def main():
    print("=" * 60)
    print(f"DETECT_PROMPT 召回率测试  (provider={VLM_PROVIDER})")
    print("=" * 60)

    images = sorted(
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not images:
        print(f"[FAIL] 图片目录为空: {IMAGE_DIR}")
        return

    from llm.vlm_client import get_vlm_client
    vlm = get_vlm_client(VLM_PROVIDER)

    results = []
    for fname in images:
        path = os.path.join(IMAGE_DIR, fname)
        try:
            r = vlm.detect(path)
            label = "HIT " if r.has_waldo else "MISS"
            print(f"  [{label}] {fname:<20} present={r.has_waldo}  conf={r.confidence:.3f}")
            results.append((fname, r.has_waldo, r.confidence, r.raw_response))
        except Exception as e:
            print(f"  [ERR ] {fname:<20} {e}")
            results.append((fname, None, 0.0, str(e)))

    total = len(results)
    detected = sum(1 for _, hw, _, _ in results if hw is True)
    missed = sum(1 for _, hw, _, _ in results if hw is False)
    errors = sum(1 for _, hw, _, _ in results if hw is None)

    print()
    print("-" * 60)
    print(f"总图片数 : {total}")
    print(f"检测到   : {detected}  (HIT)")
    print(f"漏检     : {missed}   (MISS = false negative)")
    print(f"报错     : {errors}")
    if total - errors > 0:
        recall = detected / (total - errors) * 100
        print(f"召回率   : {recall:.1f}%")
    print("-" * 60)

    if missed:
        print("\n漏检详情（raw response 前 120 字符）：")
        for fname, hw, conf, raw in results:
            if hw is False:
                print(f"  {fname}: {raw[:120]!r}")


if __name__ == "__main__":
    main()
