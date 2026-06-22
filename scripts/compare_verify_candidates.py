"""诊断脚本：把 verify 阶段的多张候选裁剪图一起发给 Gemini，让它单选哪张才是真 Waldo。

背景：在密集难图（如 2.jpg）上，gpt-5.5 verify 会把多个候选都判成 Waldo（实测 4 个里
判了 3 个 True），最终只能靠 verify_confidence 挑。本脚本换个思路——把所有候选**一次性**
摆给 Gemini 做**单选**，看它能否更干净地指出唯一的真 Waldo，用于对比/诊断。

用法（需项目根 .env 配 GOOGLE_API_KEY）：
    python scripts/compare_verify_candidates.py                 # 默认 outputs/verify/verify*.jpg
    python scripts/compare_verify_candidates.py outputs/verify  # 指定目录
"""

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
import PIL.Image

ROOT = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(ROOT, ".env"))

MODEL = "gemini-3.5-flash"

COMPARE_PROMPT = (
    "These are several close-up crops from a 'Where's Waldo' puzzle, each suspected to "
    "contain Waldo (Wally). They are given in order, indexed from 0. At most ONE of them "
    "is the real Waldo. Use your own knowledge of what Waldo looks like (red-and-white "
    "striped bobble hat, round glasses, red-white striped shirt, slim build).\n\n"
    "Reply with ONLY this JSON, no markdown:\n"
    '{"choice": <index of the real Waldo, or -1 if none is Waldo>, '
    '"confidence": 0.0-1.0, "per_image": [true/false, ...], "reason": "one short sentence"}\n'
    "  - choice: 0-based index of the crop that is the real Waldo (-1 if none)\n"
    "  - per_image: for each crop in given order, true if that crop looks like Waldo\n"
)


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except Exception:
        return {}


def _verify_index(path: str) -> int:
    m = re.search(r"verify(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def main() -> None:
    verify_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "outputs", "verify")
    if not os.path.isabs(verify_dir):
        verify_dir = os.path.join(ROOT, verify_dir)

    paths = sorted(glob.glob(os.path.join(verify_dir, "verify*.jpg")), key=_verify_index)
    if not paths:
        print(f"未找到 {verify_dir}/verify*.jpg")
        return

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ 未找到 GOOGLE_API_KEY，请在项目根 .env 配置。")
        return

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL)

    print(f"模型：{MODEL}")
    print(f"候选图（{len(paths)} 张）：")
    for i, p in enumerate(paths):
        print(f"  [{i}] {os.path.relpath(p, ROOT)}")

    content: list = [COMPARE_PROMPT]
    for i, p in enumerate(paths):
        content.append(f"Image {i}:")
        content.append(PIL.Image.open(p))

    resp = model.generate_content(content, generation_config={"max_output_tokens": 1024})
    data = _extract_json(resp.text)

    print("\n--- Gemini 原始响应 ---")
    print((resp.text or "").strip())

    print("\n--- 解析 ---")
    choice = data.get("choice")
    if isinstance(choice, int) and 0 <= choice < len(paths):
        print(f"  Gemini 单选: 候选 [{choice}] → {os.path.relpath(paths[choice], ROOT)}")
    else:
        print(f"  Gemini 单选: {choice}（无有效候选 / 解析失败）")
    print(f"  confidence : {data.get('confidence')}")
    print(f"  per_image  : {data.get('per_image')}")
    print(f"  reason     : {data.get('reason')}")


if __name__ == "__main__":
    main()
