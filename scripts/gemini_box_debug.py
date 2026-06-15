"""排查 gemini_limit「说有但框对=0」疑点：是 bbox 坐标顺序解析错，还是 Gemini 真框错。

对指定 图+尺寸 裁同一区域、调 Gemini 若干次，逐次打印**原始返回**，并把返回框按两种
坐标顺序都算一遍重叠 + 画到图上（红=真值，蓝=Gemini 预测）供肉眼核对。

用法：
    python scripts/gemini_box_debug.py --image 2 --size 384
    python scripts/gemini_box_debug.py --image 10 --size 512 --repeats 2
"""

import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.patch_sampler import parse_bbox_file  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL_NAME = "gemini-3.5-flash"
TEMPERATURE = 0.0
MAX_TOKENS = 2048
OUT_DIR = os.path.join(ROOT, "outputs", "limit_crops")

PROMPT = """This is a crop from a "Where's Waldo?" (Where's Wally?) illustration.
Using your own knowledge of what the Waldo character looks like, decide whether he is
present in this image. He may be small, partially hidden, or blurry — look carefully.

Return strictly this JSON and nothing else:
{"found": true or false, "box_2d": [ymin, xmin, ymax, xmax]}
box_2d must be normalized to 0-1000 with the top-left corner as origin, tightly around
Waldo. If you do not find him, set "found": false and "box_2d": [].
"""


def centered_crop(img_w, img_h, gt, size):
    s = min(img_w, img_h) if size <= 0 else min(size, img_w, img_h)
    cx, cy = (gt[0] + gt[2]) // 2, (gt[1] + gt[3]) // 2
    x = max(0, min(cx - s // 2, img_w - s))
    y = max(0, min(cy - s // 2, img_h - s))
    return x, y, s


def _overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def main():
    img_no, size, repeats = 2, 384, 1
    if "--image" in sys.argv:
        img_no = int(sys.argv[sys.argv.index("--image") + 1])
    if "--size" in sys.argv:
        size = int(sys.argv[sys.argv.index("--size") + 1])
    if "--repeats" in sys.argv:
        repeats = int(sys.argv[sys.argv.index("--repeats") + 1])

    img_path = os.path.join(ROOT, "original-images", f"{img_no}.jpg")
    gt = parse_bbox_file(os.path.join(ROOT, "original-images", "bbox")).get(img_no)
    base = Image.open(img_path).convert("RGB")
    W, H = base.size
    x, y, s = centered_crop(W, H, gt, size)
    gt_in = [gt[0] - x, gt[1] - y, gt[2] - x, gt[3] - y]  # 真值在裁剪坐标系

    os.makedirs(OUT_DIR, exist_ok=True)
    client = genai.Client(api_key=API_KEY)

    print("=" * 72)
    print(f"box debug  image={img_no}.jpg  size={s}px  Waldo gt_in(裁剪坐标)={gt_in}")
    print("=" * 72)

    for i in range(repeats):
        crop = base.crop((x, y, x + s, y + s))
        resp = client.models.generate_content(
            model=MODEL_NAME, contents=[PROMPT, crop],
            config=types.GenerateContentConfig(
                temperature=TEMPERATURE, max_output_tokens=MAX_TOKENS,
                response_mime_type="application/json"),
        )
        raw = (resp.text or "").strip()
        print(f"\n── 第 {i+1} 次 原始返回 ──\n{raw}")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  JSON 解析失败: {e}")
            continue
        box = data.get("box_2d")
        if not box or len(box) != 4:
            print(f"  found={data.get('found')}  无有效 box，跳过")
            continue

        b = [v / 1000 * s for v in box]  # 归一化 → 裁剪像素，保持原顺序
        # 解释 A（脚本现用）：[ymin,xmin,ymax,xmax] → [x1,y1,x2,y2]
        predA = [b[1], b[0], b[3], b[2]]
        # 解释 B（备选）：[xmin,ymin,xmax,ymax] → [x1,y1,x2,y2]
        predB = [b[0], b[1], b[2], b[3]]
        okA, okB = _overlap(predA, gt_in), _overlap(predB, gt_in)
        print(f"  found={data.get('found')}  raw box_2d={box}")
        print(f"  解释A [ymin,xmin,ymax,xmax]→ {[round(v) for v in predA]}  overlap={okA}")
        print(f"  解释B [xmin,ymin,xmax,ymax]→ {[round(v) for v in predB]}  overlap={okB}")

        # 画图：红=真值，蓝=解释A，绿=解释B
        vis = crop.copy()
        d = ImageDraw.Draw(vis)
        d.rectangle(gt_in, outline=(255, 0, 0), width=3)       # 真值
        d.rectangle(predA, outline=(0, 0, 255), width=2)       # 解释A
        d.rectangle(predB, outline=(0, 200, 0), width=2)       # 解释B
        out = os.path.join(OUT_DIR, f"debug_{img_no}_s{s}_{i+1}.jpg")
        vis.save(out)
        print(f"  已存 {os.path.basename(out)}（红=真值 蓝=解释A 绿=解释B）")

    print("\n" + "=" * 72)
    print("判读：若某解释的框稳定压在红框上 → 是解析顺序锅，改顺序即可；")
    print("      若两种解释都偏离红框 → Gemini 真把诱饵当 Waldo（真框错）。")


if __name__ == "__main__":
    main()
