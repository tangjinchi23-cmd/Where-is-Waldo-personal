"""Gemini 极限能力探针 —— 单图、逐渐放大区域，找 Gemini 还能"真正注意到"Waldo 的最大尺寸。

动机：detect 现按 200×200 切图。若 Gemini 能在更大的区域里依然定位 Waldo，就能大幅
减少 patch 数。本脚本以**单张图**的 Waldo 真值框为中心，裁出一系列**逐渐增大**的正方形
区域（Waldo 绝对像素不变，但相对越来越小、干扰物越来越多），逐尺寸问 Gemini。

⚠️ 关键：按项目铁律「高召回会骗人」——只问 found 会被"无脑说有"骗成极限无限大。故让
   Gemini **同时回传 bbox**，再用真值框校验位置：只有 `found=true 且框与真值有重叠` 才算
   "真正注意到"。每个尺寸重复 REPEATS 次取命中率，找出仍稳定命中的最大尺寸。

用法（需项目根 .env 配 GOOGLE_API_KEY）：
    python scripts/gemini_limit.py                  # 默认测 1.jpg
    python scripts/gemini_limit.py --image 7        # 测 7.jpg（用 bbox 第 7 行真值）
    python scripts/gemini_limit.py --image 1 --repeats 5
"""

import json
import os
import re
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw

# 复用项目里的真值解析
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.patch_sampler import parse_bbox_file  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
API_KEY = os.getenv("GOOGLE_API_KEY")

# ============ 配置区 ============
MODEL_NAME = "gemini-3.5-flash"
TEMPERATURE = 0.0
MAX_TOKENS = 2048

# 逐尺寸探测的正方形边长（px）。0 = 全图。超过图像尺寸的会被 clamp 到全图。
# 含 Gemini 按分辨率推荐的 patch 档位：中高分辨率(~2000×1500)宜 256/384，
# 超高分辨率(>4000×3000)宜 512；更大的尺寸用于探边界（最大能撑多大区域）。
SIZES = [256, 384, 512, 768, 1024, 1536, 0]
REPEATS = 3            # 每个尺寸重复次数（取命中率，抵消模型随机性）
REQUEST_INTERVAL = 1.0  # 请求间主动间隔，防限流
MAX_RETRIES = 5

BBOX_FILE = os.path.join(ROOT, "original-images", "bbox")
OUT_DIR = os.path.join(ROOT, "outputs", "limit_crops")
# ================================

PROMPT = """This is a crop from a "Where's Waldo?" (Where's Wally?) illustration.
Using your own knowledge of what the Waldo character looks like, decide whether he is
present in this image. He may be small, partially hidden, or blurry — look carefully.

Return strictly this JSON and nothing else:
{"found": true or false, "box_2d": [ymin, xmin, ymax, xmax]}
box_2d must be normalized to 0-1000 with the top-left corner as origin, tightly around
Waldo. If you do not find him, set "found": false and "box_2d": [].
"""


def centered_crop(img_w, img_h, gt, size):
    """以真值框中心裁 size×size 正方形；size<=0 或超界则取全图最大可容尺寸。返回 (x,y,s)。"""
    s = min(img_w, img_h) if size <= 0 else min(size, img_w, img_h)
    cx, cy = (gt[0] + gt[2]) // 2, (gt[1] + gt[3]) // 2
    x = max(0, min(cx - s // 2, img_w - s))
    y = max(0, min(cy - s // 2, img_h - s))
    return x, y, s


def _overlap(a, b):
    """两个 [x1,y1,x2,y2] 是否有非零交集。"""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def query(client, image_path):
    """调 Gemini，返回 (found, box_2d|None, error|None)。box_2d = [ymin,xmin,ymax,xmax] 0-1000。"""
    img = Image.open(image_path)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=[PROMPT, img],
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS,
                    response_mime_type="application/json",
                ),
            )
            content = (resp.text or "").strip()
            if not content:
                return None, None, "空响应（token 截断或被拦截）"
            data = json.loads(content)
            box = data.get("box_2d") or None
            if box is not None and len(box) != 4:
                box = None
            return bool(data.get("found", False)), box, None
        except json.JSONDecodeError as e:
            return None, None, f"JSON 解析失败: {e}"
        except Exception as e:
            msg = str(e)
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and attempt < MAX_RETRIES - 1:
                m = re.search(r"retryDelay.*?(\d+)\s*s", msg)
                wait = (int(m.group(1)) + 2) if m else 5 * (2 ** attempt)
                print(f"      ⏳ 限流，等待 {wait}s 重试 ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            return None, None, f"API 异常: {e}"


def main():
    dry_run = "--dry-run" in sys.argv
    if not dry_run and not API_KEY:
        print("❌ 未找到 GOOGLE_API_KEY，请在项目根 .env 配置。")
        return

    img_no = 1
    repeats = REPEATS
    if "--image" in sys.argv:
        img_no = int(sys.argv[sys.argv.index("--image") + 1])
    if "--repeats" in sys.argv:
        repeats = int(sys.argv[sys.argv.index("--repeats") + 1])

    img_path = os.path.join(ROOT, "original-images", f"{img_no}.jpg")
    if not os.path.isfile(img_path):
        print(f"❌ 找不到图片：{img_path}")
        return

    gt = parse_bbox_file(BBOX_FILE).get(img_no)
    if gt is None:
        print(f"❌ bbox 文件第 {img_no} 行无有效真值，无法做位置校验。")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    client = None if dry_run else genai.Client(api_key=API_KEY)
    base = Image.open(img_path).convert("RGB")
    W, H = base.size

    print("=" * 72)
    mode = "DRY-RUN（只裁图，不调 API）" if dry_run else f"model={MODEL_NAME}"
    print(f"Gemini 极限探针  {mode}  image={img_no}.jpg ({W}×{H})")
    print(f"Waldo 真值框 gt={gt}  (≈{gt[2]-gt[0]}×{gt[3]-gt[1]}px)  repeats={repeats}")
    print("=" * 72)
    if dry_run:
        print(f"{'区域尺寸':>10} | {'Waldo占比':>9} | 裁剪图(已画真值红框)")
    else:
        print(f"{'区域尺寸':>10} | {'Waldo占比':>9} | {'说有':>5} | {'框对':>5} | 命中率")
    print("-" * 72)

    t0 = time.perf_counter()
    rows, deduped = [], set()
    for size in SIZES:
        x, y, s = centered_crop(W, H, gt, size)
        if s in deduped:   # 全图与超界尺寸去重，避免重复测同一张
            continue
        deduped.add(s)

        # 真值框在裁剪坐标系中的位置
        gt_in = [gt[0] - x, gt[1] - y, gt[2] - x, gt[3] - y]
        waldo_ratio = ((gt[2] - gt[0]) * (gt[3] - gt[1])) / (s * s) * 100
        tag = f"{s}px" + (" (全图)" if s == min(W, H) else "")

        crop = base.crop((x, y, x + s, y + s))

        if dry_run:
            # 画真值红框存盘供肉眼核对（带 _gt 后缀，不污染真测用的干净裁剪）
            annotated = crop.copy()
            ImageDraw.Draw(annotated).rectangle(gt_in, outline=(255, 0, 0), width=3)
            view_path = os.path.join(OUT_DIR, f"{img_no}_s{s}_gt.jpg")
            annotated.save(view_path)
            print(f"{tag:>10} | {waldo_ratio:>8.3f}% | {os.path.basename(view_path)}")
            continue

        crop_path = os.path.join(OUT_DIR, f"{img_no}_s{s}.jpg")
        crop.save(crop_path)

        found_n, located_n, errs = 0, 0, 0
        for _ in range(repeats):
            found, box, err = query(client, crop_path)
            if err:
                errs += 1
            else:
                if found:
                    found_n += 1
                if found and box is not None:
                    # box_2d=[ymin,xmin,ymax,xmax] 0-1000 → 裁剪像素 [x1,y1,x2,y2]
                    ymin, xmin, ymax, xmax = box
                    pred = [xmin / 1000 * s, ymin / 1000 * s,
                            xmax / 1000 * s, ymax / 1000 * s]
                    if _overlap(pred, gt_in):
                        located_n += 1
            time.sleep(REQUEST_INTERVAL)

        rate = located_n / repeats * 100
        err_s = f"  ⚠️{errs}err" if errs else ""
        print(f"{tag:>10} | {waldo_ratio:>8.3f}% | {found_n:>2}/{repeats} | "
              f"{located_n:>2}/{repeats} | {rate:>5.0f}%{err_s}")
        rows.append((s, located_n, repeats))

    wall = time.perf_counter() - t0
    print("-" * 72)
    if dry_run:
        print(f"✅ DRY-RUN 完成：{len(deduped)} 张带真值红框的裁剪图已存到")
        print(f"   {OUT_DIR}（文件名 *_gt.jpg，红框=Waldo 真实位置）")
        print("   肉眼确认无误后，去掉 --dry-run 即可跑真测。")
    else:
        # 极限 = 仍过半命中的最大尺寸
        ok = [s for s, loc, r in rows if loc * 2 >= r]
        if ok:
            print(f"✅ 仍能稳定（≥半数）真正定位 Waldo 的最大区域：{max(ok)}px")
        else:
            print("⚠️ 没有任何尺寸达到过半命中——换张图或检查真值。")
        print(f"裁剪图已存：{OUT_DIR}（可肉眼核对）")
    print(f"总耗时 {wall:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
