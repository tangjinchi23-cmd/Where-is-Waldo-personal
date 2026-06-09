"""analyze 节点：VLM 根据图片复杂度推荐切割行列数，生成覆盖全图的 focus_regions。

设计原则：每个格子约 200×200px，是 VLM 可靠识别 Waldo 的最小单元。
"""

import json
import os
import re

from PIL import Image, ImageDraw, ImageFont

from agent.state import WaldoState
from llm.vlm_client import get_vlm_client
from prompts import build_analyze_prompt

VLM_PROVIDER = "gpt4o"
THUMBNAIL_MAX = 900
ANALYZE_MAX_TOKENS = 64
THUMB_DIR = "outputs/thumbs"


def analyze_node(state: WaldoState) -> dict:
    """
    输入：original_image_path
    输出：focus_regions（N×M 个格子），grid_rows，grid_cols，region_complexity

    流程：
    1. 生成带网格线的缩略图（以 suggest_rows×suggest_cols 为初始格子提示）
    2. 发给 VLM，让它根据复杂度推荐 {"rows": N, "cols": M}
    3. 按推荐行列数将全图均匀切分为 N×M 个 focus_regions
    """
    image_path = state["original_image_path"]
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    prompt = build_analyze_prompt(W, H)

    # 从 prompt 中提取建议行列数（用于缩略图绘制）
    target = 200
    suggest_cols = max(2, round(W / target))
    suggest_rows = max(2, round(H / target))

    # 1. 生成标注缩略图
    thumb = _make_annotated_thumbnail(img, suggest_rows, suggest_cols, THUMBNAIL_MAX)
    os.makedirs(THUMB_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    thumb_path = os.path.join(THUMB_DIR, f"{base}_analyze_grid.jpg")
    thumb.save(thumb_path)

    # 2. 调用 VLM
    vlm = get_vlm_client(VLM_PROVIDER)
    raw = vlm.call(thumb_path, prompt, max_tokens=ANALYZE_MAX_TOKENS)
    rows, cols = _parse_grid_dims(raw, suggest_rows, suggest_cols)
    print(f"[analyze] VLM recommended grid: {rows}×{cols}  (image: {W}×{H})")

    # 3. 均匀切分全图为 rows×cols 个格子
    focus_regions = []
    for r in range(rows):
        for c in range(cols):
            x1 = int(round(c * W / cols))
            y1 = int(round(r * H / rows))
            x2 = min(W, int(round((c + 1) * W / cols)))
            y2 = min(H, int(round((r + 1) * H / rows)))
            focus_regions.append([x1, y1, x2 - x1, y2 - y1])

    return {
        "focus_regions": focus_regions,
        "grid_rows": rows,
        "grid_cols": cols,
        "region_complexity": [0.5] * len(focus_regions),
        "region_grid_sizes": {},
    }


# ── 内部工具函数 ──────────────────────────────────────────────────────

def _make_annotated_thumbnail(
    img: Image.Image,
    rows: int,
    cols: int,
    max_size: int,
) -> Image.Image:
    """生成带网格线的缩略图（用建议的行列数）。"""
    W, H = img.size
    scale = min(1.0, max_size / max(W, H))
    tw, th = int(W * scale), int(H * scale)
    thumb = img.resize((tw, th), Image.LANCZOS)
    draw = ImageDraw.Draw(thumb)

    for i in range(1, cols):
        x = int(i * tw / cols)
        draw.line([(x, 0), (x, th)], fill="yellow", width=2)
    for j in range(1, rows):
        y = int(j * th / rows)
        draw.line([(0, y), (tw, y)], fill="yellow", width=2)

    return thumb


def _parse_grid_dims(text: str, fallback_rows: int, fallback_cols: int) -> tuple[int, int]:
    """从 VLM 响应中解析 {"rows": N, "cols": M}，失败返回 fallback。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        rows = int(data.get("rows", fallback_rows))
        cols = int(data.get("cols", fallback_cols))
        rows = max(1, rows)
        cols = max(1, cols)
        return rows, cols
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # fallback：从文本中找两个整数
    nums = re.findall(r"\b(\d+)\b", text)
    if len(nums) >= 2:
        try:
            return max(1, int(nums[0])), max(1, int(nums[1]))
        except ValueError:
            pass

    print(f"[analyze] Failed to parse grid dims from: {text!r:.80}, using fallback {fallback_rows}×{fallback_cols}")
    return fallback_rows, fallback_cols
