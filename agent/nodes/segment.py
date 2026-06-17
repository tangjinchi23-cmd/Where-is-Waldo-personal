"""segment 节点（入口）：把 focus_regions 切成固定尺寸滑窗 patch 列表。

设计原则：
- 切图是确定性几何运算，不调 VLM；第一次 VLM 调用在 detect。
- 固定 TILE_SIZE×TILE_SIZE、末块贴边对齐，detect 永远收到一致尺寸的 patch。
- 跳过宽或高 < MIN_PATCH_PX 的块（仅小图/小区域会触发）。
"""

from PIL import Image

from agent.state import WaldoState
from vision.segment import tile_region

# ── 可调参数 ───────────────────────────────────────────────────────────
TILE_SIZE = 256       # 每块边长（px）；极限测试证明 256 覆盖含 2.jpg 在内的所有可检出图
TILE_OVERLAP = 0.15   # 相邻块重叠比例，防 Waldo 骑在切片边界被切两半
MIN_PATCH_PX = 150    # 低于此尺寸的 patch 跳过


def segment_node(state: WaldoState) -> dict:
    """
    输入：original_image_path, focus_regions（初始=全图）
    输出：candidates（初始化，仅含 patch_bbox 等几何字段）
    """
    img = Image.open(state["original_image_path"])
    image_size = img.size

    candidates = []
    skipped = 0
    for region_idx, region in enumerate(state["focus_regions"]):
        for p in tile_region(region, TILE_SIZE, image_size, TILE_OVERLAP):
            pw, ph = p["bbox"][2], p["bbox"][3]
            if pw < MIN_PATCH_PX or ph < MIN_PATCH_PX:
                skipped += 1
                continue
            candidates.append({
                "patch_bbox": p["bbox"],
                "region_idx": region_idx,
                "row": p["row"],
                "col": p["col"],
                "confidence": 0.0,
                "crop_path": None,
                "verified": False,
            })

    if skipped:
        print(f"[segment] Skipped {skipped} patches smaller than {MIN_PATCH_PX}px")
    print(
        f"[segment] Generated {len(candidates)} patches "
        f"(tile={TILE_SIZE} overlap={TILE_OVERLAP}) from {len(state['focus_regions'])} region(s)"
    )

    return {"candidates": candidates}
