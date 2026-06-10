"""segment 节点：将 focus_regions 切分为 patch 列表。

设计原则：
- focus_regions 已是 analyze 切出的 N×M 个合理大小格子，grid_size=1 时每格直接作为一个 patch
- 始终跳过宽或高 < MIN_PATCH_PX 的区域，防止过小 patch 传入 VLM
"""

from PIL import Image

from agent.state import WaldoState
from vision.segment import segment_all_regions

MIN_PATCH_PX = 150   # 低于此尺寸的 patch 跳过


def segment_node(state: WaldoState) -> dict:
    """
    输入：original_image_path, focus_regions, grid_size
    输出：candidates（初始化，仅含 patch_bbox）

    注意：每次进入此节点都重置 candidates（新一轮检测）。
    """
    img = Image.open(state["original_image_path"])
    image_size = img.size

    patches = segment_all_regions(
        focus_regions=state["focus_regions"],
        default_grid_size=state["grid_size"],
        image_size=image_size,
        min_patch_size=MIN_PATCH_PX,
        overlap=0.12,
    )

    candidates = []
    skipped = 0
    for p in patches:
        pw, ph = p["bbox"][2], p["bbox"][3]
        if pw < MIN_PATCH_PX or ph < MIN_PATCH_PX:
            skipped += 1
            continue
        candidates.append({
            "patch_bbox": p["bbox"],
            "region_idx": p["region_idx"],
            "row": p["row"],
            "col": p["col"],
            "confidence": 0.0,
            "crop_path": None,
            "verified": False,
        })

    if skipped:
        print(f"[segment] Skipped {skipped} patches smaller than {MIN_PATCH_PX}px")
    print(f"[segment] Generated {len(candidates)} patches from {len(state['focus_regions'])} regions (grid_size={state['grid_size']})")

    return {"candidates": candidates}
