"""calibrate 节点：聚焦高置信度区域，向外扩展为 ~400px 的搜索窗口。

设计原则：
- 不直接用 patch_bbox 作为 focus_region（会导致区域指数级缩小）
- 以检测命中的 patch 中心为圆心，向外扩展到 EXPAND_TO 像素的正方形
- 扩展后的区域在 segment 中以 grid_size=2 切分 → 每 patch ≈ 200×200px
"""

from PIL import Image

from agent.state import WaldoState

EXPAND_TO = 400        # 扩展后的目标区域尺寸（像素）
CALIBRATE_GRID_SIZE = 2  # calibrate 后 segment 使用的固定 grid_size（2×2=4 patch/region）
TOP_K_REGIONS = 3      # 取置信度前 K 的 patch 作为新焦点


def calibrate_node(state: WaldoState) -> dict:
    """
    输入：candidates, original_image_path
    输出：focus_regions（扩展后的搜索窗口），grid_size（固定为 CALIBRATE_GRID_SIZE）

    策略：
    1. 优先从 has_waldo=True 的候选中选，否则取全部候选
    2. 按置信度降序取 top-K
    3. 以每个 patch 中心向外扩展到 EXPAND_TO×EXPAND_TO 区域（clamp 至图像边界）
    4. 若 candidates 为空，保持当前 focus_regions 不变
    """
    img = Image.open(state["original_image_path"])
    img_w, img_h = img.size

    detected = [c for c in state["candidates"] if c.get("has_waldo", False)]
    pool = detected if detected else state["candidates"]
    top_candidates = sorted(pool, key=lambda c: c.get("confidence", 0.0), reverse=True)[:TOP_K_REGIONS]

    if not top_candidates:
        print("[calibrate] No candidates found, keeping current focus_regions")
        return {
            "focus_regions": state["focus_regions"],
            "grid_size": CALIBRATE_GRID_SIZE,
            "region_grid_sizes": {},
        }

    new_focus_regions = []
    for cand in top_candidates:
        region = _expand_around_patch(cand["patch_bbox"], img_w, img_h, EXPAND_TO)
        new_focus_regions.append(region)
        print(f"[calibrate] patch_bbox={cand['patch_bbox']} → expanded={region} (conf={cand.get('confidence', 0):.2f})")

    return {
        "focus_regions": new_focus_regions,
        "grid_size": CALIBRATE_GRID_SIZE,
        "region_grid_sizes": {},
    }


def _expand_around_patch(patch_bbox: list[int], img_w: int, img_h: int, expand_to: int) -> list[int]:
    """以 patch 中心为圆心，向外扩展到 expand_to×expand_to 正方形，clamp 至图像边界。"""
    px, py, pw, ph = patch_bbox
    cx = px + pw // 2
    cy = py + ph // 2
    half = expand_to // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img_w, x1 + expand_to)
    y2 = min(img_h, y1 + expand_to)

    # 若 clamp 导致区域偏移，尝试反向补足
    if x2 - x1 < expand_to:
        x1 = max(0, x2 - expand_to)
    if y2 - y1 < expand_to:
        y1 = max(0, y2 - expand_to)

    return [x1, y1, x2 - x1, y2 - y1]
