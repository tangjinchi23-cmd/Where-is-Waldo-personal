"""verify 节点：对 top-K 候选裁出带 padding 的原图区域，VLM 二次确认是否是 Waldo。"""

import os

from PIL import Image

from agent.state import WaldoState
from vision.image_utils import crop_to_pil, save_patch
from vision.segment import waldo_orig_bbox
from llm.vlm_client import get_vlm_client

# ── 可调参数 ───────────────────────────────────────────────────────────
TOP_K = 3                   # 只对置信度前 K 个候选做二次验证
PADDING_RATIO = 0.3         # 向外扩展 bbox 的比例（相对 bbox 宽/高）
MIN_VERIFY_SIZE = 120       # 发给 VLM 的 verify 图最小边长（像素）
VLM_PROVIDER = "gpt4o"
VERIFY_DIR = "outputs/verify"


def verify_node(state: WaldoState) -> dict:
    """
    输入：original_image_path, candidates（已按 confidence 降序排列）
    输出：candidates（更新 verified / verify_confidence / orig_bbox / verify_crop_path）
          verified_result（取 is_waldo=True 中 verify_confidence 最高的 bbox）

    流程：
    1. 取 top-K candidates，将 patch 内 bbox → 原图坐标
    2. 向外扩展 bbox（加 padding），确保 VLM 有足够上下文
    3. 裁出原图区域发给 VLM 二次确认
    4. 收集所有通过验证的候选，取最高置信度作为 verified_result
    """
    vlm = get_vlm_client(VLM_PROVIDER)
    image_path = state["original_image_path"]
    iteration = state["iteration"]
    os.makedirs(VERIFY_DIR, exist_ok=True)

    img = Image.open(image_path)
    img_w, img_h = img.size

    candidates = list(state["candidates"])
    passed: list[tuple[float, list[int]]] = []   # (verify_confidence, orig_bbox)

    for i, cand in enumerate(candidates[:TOP_K]):
        # 1. patch 内 bbox → 原图坐标（无精确子 bbox 时退化为整块 patch）
        orig_bbox = waldo_orig_bbox(cand["patch_bbox"], cand.get("waldo_bbox_in_patch"))

        # 2. 扩展 bbox，确保最小尺寸
        padded_bbox = _expand_bbox(orig_bbox, img_w, img_h, PADDING_RATIO, MIN_VERIFY_SIZE)

        # 3. 裁图并发给 VLM
        crop_img = crop_to_pil(image_path, padded_bbox)
        verify_path = os.path.join(VERIFY_DIR, f"iter{iteration}_verify{i}.jpg")
        save_patch(crop_img, verify_path)

        result = vlm.verify(verify_path)
        print(
            f"[verify] iter={iteration} cand={i} "
            f"is_waldo={result.is_waldo} conf={result.confidence:.2f} "
            f"| detect_conf={cand['confidence']:.2f}"
        )

        candidates[i] = {
            **cand,
            "verified": result.is_waldo,
            "verify_confidence": result.confidence,
            "orig_bbox": orig_bbox,        # 精确 bbox（未 padding）
            "verify_crop_path": verify_path,
        }

        if result.is_waldo:
            passed.append((result.confidence, orig_bbox))

    # 4. 取 verify_confidence 最高的通过项作为 verified_result
    verified_result = None
    if passed:
        passed.sort(key=lambda t: t[0], reverse=True)
        verified_result = passed[0][1]
        print(f"[verify] Found Waldo! bbox={verified_result}, conf={passed[0][0]:.2f}")
    else:
        print(f"[verify] No candidate passed verification (iter={iteration})")

    return {
        "candidates": candidates,
        "verified_result": verified_result,
    }


# ── 内部工具 ───────────────────────────────────────────────────────────

def _expand_bbox(
    bbox: list[int],
    img_w: int,
    img_h: int,
    ratio: float,
    min_size: int,
) -> list[int]:
    """向外扩展 bbox，给 VLM 更多上下文，并保证最小边长。"""
    x, y, w, h = bbox
    pad_x = int(w * ratio)
    pad_y = int(h * ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    # 保证最小边长
    if x2 - x1 < min_size:
        extra = (min_size - (x2 - x1)) // 2
        x1 = max(0, x1 - extra)
        x2 = min(img_w, x2 + extra)
    if y2 - y1 < min_size:
        extra = (min_size - (y2 - y1)) // 2
        y1 = max(0, y1 - extra)
        y2 = min(img_h, y2 + extra)

    return [x1, y1, x2 - x1, y2 - y1]
