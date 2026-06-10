"""detect 节点：并发调用 VLM，对每个 patch 判断是否含 Waldo。"""

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent.state import WaldoState
from vision.image_utils import crop_to_pil, save_patch
from llm.vlm_client import get_vlm_client, BaseVLMClient, DetectResult

# ── 可调参数 ───────────────────────────────────────────────────────────
DETECT_CONFIDENCE_THRESHOLD = 0.15  # 低于此值的 patch 直接丢弃
MAX_CONCURRENT = 1                   # 串行调用：50 req/min 限制下最安全
MAX_PATCHES_PER_ITER = 80            # 每轮 patch 硬性上限，超出则随机截断
MIN_DETECT_PATCH_PX = 150            # 低于此尺寸的 patch 跳过（VLM 无法可靠识别）
VLM_PROVIDER = "gpt4o"
PATCH_DIR = "outputs/patches"

# 限流重试
MAX_RETRIES = 4
RETRY_BASE_WAIT = 15                 # 首次等待秒数，指数退避：15→30→60→120


def detect_node(state: WaldoState) -> dict:
    """
    输入：original_image_path, candidates（segment 节点生成，含 patch_bbox）
    输出：candidates（更新 confidence / has_waldo / waldo_bbox_in_patch / crop_path）

    流程：
    1. 把所有 patch 从原图裁剪并保存（串行，I/O 快）
    2. 并发调用 VLM（ThreadPoolExecutor）
    3. 合并结果，过滤低置信度，按置信度降序排列
    """
    vlm = get_vlm_client(VLM_PROVIDER)
    image_path = state["original_image_path"]
    iteration = state["iteration"]
    os.makedirs(PATCH_DIR, exist_ok=True)

    # 1. 裁剪并保存所有 patch（超出上限时随机采样，避免系统性漏检右下角）
    candidates = list(state["candidates"])
    if len(candidates) > MAX_PATCHES_PER_ITER:
        print(f"[detect] Sampling {MAX_PATCHES_PER_ITER}/{len(candidates)} patches randomly")
        candidates = random.sample(candidates, MAX_PATCHES_PER_ITER)

    tasks: list[tuple[int, dict, str]] = []
    skipped = 0
    for i, cand in enumerate(candidates):
        pw, ph = cand["patch_bbox"][2], cand["patch_bbox"][3]
        if pw < MIN_DETECT_PATCH_PX or ph < MIN_DETECT_PATCH_PX:
            skipped += 1
            continue
        crop_path = os.path.join(PATCH_DIR, f"iter{iteration}_patch{i}.jpg")
        patch_img = crop_to_pil(image_path, cand["patch_bbox"])
        save_patch(patch_img, crop_path)
        tasks.append((i, cand, crop_path))
    if skipped:
        print(f"[detect] Skipped {skipped} patches smaller than {MIN_DETECT_PATCH_PX}px")

    print(f"[detect] iter={iteration}, patches={len(tasks)}, workers={MAX_CONCURRENT}")

    # 2. 并发调用 VLM
    results: list[tuple[dict, str, DetectResult] | None] = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        future_map = {
            pool.submit(_call_vlm, vlm, crop_path): (i, cand, crop_path)
            for i, cand, crop_path in tasks
        }
        for future in as_completed(future_map):
            i, cand, crop_path = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[detect] patch {i} error: {exc}")
                result = DetectResult(has_waldo=False, confidence=0.0)
            results[i] = (cand, crop_path, result)

    # 3. 合并、过滤、排序
    updated = []
    for entry in results:
        if entry is None:
            continue
        cand, crop_path, result = entry
        updated.append({
            **cand,
            "crop_path": crop_path,
            "confidence": result.confidence,
            "has_waldo": result.has_waldo,
            "waldo_bbox_in_patch": _validated_bbox(result.bbox, cand["patch_bbox"]),
        })

    before = len(updated)
    updated = [c for c in updated if c["confidence"] >= DETECT_CONFIDENCE_THRESHOLD]
    updated.sort(key=lambda c: c["confidence"], reverse=True)
    print(f"[detect] {len(updated)}/{before} patches passed threshold={DETECT_CONFIDENCE_THRESHOLD}")

    return {"candidates": updated}


# ── 内部工具 ───────────────────────────────────────────────────────────

def _call_vlm(vlm: BaseVLMClient, crop_path: str) -> DetectResult:
    """单次 VLM 调用，带指数退避重试（处理 429 限流）。"""
    for attempt in range(MAX_RETRIES):
        try:
            return vlm.detect(crop_path)
        except Exception as exc:
            is_rate_limit = "429" in str(exc) or "rate_limit" in str(exc).lower()
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_WAIT * (2 ** attempt)
                print(f"[detect] Rate limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise


def _validated_bbox(bbox: list | None, patch_bbox: list[int]) -> list[int] | None:
    """将 VLM 返回的 patch 内 bbox 裁剪到合法范围，防止越界。"""
    if not bbox or len(bbox) != 4:
        return None
    _, _, pw, ph = patch_bbox
    x, y, w, h = [int(v) for v in bbox]
    x = max(0, min(x, pw - 1))
    y = max(0, min(y, ph - 1))
    w = max(1, min(w, pw - x))
    h = max(1, min(h, ph - y))
    return [x, y, w, h]
