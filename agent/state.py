from typing import TypedDict


class WaldoState(TypedDict):
    # ── 输入 ──────────────────────────────────────────
    original_image_path: str        # 原图路径

    # ── 搜索区域（segment 输入契约，初始=全图）────────
    focus_regions: list             # [[x, y, w, h], ...]

    # ── detect / verify 输出 ─────────────────────────
    candidates: list                # [{patch_bbox, confidence, verified, ...}, ...]

    # ── 最终结果 ──────────────────────────────────────
    verified_result: list | None    # [x, y, w, h]（原图坐标），未找到则 None

    # ── 运行标识 ──────────────────────────────────────
    iteration: int                  # 恒为 0；detect/verify 用于命名输出文件


def initial_state(image_path: str) -> WaldoState:
    """构造初始 State，全图作为唯一 focus_region。"""
    from PIL import Image
    img = Image.open(image_path)
    w, h = img.size
    return WaldoState(
        original_image_path=image_path,
        focus_regions=[[0, 0, w, h]],
        candidates=[],
        verified_result=None,
        iteration=0,
    )
