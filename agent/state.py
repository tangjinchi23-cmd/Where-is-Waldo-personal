from typing import TypedDict


class WaldoState(TypedDict):
    # ── 输入 ──────────────────────────────────────────
    original_image_path: str        # 原图路径

    # ── 搜索参数 ──────────────────────────────────────
    grid_size: int                  # 切分粒度（线性流水线下恒为 1）
    focus_regions: list             # [[x, y, w, h], ...]，当前重点搜索区域（原图坐标）

    # ── analyze 输出 ──────────────────────────────────
    grid_rows: int                  # VLM 推荐的行数（每格约 200px）
    grid_cols: int                  # VLM 推荐的列数（每格约 200px）

    # ── detect / verify 输出 ─────────────────────────
    candidates: list                # [{patch_bbox, confidence, crop_path, verified}, ...]

    # ── 最终结果 ──────────────────────────────────────
    verified_result: list | None    # [x, y, w, h]（原图坐标），未找到则 None

    # ── 运行标识 ──────────────────────────────────────
    iteration: int                  # 线性流水线下恒为 0；detect/verify 用于命名输出文件


def initial_state(image_path: str, grid_size: int = 1) -> WaldoState:
    """构造初始 State，全图作为唯一 focus_region。analyze 节点会替换 focus_regions。"""
    from PIL import Image
    img = Image.open(image_path)
    w, h = img.size
    return WaldoState(
        original_image_path=image_path,
        grid_size=grid_size,
        focus_regions=[[0, 0, w, h]],
        grid_rows=0,
        grid_cols=0,
        candidates=[],
        verified_result=None,
        iteration=0,
    )
