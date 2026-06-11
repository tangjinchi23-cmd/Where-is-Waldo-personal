"""评测用纯函数：从 ground truth 生成正/负 patch 框 + 悲观平局排名。

无 API、无图像 I/O（除读取 bbox 真值文件）；可独立单元测试。
坐标约定：
  - 真值框 gt = [x1, y1, x2, y2]（左上角 + 右下角）
  - patch 框 = [x, y, w, h]（左上角 + 宽高），与 crop_to_pil 一致
"""

from __future__ import annotations

import random


def parse_bbox_file(path: str) -> dict[int, list[int] | None]:
    """读取 bbox 真值文件，第 N 行 → N.jpg 的 [x1,y1,x2,y2]；空行 → None。

    返回 {行号(从1起): [x1,y1,x2,y2] | None}。
    """
    result: dict[int, list[int] | None] = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip().strip("[]")
            if not s:
                result[i] = None
                continue
            parts = [int(p) for p in s.replace(" ", "").split(",") if p != ""]
            result[i] = parts if len(parts) == 4 else None
    return result


def positive_patch_bbox(
    img_w: int, img_h: int, gt: list[int], patch_px: int = 200
) -> list[int]:
    """以真值框中心裁一个 patch_px×patch_px 窗口，越界则 clamp 回界内。"""
    cx = (gt[0] + gt[2]) // 2
    cy = (gt[1] + gt[3]) // 2
    x = cx - patch_px // 2
    y = cy - patch_px // 2
    x = max(0, min(x, img_w - patch_px))
    y = max(0, min(y, img_h - patch_px))
    return [x, y, patch_px, patch_px]


def _overlaps(patch: list[int], gt: list[int]) -> bool:
    """patch=[x,y,w,h] 与 gt=[x1,y1,x2,y2] 是否有非零交集。"""
    px1, py1, pw, ph = patch
    px2, py2 = px1 + pw, py1 + ph
    gx1, gy1, gx2, gy2 = gt
    return px1 < gx2 and px2 > gx1 and py1 < gy2 and py2 > gy1


def negative_patch_bboxes(
    img_w: int,
    img_h: int,
    gt: list[int],
    patch_px: int,
    n: int,
    rng: random.Random,
    max_attempts: int = 10000,
) -> list[list[int]]:
    """随机采样 n 个 patch_px 窗口，均与真值框零重叠且在图像内。

    用传入的 rng 保证可复现。采样 max_attempts 仍凑不齐则抛错。
    """
    if img_w < patch_px or img_h < patch_px:
        raise ValueError(f"image {img_w}x{img_h} smaller than patch {patch_px}")
    negs: list[list[int]] = []
    attempts = 0
    while len(negs) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"failed to sample {n} negatives after {max_attempts} tries")
        x = rng.randint(0, img_w - patch_px)
        y = rng.randint(0, img_h - patch_px)
        patch = [x, y, patch_px, patch_px]
        if not _overlaps(patch, gt):
            negs.append(patch)
    return negs


def pessimistic_rank(pos_conf: float, neg_confs: list[float]) -> int:
    """正样本在「正+负」集合中的排名，平局时正样本排在并列者之后（悲观）。

    rank = 1 + #(neg > pos) + #(neg == pos)；命中 top-k 即 rank <= k。
    """
    greater = sum(1 for c in neg_confs if c > pos_conf)
    equal = sum(1 for c in neg_confs if c == pos_conf)
    return 1 + greater + equal