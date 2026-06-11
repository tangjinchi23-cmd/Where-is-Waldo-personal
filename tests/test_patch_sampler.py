"""patch_sampler 纯函数单元测试（不调用 API）。"""

import os
import random

from tests.patch_sampler import (
    parse_bbox_file,
    positive_patch_bbox,
    negative_patch_bboxes,
    pessimistic_rank,
)

_BBOX_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "original-images", "bbox"
)


# ---------- parse_bbox_file ----------

def test_parse_bbox_file_line1():
    boxes = parse_bbox_file(_BBOX_FILE)
    # 第 1 行对应 1.jpg
    assert boxes[1] == [690, 520, 740, 610]


def test_parse_bbox_file_line16_is_none():
    boxes = parse_bbox_file(_BBOX_FILE)
    # 第 16 行为空 → 无真值
    assert boxes[16] is None


def test_parse_bbox_file_has_18_valid_entries():
    boxes = parse_bbox_file(_BBOX_FILE)
    valid = [k for k, v in boxes.items() if v is not None]
    assert len(valid) == 18


# ---------- positive_patch_bbox ----------

def test_positive_patch_is_patch_px_square_and_contains_gt():
    gt = [690, 520, 740, 610]  # center (715, 565)
    x, y, w, h = positive_patch_bbox(2048, 1251, gt, patch_px=200)
    assert w == 200 and h == 200
    # patch 完整包含真值框
    assert x <= gt[0] and y <= gt[1]
    assert x + w >= gt[2] and y + h >= gt[3]
    # 在图像内
    assert 0 <= x and 0 <= y and x + w <= 2048 and y + h <= 1251


def test_positive_patch_clamps_at_top_left_corner():
    gt = [10, 10, 40, 40]  # center (25,25) → 居中会越界，应 clamp 到 0
    x, y, w, h = positive_patch_bbox(2048, 1251, gt, patch_px=200)
    assert x == 0 and y == 0 and w == 200 and h == 200


# ---------- negative_patch_bboxes ----------

def test_negatives_count_and_no_overlap_with_gt():
    gt = [690, 520, 740, 610]
    rng = random.Random(42)
    negs = negative_patch_bboxes(2048, 1251, gt, patch_px=200, n=9, rng=rng)
    assert len(negs) == 9
    gx1, gy1, gx2, gy2 = gt
    for x, y, w, h in negs:
        # 与真值框零重叠
        no_overlap = (x >= gx2) or (x + w <= gx1) or (y >= gy2) or (y + h <= gy1)
        assert no_overlap, f"patch {[x, y, w, h]} overlaps gt {gt}"
        # 在图像内
        assert 0 <= x and 0 <= y and x + w <= 2048 and y + h <= 1251


def test_negatives_reproducible_with_same_seed():
    gt = [690, 520, 740, 610]
    a = negative_patch_bboxes(2048, 1251, gt, 200, 9, random.Random(42))
    b = negative_patch_bboxes(2048, 1251, gt, 200, 9, random.Random(42))
    assert a == b


# ---------- pessimistic_rank ----------

def test_rank_positive_clearly_first():
    assert pessimistic_rank(0.9, [0.5, 0.5, 0.1]) == 1


def test_rank_pessimistic_ties_push_positive_last():
    # 2 个负样本与正样本同分 → 正样本排第 3
    assert pessimistic_rank(0.99, [0.99, 0.99, 0.5]) == 3


def test_rank_counts_strictly_greater_negatives():
    # 2 个负样本严格大于 → 正样本排第 3
    assert pessimistic_rank(0.5, [0.9, 0.7, 0.4]) == 3