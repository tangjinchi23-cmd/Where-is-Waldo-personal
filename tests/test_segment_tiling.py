"""tile_region 纯几何单测：固定尺寸滑窗 + 末块贴边对齐。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vision.segment import tile_region, waldo_orig_bbox

TILE = 256
OVERLAP = 0.15
# stride = round(256 * 0.85) = 218


def _starts(patches, axis):
    """从 patch 列表取某轴去重后的升序起点（axis=0→x, 1→y）。"""
    return sorted({p["bbox"][axis] for p in patches})


def test_tiles_are_exact_tile_size_when_region_larger():
    patches = tile_region([0, 0, 1000, 600], TILE, (1000, 600), OVERLAP)
    assert patches, "should produce tiles"
    for p in patches:
        assert p["bbox"][2] == TILE
        assert p["bbox"][3] == TILE


def test_last_tile_is_edge_aligned():
    patches = tile_region([0, 0, 1000, 600], TILE, (1000, 600), OVERLAP)
    xs = _starts(patches, 0)
    ys = _starts(patches, 1)
    assert xs == [0, 218, 436, 654, 744]   # 744 = 1000 - 256，贴边
    assert ys == [0, 218, 344]             # 344 = 600 - 256，贴边


def test_full_coverage_no_gap():
    W, H = 1000, 600
    patches = tile_region([0, 0, W, H], TILE, (W, H), OVERLAP)
    # 最右/下沿恰好覆盖到图边
    assert max(p["bbox"][0] + p["bbox"][2] for p in patches) == W
    assert max(p["bbox"][1] + p["bbox"][3] for p in patches) == H
    assert min(p["bbox"][0] for p in patches) == 0
    assert min(p["bbox"][1] for p in patches) == 0


def test_adjacent_tiles_overlap():
    patches = tile_region([0, 0, 1000, 600], TILE, (1000, 600), OVERLAP)
    xs = _starts(patches, 0)
    # 相邻起点间距 = stride 218 < 256 → 重叠 38px（贴边那对重叠更多）
    assert xs[1] - xs[0] == 218
    assert TILE - (xs[1] - xs[0]) == 38


def test_small_region_degrades_to_single_tile():
    patches = tile_region([0, 0, 200, 200], TILE, (1000, 600), OVERLAP)
    assert len(patches) == 1
    assert patches[0]["bbox"] == [0, 0, 200, 200]


def test_non_square_uses_axes_independently():
    patches = tile_region([0, 0, 800, 300], TILE, (800, 300), OVERLAP)
    xs = _starts(patches, 0)
    ys = _starts(patches, 1)
    assert xs == [0, 218, 436, 544]   # 544 = 800 - 256
    assert ys == [0, 44]              # 44  = 300 - 256


def test_tiles_never_exceed_image_bounds():
    W, H = 1000, 600
    patches = tile_region([0, 0, W, H], TILE, (W, H), OVERLAP)
    for p in patches:
        x, y, w, h = p["bbox"]
        assert 0 <= x and x + w <= W
        assert 0 <= y and y + h <= H


def test_region_exceeding_image_is_clamped_to_fixed_size():
    # region 越界（rx+rw=1200 > img_w=1000）；clamp 后末块仍应是 TILE，不出界
    patches = tile_region([700, 0, 500, 600], TILE, (1000, 600), OVERLAP)
    assert patches
    for p in patches:
        x, y, w, h = p["bbox"]
        assert w == TILE and h == TILE        # 仍是固定尺寸
        assert x + w <= 1000 and y + h <= 600  # 不出界
    xs = sorted({p["bbox"][0] for p in patches})
    assert xs[-1] == 1000 - TILE               # 末块贴图右边


# ── waldo_orig_bbox：patch 内 bbox → 原图坐标 ──────────────────────────

def test_waldo_orig_bbox_offsets_sub_bbox_by_patch_origin():
    # patch 左上角 (100,200)，Waldo 在 patch 内 (10,20)，尺寸 30×40
    assert waldo_orig_bbox([100, 200, 256, 256], [10, 20, 30, 40]) == [110, 220, 30, 40]


def test_waldo_orig_bbox_falls_back_to_whole_patch_when_no_sub_bbox():
    # 无 patch 内 bbox（None）→ 退化为整块 patch
    assert waldo_orig_bbox([100, 200, 256, 256], None) == [100, 200, 256, 256]


def test_waldo_orig_bbox_treats_empty_list_as_no_sub_bbox():
    # 空列表（falsy）同样退化为整块 patch
    assert waldo_orig_bbox([100, 200, 256, 256], []) == [100, 200, 256, 256]
