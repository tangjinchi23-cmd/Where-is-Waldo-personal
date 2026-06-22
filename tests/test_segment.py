"""核心逻辑测试：切片几何（tile_region）+ segment 节点（无 API）。"""

import os

from PIL import Image

from vision.segment import tile_region
from agent.nodes.segment import segment_node, TILE_SIZE, MIN_PATCH_PX

_TEST_IMAGE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "original-images", "2.jpg"
)


# ── tile_region 纯几何 ──────────────────────────────────────────────

def test_tile_region_covers_full_region_no_gaps():
    w, h = 1000, 600
    tiles = tile_region([0, 0, w, h], 256, (w, h), overlap=0.15)
    # 末块贴边对齐 → 覆盖到右下角
    assert max(b["bbox"][0] + b["bbox"][2] for b in tiles) == w
    assert max(b["bbox"][1] + b["bbox"][3] for b in tiles) == h
    # 起点从 0 开始
    assert min(b["bbox"][0] for b in tiles) == 0
    assert min(b["bbox"][1] for b in tiles) == 0


def test_tile_region_each_tile_is_tile_size_when_region_larger():
    tiles = tile_region([0, 0, 1000, 600], 256, (1000, 600), overlap=0.15)
    for b in tiles:
        assert b["bbox"][2] == 256 and b["bbox"][3] == 256


def test_tile_region_degenerates_to_single_block_when_small():
    tiles = tile_region([0, 0, 200, 150], 256, (200, 150), overlap=0.15)
    assert len(tiles) == 1
    assert tiles[0]["bbox"] == [0, 0, 200, 150]


# ── segment 节点（真图，但不调 VLM）────────────────────────────────

def _state():
    return {
        "original_image_path": _TEST_IMAGE,
        "candidates": [],
        "verified_result": None,
    }


def test_segment_node_produces_valid_candidates():
    out = segment_node(_state())
    cands = out["candidates"]
    assert len(cands) > 0
    for c in cands:
        assert "patch_bbox" in c
        x, y, w, h = c["patch_bbox"]
        assert w >= MIN_PATCH_PX and h >= MIN_PATCH_PX


def test_segment_node_covers_full_image():
    w, h = Image.open(_TEST_IMAGE).size
    boxes = [c["patch_bbox"] for c in segment_node(_state())["candidates"]]
    assert max(b[0] + b[2] for b in boxes) == w
    assert max(b[1] + b[3] for b in boxes) == h
