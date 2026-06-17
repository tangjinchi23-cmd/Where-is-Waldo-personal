"""segment_node：在真图上把 focus_regions 切成有效 candidates。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PIL import Image

from agent.nodes.segment import segment_node, TILE_SIZE, MIN_PATCH_PX

_TEST_IMAGE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "original-images", "2.jpg"
)


def _state():
    w, h = Image.open(_TEST_IMAGE).size
    return {
        "original_image_path": _TEST_IMAGE,
        "focus_regions": [[0, 0, w, h]],
        "candidates": [],
        "verified_result": None,
        "iteration": 0,
    }


def test_segment_node_produces_candidates():
    out = segment_node(_state())
    cands = out["candidates"]
    assert len(cands) > 0
    for c in cands:
        assert "patch_bbox" in c
        x, y, w, h = c["patch_bbox"]
        assert w >= MIN_PATCH_PX and h >= MIN_PATCH_PX


def test_segment_node_covers_full_image():
    w, h = Image.open(_TEST_IMAGE).size
    out = segment_node(_state())
    boxes = [c["patch_bbox"] for c in out["candidates"]]
    assert max(b[0] + b[2] for b in boxes) == w
    assert max(b[1] + b[3] for b in boxes) == h
