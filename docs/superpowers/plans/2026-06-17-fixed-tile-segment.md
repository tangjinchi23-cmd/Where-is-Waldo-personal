# 固定尺寸滑窗切片（删除 analyze，segment 直切）实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 删除 analyze 节点，让 segment 成为入口节点，用确定性的「固定尺寸滑窗 + 末块贴边」算法把全图切成 `TILE_SIZE×TILE_SIZE`（默认 256、可调）的 patch。

**架构：** 切图变成纯几何运算（`vision/segment.py::tile_region` 纯函数，可单测、不花 API），第一次 VLM 调用推迟到 detect。流水线：`START → segment(算术) → detect → [路由] → verify/visualize → END`。删除死状态 `grid_rows/grid_cols/grid_size`，保留 `focus_regions` 作输入契约（初始=全图）。

**技术栈：** Python、Pillow（仅几何）、pytest、LangGraph。

**关联：** 已批准 spec `docs/superpowers/specs/2026-06-15-remove-analyze-merge-segment-design.md`（含 2026-06-17 §0 几何细化）。分支 `feat/fixed-tile-segment`。

---

## 文件结构（创建/修改/删除及职责）

| 文件 | 操作 | 职责 |
|------|------|------|
| `vision/segment.py` | 修改 | 新增 `tile_region` + `_axis_starts`（固定尺寸滑窗+贴边，纯几何）；任务 5 删除旧 `segment_region/segment_all_regions` |
| `tests/test_segment_tiling.py` | 创建 | `tile_region` 单测（全覆盖、贴边、重叠、退化、不越界） |
| `agent/nodes/segment.py` | 修改 | 入口节点：按 `TILE_SIZE` 切 `focus_regions`，过滤 `MIN_PATCH_PX`，建 candidates |
| `tests/test_segment_node.py` | 创建 | segment_node 在真图上产出有效 candidates |
| `agent/graph.py` | 修改 | 入口改 `segment`，删 analyze 节点/边/import；`run_agent` 去 `grid_size` |
| `agent/nodes/analyze.py` | 删除 | 节点已废弃 |
| `tests/test_analyze_node.py` | 删除 | 节点已不存在 |
| `agent/nodes/__init__.py` | 修改 | 去掉 `analyze_node` 导出 |
| `prompts.py` | 修改 | 删 `build_analyze_prompt` 及其文档行 |
| `agent/state.py` | 修改 | 删 `grid_rows/grid_cols/grid_size`；`initial_state` 去 `grid_size` 参数 |
| `main.py` | 修改 | `run_agent(image_path)` 去 `grid_size` |
| `vision/__init__.py` | 修改 | 导出改为 `tile_region`（去掉旧两函数） |
| `CLAUDE.md`、`docs/agent_logic.md` | 修改 | 流程/节点/参数表同步 |

---

### 任务 1：`tile_region` 纯函数（TDD）

**文件：**
- 创建：`tests/test_segment_tiling.py`
- 修改：`vision/segment.py`（新增函数，旧函数暂留以免破坏 `vision/__init__.py` 导入）

- [ ] **步骤 1：编写失败的测试**

创建 `tests/test_segment_tiling.py`：

```python
"""tile_region 纯几何单测：固定尺寸滑窗 + 末块贴边对齐。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vision.segment import tile_region

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
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_segment_tiling.py -v`
预期：FAIL，报错 `ImportError: cannot import name 'tile_region'`。

- [ ] **步骤 3：编写最少实现代码**

在 `vision/segment.py` 顶部（`from PIL import Image` 之后、`segment_region` 之前）插入：

```python
def _axis_starts(start: int, length: int, tile: int, stride: int) -> list[int]:
    """一根轴上的切片起点：滑窗 + 末块贴边对齐。

    起点 start, start+stride, ... 保留所有 < (start+length-tile) 者，
    再补一个贴边起点 start+length-tile，保证覆盖到边、每块恰好 tile。
    length <= tile 时退化为单块。
    """
    if length <= tile:
        return [start]
    last = start + length - tile
    starts: list[int] = []
    x = start
    while x < last:
        starts.append(x)
        x += stride
    starts.append(last)
    return starts


def tile_region(
    region: list[int],
    tile_size: int,
    image_size: tuple[int, int],
    overlap: float = 0.15,
) -> list[dict]:
    """把一个区域切成 tile_size×tile_size 的固定尺寸滑窗 patch（末块贴边对齐）。

    Args:
        region: [x, y, w, h]，原图坐标的目标区域。
        tile_size: 每块边长（像素）。
        image_size: (img_width, img_height)，用于边界 clamp。
        overlap: 相邻 patch 重叠比例；stride = round(tile_size*(1-overlap))。

    Returns:
        patch 字典列表，每项 {"bbox": [x, y, w, h], "row": int, "col": int}。
        区域 > tile 时每块宽高恰为 tile_size；区域 <= tile 时单块取整边。
    """
    rx, ry, rw, rh = region
    img_w, img_h = image_size
    stride = max(1, round(tile_size * (1 - overlap)))
    xs = _axis_starts(rx, rw, tile_size, stride)
    ys = _axis_starts(ry, rh, tile_size, stride)

    patches: list[dict] = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            x2 = min(img_w, x + tile_size, rx + rw)
            y2 = min(img_h, y + tile_size, ry + rh)
            patches.append({"bbox": [x, y, x2 - x, y2 - y], "row": row, "col": col})
    return patches
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_segment_tiling.py -v`
预期：7 个测试全部 PASS。

- [ ] **步骤 5：Commit**

```bash
git add vision/segment.py tests/test_segment_tiling.py
git commit -m "feat(vision): add fixed-size sliding-window tile_region with edge-clamp"
```

---

### 任务 2：segment_node 改为入口切片节点

**文件：**
- 修改：`agent/nodes/segment.py`（整体替换）
- 创建：`tests/test_segment_node.py`

- [ ] **步骤 1：编写失败的测试**

创建 `tests/test_segment_node.py`：

```python
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
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_segment_node.py -v`
预期：FAIL（`ImportError: cannot import name 'TILE_SIZE'`，因 segment.py 尚未改）。

- [ ] **步骤 3：编写最少实现代码**

整体替换 `agent/nodes/segment.py`：

```python
"""segment 节点（入口）：把 focus_regions 切成固定尺寸滑窗 patch 列表。

设计原则：
- 切图是确定性几何运算，不调 VLM；第一次 VLM 调用在 detect。
- 固定 TILE_SIZE×TILE_SIZE、末块贴边对齐，detect 永远收到一致尺寸的 patch。
- 跳过宽或高 < MIN_PATCH_PX 的块（仅小图/小区域会触发）。
"""

from PIL import Image

from agent.state import WaldoState
from vision.segment import tile_region

# ── 可调参数 ───────────────────────────────────────────────────────────
TILE_SIZE = 256       # 每块边长（px）；极限测试证明 256 覆盖含 2.jpg 在内的所有可检出图
TILE_OVERLAP = 0.15   # 相邻块重叠比例，防 Waldo 骑在切片边界被切两半
MIN_PATCH_PX = 150    # 低于此尺寸的 patch 跳过


def segment_node(state: WaldoState) -> dict:
    """
    输入：original_image_path, focus_regions（初始=全图）
    输出：candidates（初始化，仅含 patch_bbox 等几何字段）
    """
    img = Image.open(state["original_image_path"])
    image_size = img.size

    candidates = []
    skipped = 0
    for region_idx, region in enumerate(state["focus_regions"]):
        for p in tile_region(region, TILE_SIZE, image_size, TILE_OVERLAP):
            pw, ph = p["bbox"][2], p["bbox"][3]
            if pw < MIN_PATCH_PX or ph < MIN_PATCH_PX:
                skipped += 1
                continue
            candidates.append({
                "patch_bbox": p["bbox"],
                "region_idx": region_idx,
                "row": p["row"],
                "col": p["col"],
                "confidence": 0.0,
                "crop_path": None,
                "verified": False,
            })

    if skipped:
        print(f"[segment] Skipped {skipped} patches smaller than {MIN_PATCH_PX}px")
    print(
        f"[segment] Generated {len(candidates)} patches "
        f"(tile={TILE_SIZE} overlap={TILE_OVERLAP}) from {len(state['focus_regions'])} region(s)"
    )

    return {"candidates": candidates}
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_segment_node.py -v`
预期：2 个测试 PASS。

- [ ] **步骤 5：Commit**

```bash
git add agent/nodes/segment.py tests/test_segment_node.py
git commit -m "feat(segment): make segment the entry tiler using tile_region"
```

---

### 任务 3：从图中移除 analyze 节点（删节点/边/文件/旧测试/prompt）

**文件：**
- 修改：`agent/graph.py:24,32`（删 analyze 节点与边、入口改 segment）
- 修改：`agent/nodes/__init__.py`（去 analyze 导出）
- 删除：`agent/nodes/analyze.py`
- 删除：`tests/test_analyze_node.py`
- 修改：`prompts.py`（删 `build_analyze_prompt`）

> 注：本任务后 `agent/state.py` 仍含 `grid_*` 字段（暂不影响，下一任务清理）。`run_agent` 的 `grid_size` 参数本任务先不动，任务 4 一并处理，保持本次提交聚焦「删 analyze」。

- [ ] **步骤 1：改 `agent/graph.py`**

把 import 块（`from agent.nodes import (...)`）中的 `analyze_node,` 删除，得到：

```python
from agent.nodes import (
    segment_node,
    detect_node,
    verify_node,
    visualize_node,
)
```

在 `build_graph()` 中删除这一行：

```python
    g.add_node("analyze",   analyze_node)
```

把入口与第一条边从：

```python
    g.set_entry_point("analyze")
    g.add_edge("analyze",   "segment")
    g.add_edge("segment",   "detect")
```

改为：

```python
    g.set_entry_point("segment")
    g.add_edge("segment",   "detect")
```

（`add_conditional_edges("detect", route_after_detect, ...)` 及其后均不变。）

- [ ] **步骤 2：改 `agent/nodes/__init__.py`**

整体替换为：

```python
from agent.nodes.segment import segment_node
from agent.nodes.detect import detect_node
from agent.nodes.verify import verify_node
from agent.nodes.visualize import visualize_node

__all__ = [
    "segment_node",
    "detect_node",
    "verify_node",
    "visualize_node",
]
```

- [ ] **步骤 3：删除文件**

```bash
git rm agent/nodes/analyze.py tests/test_analyze_node.py
```

- [ ] **步骤 4：改 `prompts.py`**

删除 `build_analyze_prompt` 函数（`def build_analyze_prompt(img_w: int, img_h: int) -> str:` 整个函数体）以及模块顶部 docstring 中引用它的那一行（`prompts.py:6` 的 `- build_analyze_prompt ...`）。

- [ ] **步骤 5：验证图可构建 + 全测试不引用 analyze**

运行：`python draw_graph.py`
预期：打印 `Saved → agent_graph.png`，无 import 错误。

运行：`pytest -q`
预期：通过（test_analyze_node 已删；其余不依赖 analyze）。

- [ ] **步骤 6：Commit**

```bash
git add agent/graph.py agent/nodes/__init__.py prompts.py agent_graph.png
git commit -m "refactor(graph): remove analyze node, segment becomes entry point"
```

---

### 任务 4：清理死状态 `grid_*` 与 `grid_size` 参数

**文件：**
- 修改：`agent/state.py`（删 3 个字段 + `initial_state` 签名）
- 修改：`agent/graph.py:56-67`（`run_agent` 去 `grid_size`）
- 修改：`main.py:20`（调用去 `grid_size`）

- [ ] **步骤 1：改 `agent/state.py`**

`WaldoState` 删除这三行字段：`grid_size`、`grid_rows`、`grid_cols`。删后 TypedDict 为：

```python
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
```

`initial_state` 改为（去 `grid_size` 参数与赋值、去 `grid_rows/grid_cols`）：

```python
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
```

- [ ] **步骤 2：改 `agent/graph.py` 的 `run_agent`**

替换为：

```python
def run_agent(image_path: str) -> WaldoState:
    """端到端运行 WaldoAgent，返回最终 State。

    Args:
        image_path: 待检测图片路径。

    Returns:
        最终 WaldoState，verified_result 字段为检测结果。
    """
    graph = build_graph()
    state = initial_state(image_path)
    return graph.invoke(state)
```

- [ ] **步骤 3：改 `main.py`**

把 `main.py:20` 的：

```python
    final_state = run_agent(image_path, grid_size=1)
```

改为：

```python
    final_state = run_agent(image_path)
```

- [ ] **步骤 4：验证**

运行：`python -c "from agent.state import initial_state; from agent import run_agent; print('imports ok')"`
预期：打印 `imports ok`，无 `TypeError`（确认没有残留 `grid_size=` 调用）。

运行：`pytest -q`
预期：通过。

- [ ] **步骤 5：Commit**

```bash
git add agent/state.py agent/graph.py main.py
git commit -m "refactor(state): drop dead grid_rows/grid_cols/grid_size fields"
```

---

### 任务 5：删除旧切分函数并更新 `vision` 导出

**文件：**
- 修改：`vision/segment.py`（删 `segment_region`、`segment_all_regions`）
- 修改：`vision/__init__.py`（导出改 `tile_region`）

> 此时已无任何代码引用旧两函数（segment_node 任务 2 已切到 `tile_region`），可安全删除。

- [ ] **步骤 1：删 `vision/segment.py` 旧函数**

删除 `def segment_region(...)` 与 `def segment_all_regions(...)` 两个完整函数（保留 `_axis_starts`、`tile_region`、`get_image_size`）。

- [ ] **步骤 2：改 `vision/__init__.py`**

把：

```python
from vision.segment import segment_region, segment_all_regions, get_image_size
```

及 `__all__` 中的 `"segment_region"`, `"segment_all_regions"` 改为：

```python
from vision.segment import tile_region, get_image_size
```

`__all__` 对应改为含 `"tile_region"`、`"get_image_size"`（保留原有其它项不变）。

- [ ] **步骤 3：验证**

运行：`python -c "import vision; from vision.segment import tile_region; print('ok')"`
预期：打印 `ok`。

运行：`pytest -q`
预期：通过。

- [ ] **步骤 4：Commit**

```bash
git add vision/segment.py vision/__init__.py
git commit -m "refactor(vision): remove obsolete grid segment helpers"
```

---

### 任务 6：端到端验证

**文件：** 无（仅运行）

- [ ] **步骤 1：跑通完整流水线（小 Waldo 难图）**

运行：`python main.py original-images/2.jpg`
预期：依次出现 `[segment] Generated N patches (tile=256 overlap=0.15) ...` → `[detect] ...` → 路由后 `[verify]` 或直接 `[visualize] Result saved → outputs/2_result.jpg`，进程 exit 0。

- [ ] **步骤 2：跑一张大图确认 patch 数在上限内**

运行：`python main.py original-images/1.jpg`
预期：`[segment] Generated` 的 N 远小于 `detect` 的 `MAX_PATCHES_PER_ITER=80`（256px@2048×1251 约 54 块），无 `Sampling` 截断日志。

- [ ] **步骤 3：全测试套件**

运行：`pytest -q`
预期：全部通过，无 analyze/grid_* 相关失败。

---

### 任务 7：文档同步

**文件：**
- 修改：`CLAUDE.md`（流程图、节点职责表、关键可调参数表、State 定义、文件结构）
- 修改：`docs/agent_logic.md`（流程链路）

- [ ] **步骤 1：改 `CLAUDE.md`**

- 流程图去掉 `[analyze]` 节点，入口改 `[segment]`；segment 描述改为「确定性固定尺寸滑窗切片，TILE_SIZE×TILE_SIZE，末块贴边」。
- 节点职责表删 analyze 行；segment 行的 Provider 改「—」、说明改新算法。
- 「关键可调参数」表：删 analyze 的 `THUMBNAIL_MAX`/`ANALYZE_MAX_TOKENS` 行；segment 增加 `TILE_SIZE=256`、`TILE_OVERLAP=0.15`、`MIN_PATCH_PX=150`。
- State 定义同步删 `grid_rows/grid_cols/grid_size`。
- 文件结构去掉 `analyze.py`。

- [ ] **步骤 2：改 `docs/agent_logic.md`**

流程链路同步：入口 segment、去 analyze、切片算法说明。

- [ ] **步骤 3：Commit**

```bash
git add CLAUDE.md docs/agent_logic.md
git commit -m "docs: sync analyze removal and fixed-tile segment design"
```

---

## 自检结果

**规格覆盖度：**
- §0 固定尺寸滑窗 + 贴边 + `TILE_SIZE=256`/`TILE_OVERLAP=0.15` → 任务 1、2 ✅
- §3/§5 删 analyze 节点/文件/prompt、入口改 segment → 任务 3 ✅
- §3/§5 删 `grid_rows/grid_cols/grid_size` + `run_agent`/`main.py` 去 `grid_size` → 任务 4 ✅
- §5 删旧 `segment_region/segment_all_regions` + `vision` 导出 → 任务 5 ✅
- §5 保留 `focus_regions` 输入契约 → 任务 2/4 保留该字段 ✅
- §6 TDD 单测（覆盖/重叠/退化/边界） → 任务 1 ✅；端到端 2.jpg → 任务 6 ✅
- §8 文档同步 → 任务 7 ✅

**占位符扫描：** 无 TODO/待定；每个代码步骤均含完整代码与精确命令。

**类型一致性：** `tile_region(region, tile_size, image_size, overlap)` 在任务 1 定义、任务 2 同签名调用；`segment_node` 输出 candidate 字段（`patch_bbox/region_idx/row/col/confidence/crop_path/verified`）与 detect/verify 现有读取一致；`initial_state(image_path)` 新签名在任务 4 定义、`run_agent` 同步调用。

---

## 执行交接

计划已完成并保存到 `docs/superpowers/plans/2026-06-17-fixed-tile-segment.md`。两种执行方式：

1. **子代理驱动（推荐）** — 每个任务调度一个新子代理，任务间审查，快速迭代。
2. **内联执行** — 在当前会话用 executing-plans 批量执行并设检查点。
