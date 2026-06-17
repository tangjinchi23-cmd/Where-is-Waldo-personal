# 设计规格：删除 analyze 节点，切图职责并入 segment

- 日期：2026-06-15（2026-06-17 几何/参数细化，见 §0）
- 状态：已批准，待写实现计划
- 关联：极限尺寸测试（`scripts/gemini_limit.py` + `outputs/gemini_limit_*.log`）、detect 选型结论（记忆 `detect-bottleneck-findings`）、detect 切 Gemini + 路由（记忆 `detect-gemini-routing`）

---

## 0. 2026-06-17 细化（切片几何与参数，优先于下文相应描述）

§4–§6 的「按目标边长 ceil 均匀分格」在本次细化中**改为固定尺寸滑窗 + 末块贴边对齐**，理由：固定尺寸让 detect 永远收到**恰好 `TILE_SIZE`px** 的切片，与 detect 评测所用 256px patch 尺寸一致，判别行为更可复现。决策如下，**与下文冲突处以本节为准**：

| 决策点 | 06-15 原案 | 06-17 细化（生效） |
|--------|-----------|---------------------|
| 切分方式 | `ceil(dim/target)` 均匀分格（块大小随图浮动） | **固定 `TILE_SIZE` 滑窗 + 末块贴边**（每块恰好 `TILE_SIZE×TILE_SIZE`，末排/列贴 `length-TILE_SIZE` 起点） |
| 目标边长常量名 | `PATCH_TARGET_PX` | **`TILE_SIZE`**，默认 **256**、可调（极限测试证明 256 覆盖含 2.jpg 在内的所有可检出图；384 在小 Waldo 图上会漏，故默认仍取 256，留作可调以便提速时手动上调） |
| overlap | 0.12 | **`TILE_OVERLAP=0.15`**，可调；`stride = round(TILE_SIZE×(1-TILE_OVERLAP))` |
| 切分纯函数 | `grid_segment_region(region, target_px, …)`（ceil 分格） | **`tile_region(region, tile_size, image_size, overlap, min_patch_size)`**（固定尺寸滑窗+贴边） |

**切片算法（每轴）**：起点 `0, stride, 2·stride, …` 保留所有 `< length-TILE_SIZE` 者，再补一个贴边起点 `length-TILE_SIZE` → 保证每块恰好 `TILE_SIZE`、全图无空洞（末排与前排多重叠些）。`length ≤ TILE_SIZE` 时退化为单块（取整边）。末块贴边可能与前一块近重复（差几十 px），属可接受冗余，不去重以保证覆盖。

**与已落地的 detect/路由的衔接**：detect 已切 Gemini、按 `present(has_waldo)` 过滤、`MAX_PATCHES_PER_ITER=80`（256px@2048×1251 约 54 块，仍在上限内）；detect 后已有「单候选直接 visualize」条件路由（见记忆 `detect-gemini-routing`），本次切片改动不影响该路由。

---

## 1. 背景与动机

`gemini_limit.py` 极限测试（2026-06-15）证明：VLM 能"真正定位"（bbox 校验过）Waldo 的最大区域**因图而异，且由 Waldo 的绝对像素大小决定**——7.jpg（Waldo 72×153）全图可定位，1.jpg（50×90）到 768px，13.jpg 512px，10.jpg 384px，2.jpg 仅 256px，19.jpg 任何尺寸全漏（真漏检）。

这戳穿了 analyze 节点的设计前提。analyze 现在的职责是"用 VLM 看缩略图、按视觉复杂度推荐切割行列数"，但：

1. **优化了错误的变量**：该切多大由 Waldo 绝对像素决定，而 Waldo 多大/在哪在切图前不可知（知道就不用找了）。视觉复杂度不是正确的决策轴。
2. **正确解其实是确定性算术**：安全下限须照最难的可检出图走（≈256px），切法 = `ceil(dim/256)`，一行算术即可，VLM 那步只在算术值上下微调，不带真实信号。
3. **VLM 那步是净负担**：analyze 曾栽过推理 token 截断坑（见 `docs/run_issues_2026-06-09.md`），为本应确定性的决策引入一次 API 调用 + 一个失败面。

此外发现 `grid_rows`/`grid_cols` 是**死字段**：只有 analyze 写、state 定义，下游 detect/verify/visualize 无任何读取（早期热力图/calibrate 设计残留）。

## 2. 目标与非目标

**目标**：简化 + 鲁棒性。
- 移除 analyze 的 VLM 调用及其 token 截断失败面。
- 把"按目标尺寸切全图"变成确定性、可复现、可纯单测的算术。
- 清掉死状态（`grid_rows`/`grid_cols`）与失去意义的 `grid_size`。

**非目标（YAGNI）**：
- 不引入"粗到细"迭代回路（先大区域、只细分没把握的）——留作未来选项 B，本次仅保留 `focus_regions` 接口为其留门，不实现。
- 不解决"Waldo 骑在网格边界被切两半"的最坏情况（见 §6 风险），仅记录为后续评测项。

## 3. 决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 核心动机 | 简化 + 鲁棒性 | 走算术化，不走粗到细 |
| 切块目标边长 | **可配常量 `PATCH_TARGET_PX`，默认 256** | 极限测试证明 256 覆盖所有可检出图（含最难的 2.jpg）；比 200 约少 40% patch；做成常量便于后续量化调参 |
| analyze 节点 | **删除，职责并入 segment** | grid_rows/cols 已是死字段，两者职责重叠；删节点 + 删死状态最简 |
| `grid_size` 字段 | **删除**（含 `run_agent`/`main.py` 签名） | 新设计下 grid 由 `PATCH_TARGET_PX` 决定，grid_size 失去意义 |
| `focus_regions` 字段 | **保留** | 作为 segment 的输入契约（初始=全图），为未来粗到细留干净接口，零额外成本 |

## 4. 新架构

### 流程（4 节点线性流水线）
```
旧: START → analyze(VLM) → segment → detect → verify → visualize → END
新: START → segment(算术) → detect → verify → visualize → END
```
入口从 VLM 节点变为纯计算节点；整条流水线第一次 VLM 调用推迟到 detect。

### State（5 字段，瘦身后）
```python
class WaldoState(TypedDict):
    original_image_path: str       # 原图路径
    focus_regions: list            # [[x,y,w,h], ...]，待切区域；初始=全图 [[0,0,W,H]]
    candidates: list               # [{patch_bbox, confidence, verified, ...}, ...]
    verified_result: list | None   # [x,y,w,h] 原图坐标，未找到=None
    iteration: int                 # 恒为 0；detect/verify 用于命名输出文件
```
删去 `grid_rows`、`grid_cols`、`grid_size`。

### 职责边界
| 单元 | 职责 | 输入 → 输出 | 依赖 |
|------|------|------------|------|
| `vision/segment.py::grid_segment_region` | 纯函数：把一个区域按目标边长切成带 overlap 的网格 | `(region, target_px, img_size)` → patch list | 仅 PIL 几何，无 VLM、无 IO |
| `segment_node` | 入口：把 focus_regions 切成 candidates | `focus_regions` → `candidates` | 上面的纯函数 |
| `detect_node` | 判别每个 patch | `candidates` → 带置信度的 `candidates` | VLM |
| `verify_node` | 复核 top-3 | `candidates` → `verified_result` | VLM |
| `visualize_node` | 画框 | `verified_result` → 标注图 | tools/visualize |

### 核心特性
1. **切图与判别彻底解耦**：切图是确定性几何（可纯单测、不花 API），判别是 VLM。
2. **为未来粗到细留干净接口**：保留 `focus_regions`；未来选项 B 只需在 verify 后加节点重写 `focus_regions` 并回边到 segment，不改 segment 本身。

## 5. 组件改动清单

- **`vision/segment.py`**：新增 `grid_segment_region(region, target_px, image_size, min_patch_size, overlap=0.12)`——`cols=max(1, ceil(rw/target_px))`、`rows=max(1, ceil(rh/target_px))`，按 rows×cols 均匀切，每块向右/下扩 `overlap×stride`，边界 clamp。`segment_all_regions` 改为调用它（按 `target_px` 而非 `grid_size`）。删除不再使用的旧 `segment_region(grid_size)`。
- **`agent/nodes/segment.py`**：成为入口节点；新增 `PATCH_TARGET_PX = 256`；对 `focus_regions` 按目标尺寸切；保留 `MIN_PATCH_PX=150` 与 overlap=0.12。
- **`agent/nodes/analyze.py`**：删除整个文件。
- **`agent/graph.py`**：入口改 `segment`；删 analyze 节点、边与 import。
- **`agent/state.py`**：删 `grid_rows`/`grid_cols`/`grid_size`；`initial_state` 同步（去掉 `grid_size` 参数）。
- **`agent/__init__.py` / `agent/nodes/__init__.py`**：`run_agent` 去掉 `grid_size` 参数；去掉 analyze 导出。
- **`main.py`**：`run_agent` 调用去掉 `grid_size`。
- **`prompts.py`**：删 `build_analyze_prompt`。
- **`tests/test_analyze_node.py`**：删除（节点已不存在）。

## 6. 测试

- **TDD 单测** `grid_segment_region`（先写测试）：
  - 2048×1251、target=256 → 8×5 网格；覆盖全图无空洞；overlap 正确（相邻块有重叠、不超图边界）。
  - 非方形图按 W、H 分别算行列。
  - 退化情形：区域 < target_px 时至少产出 1 块。
- **端到端**：`python main.py original-images/2.jpg` 跑通全流程，输出标注图。
- **回归**：确认 detect/verify/visualize 在去掉 grid_* 字段后仍正常。

## 7. 已知风险

**边界切分未覆盖**：极限测试用居中裁剪，未测"Waldo 正好骑在网格边界被切两半"的最坏情况。固定 256 网格 + 12% overlap（≈31px）对 ~30px 小 Waldo 勉强够，对 ~90px 可能不足。本次不解决，列为后续量化评测项——边界命中率值得专门测。

## 8. 文档更新

- `CLAUDE.md`：流程图、节点表、参数表去掉 analyze；加 `PATCH_TARGET_PX`；State 定义同步。
- `docs/agent_logic.md`：流程链路同步。
