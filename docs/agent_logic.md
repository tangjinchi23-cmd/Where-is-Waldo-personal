# WhereisWaldoAgent 逻辑链路文档

> 2026-06-17 重构后：analyze 节点已删除，segment 成为入口做确定性切片；detect 改用 Gemini；detect 后加单候选跳过 verify 的条件路由。

## 1. 整体流程

确定性切片 + 条件路由流水线（第一次 VLM 调用推迟到 detect）：

```
START
  │
  ▼
[segment]  （入口，无 VLM）─────────────────────────────────────────────────
  │  对每个 focus_region（初始=全图）做固定尺寸滑窗切片：
  │  TILE_SIZE=256、末块贴边对齐、TILE_OVERLAP=0.15；跳过 < 150px 的块
  │  输出：candidates（仅含 patch_bbox 等几何字段）
  ▼
[detect]   ───────────────────────────────────────────────────────────────
  │  VLM(gemini-3.5-flash) 对每个 patch 判断是否含 Waldo
  │  按 present(has_waldo) 二元信号过滤（Gemini confidence 失效，不用于排序）
  │  输出：candidates（含 has_waldo / confidence）
  ▼
route_after_detect ── len(candidates) > 1 ? ──┐
  │                                            │
  │ 否（单候选 / 空）                          │ 是（多候选）
  ▼                                            ▼
  │                                       [verify] ────────────────────────
  │                                        │  取 top-K，从原图裁出（+30% padding），
  │                                        │  VLM(gemini-3.5-flash) 横向单选；输出 verified_result
  │                                            │
  ▼                                            ▼
[visualize]────────────────────────────────────────────────────────────────
  │  优先用 verified_result；为空（含单候选跳过 verify）则取最佳候选 patch_bbox，画红框
  ▼
 END
```

> **设计沿革**：`evaluate → calibrate` 迭代回路于 2026-06-10 移除；`analyze` 节点于 2026-06-17 移除（VLM 推荐切割行列数优化了错误的变量——该切多大由 Waldo 绝对像素决定，切图前不可知，改为确定性算术）。

---

## 2. 节点详解

### segment（入口）
- **无 VLM 调用**，纯几何切片
- **输入**：`original_image_path`，`focus_regions`（初始=全图 `[[0,0,W,H]]`）
- **过程**：对每个 focus_region 调 `vision/segment.py::tile_region(region, TILE_SIZE, image_size, TILE_OVERLAP)`：
  - 每轴起点 `0, stride, 2·stride, …`（`stride=round(TILE_SIZE×(1-TILE_OVERLAP))`），保留所有 `< length-TILE_SIZE` 者，再补贴边起点 `length-TILE_SIZE` → 每块恰好 `TILE_SIZE×TILE_SIZE`、全图无空洞；`length ≤ TILE_SIZE` 退化单块
  - 跳过宽或高 < `MIN_PATCH_PX=150` 的块
- **输出写入 State**：`candidates`（**重置**），每项 `{patch_bbox, region_idx, row, col, confidence=0.0, crop_path=None, verified=False}`

### detect
- **Provider**：gemini-3.5-flash（`detect.py:VLM_PROVIDER="gemini"` / `VLM_MODEL="gemini-3.5-flash"`）
- **输入**：`candidates`，`original_image_path`
- **过程**：
  1. 跳过宽或高 < `MIN_DETECT_PATCH_PX=150` 的 patch；超过 `MAX_PATCHES_PER_ITER=80` 随机采样
  2. 将 patch 从原图裁剪并保存到 `outputs/patches/`
  3. 串行调用 VLM（`MAX_CONCURRENT=1`），发送 `DETECT_PROMPT`，返回 `{"present": bool, "confidence": float}`
  4. **按 `has_waldo`（present）过滤**；confidence 降序仅作多候选时的稳定排序（Gemini confidence 与 present 矛盾率 77%，无判别意义）
- **限流重试**：429 指数退避（15→30→60→120s，最多 4 次）

### route_after_detect（条件路由，agent/graph.py）
- `len(candidates) > 1` → `verify`；否则（单候选 / 空）→ `visualize`，跳过 verify
- 依据：Gemini detect 高精度，大多数图每张只标 1 个真候选；少数冒 false positive 的图才需 verify 去伪存真

### verify（仅多候选时触发）
- **Provider**：gemini-3.5-flash（`verify.py:VLM_PROVIDER="gemini"`，横向单选）
- **输入**：`candidates`，`original_image_path`
- **过程**：对全部 present 候选（上限 `VERIFY_MAX=12`）：
  1. 各自 patch 内 bbox → 原图坐标（`waldo_orig_bbox`），向外扩展 30% padding（最小 120px），裁图保存到 `outputs/verify/`
  2. 把这些裁剪图**一次性**发给 Gemini 做横向单选（`SELECT_PROMPT`），返回 `{"choice": int(-1 表示都不是), "confidence": float, "per_image": [bool…]}`
  3. 不再逐张独立判断——强制候选间相对比较，避免逐张误检与红白条纹误导（诊断见 `scripts/compare_verify_candidates.py`）
- **输出写入 State**：`candidates` 更新 `verified`（仅选中项 True）/ `verify_confidence` / `orig_bbox` / `verify_crop_path` / `verify_looks_waldo`；`verified_result` = `choice` 选中候选的 `orig_bbox`，`choice=-1` 则 `None`

### visualize
- **无 VLM 调用**，流水线终点
- **输入**：`verified_result`（优先），或 candidates 中最佳候选的 bbox（兜底，单候选跳过 verify 时走此路，用 `patch_bbox`）
- **过程**：`tools/visualize.py` 的 `@tool` 在原图画红框，保存到 `outputs/{basename}_result.jpg`
- **main.py 结果分类**：`verified_result` 有值 → "confirmed (verified)"；无值但有候选且 verify 未跑过（判据：candidate 是否带 `verify_confidence`）→ "located (detect-only)"；否则 → "not found"

---

## 3. State 核心字段流转

```
字段                   由谁写入              被谁读取
──────────────────────────────────────────────────────────
original_image_path    initial_state         全部节点
focus_regions          initial_state(全图)   segment
candidates             segment(重置)         detect / verify / visualize / main
                       detect(更新)
                       verify(更新)
verified_result        verify                visualize / main
iteration              initial_state(0)      detect / verify（命名输出文件，恒为 0）
```

---

## 4. VLM 分工一览

| 节点 | Provider | Prompt | 目标 | 策略 |
|------|----------|--------|------|------|
| detect | **gemini-3.5-flash** | `DETECT_PROMPT` | 有没有 Waldo（present 二元） | 高召回高精度；只用 present，不用 confidence |
| verify | **gemini-3.5-flash** | `SELECT_PROMPT` | 这几张里哪张是 Waldo | 横向单选；多候选间相对比较、去伪存真 |

---

## 5. 关键设计决策

**256×256px 切片下限**
极限测试（`scripts/gemini_limit.py`）证明 Waldo 能否被定位由其绝对像素大小决定；256px 覆盖含小 Waldo 难图（如 2.jpg，Waldo ~50×90）在内的所有可检出图。`TILE_SIZE` 做成可调常量，提速时可上调（代价：小 Waldo 图可能漏检）。150px 是两处硬性下限（`MIN_PATCH_PX` / `MIN_DETECT_PATCH_PX`）。

**确定性切片取代 VLM 推荐网格**
切多大由 Waldo 绝对像素决定，切图前不可知，VLM 推荐视觉复杂度优化了错误的变量、还引入一次 API 调用 + 失败面。改为纯几何固定尺寸滑窗，可纯单测、不花 API、可复现。

**detect 用 Gemini 二元信号 + 单候选跳过 verify**
gemini-3.5-flash present 二元信号最强（召回 94.4% / 误检 4.9%），但 confidence 失效（与 present 矛盾率 77%），故按 present 过滤、不靠 confidence 排序。大多数图只剩 1 个真候选 → 跳过 verify 省一次调用；多候选才走 verify。verify 也用 gemini-3.5-flash，但走**横向单选**（把全部候选摆一起选一个）——实测比 gpt-5.5 逐张判断更准（逐张在难图上会多判 Yes、被红白条纹骗）。

**bbox 来源的退化链**
detect（Gemini）不返回精确 bbox → verify 退化为整个 patch 坐标 → verify 用加 padding 的区域确认 → visualize 取 `orig_bbox`（精确）或 `patch_bbox`（兜底，单候选路径即用整块 patch_bbox）。

---

## 6. 关键可调参数速查

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `TILE_SIZE` | `nodes/segment.py` | 256 | 切片边长（px），可调 |
| `TILE_OVERLAP` | `nodes/segment.py` | 0.15 | 相邻切片重叠比例 |
| `MIN_PATCH_PX` | `nodes/segment.py` | 150 | 跳过过小 patch 的下限 |
| `MIN_DETECT_PATCH_PX` | `nodes/detect.py` | 150 | detect 跳过过小 patch 的下限 |
| `MAX_PATCHES_PER_ITER` | `nodes/detect.py` | 80 | patch 硬性上限，超出随机采样 |
| `MAX_CONCURRENT` | `nodes/detect.py` | 1 | detect 并发数 |
| `VERIFY_MAX` | `nodes/verify.py` | 12 | 送 verify 的候选数安全上限（验证全部 present 候选，仅多候选触发） |
