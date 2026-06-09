# WhereisWaldoAgent 逻辑链路文档

## 1. 整体流程

```
START
  │
  ▼
[analyze]  ──────────────────────────────────────────────────────────────
  │  VLM(Claude) 看缩略图，根据复杂度推荐切割行列数（目标每格约 200×200px）
  │  输出：N×M 个 focus_regions（覆盖全图）+ grid_rows + grid_cols
  ▼
[segment]  ──────────────────────────────────────────────────────────────
  │  将 focus_regions 切分为 patch（第1轮 grid_size=1 直接用格子；
  │  calibrate 后 grid_size=2 将 ~400px 区域切成 4 个 ~200px patch）
  │  跳过宽或高 < 150px 的区域
  │  输出：candidates（仅含 patch_bbox）
  ▼
[detect]   ──────────────────────────────────────────────────────────────
  │  VLM(GPT-4o) 对每个 patch 判断是否含 Waldo，返回置信度
  │  跳过 < 150px 的 patch，过滤 confidence < 0.15，按置信度降序
  │  输出：candidates（含 has_waldo / confidence）
  ▼
[verify]   ──────────────────────────────────────────────────────────────
  │  取 top-3 候选，从原图裁出（+30% padding），VLM(Claude) 二次确认
  │  输出：candidates（含 verified / verify_confidence）+ verified_result
  ▼
[evaluate] ──────────────────────────────────────────────────────────────
  │  iteration + 1，无其他逻辑
  │  由 route_after_evaluate 决定下一跳
  │
  ├─── verified_result 非空            ──→ [visualize] → END
  ├─── iteration ≥ max_iterations      ──→ [visualize] → END
  ├─── 最佳候选 verify_confidence≥0.85 ──→ [visualize] → END
  │
  └─── 否则                            ──→ [calibrate]
                                              │
                                              │  以命中 patch 中心扩展 400px 区域
                                              │  作为新 focus_regions，grid_size=2
                                              │
                                              └─────────────→ [segment]（回路）
```

---

## 2. 节点详解

### analyze
- **Provider**：Claude（`analyze.py:VLM_PROVIDER = "claude"`）
- **输入**：原图路径
- **过程**：
  1. 根据图片尺寸计算建议行列数（`suggest = max(2, round(dim / 200))`）
  2. 生成带网格线的缩略图（≤900px），保存到 `outputs/thumbs/`
  3. 发给 VLM，要求返回 `{"rows": N, "cols": M}`；VLM 可根据复杂度在建议值上下调整
  4. 解析失败时 fallback 到建议值
  5. 将全图均匀切分为 N×M 个格子作为 `focus_regions`
- **输出写入 State**：
  - `focus_regions`：N×M 个格子的 `[x, y, w, h]`（原图坐标，每格约 200×200px）
  - `grid_rows` / `grid_cols`：VLM 推荐的行列数
  - `region_complexity`：填充 0.5（保留兼容，不再使用）
  - `region_grid_sizes`：`{}`（已废弃）

### segment
- **无 VLM 调用**，纯图像切分
- **输入**：`focus_regions`，`grid_size`（analyze 后为 1，calibrate 后为 2）
- **过程**：
  - 对每个 focus_region 按 `grid_size×grid_size` 切分，12% overlap
  - 跳过宽或高 < `MIN_PATCH_PX=150` 的区域（含告警日志）
- **输出写入 State**：
  - `candidates`：每轮**重置**，初始化为 `[{patch_bbox, region_idx, row, col, ...}]`

### detect
- **Provider**：GPT-4o（`detect.py:VLM_PROVIDER = "gpt4o"`）
- **输入**：`candidates`，`original_image_path`
- **过程**：
  1. 跳过宽或高 < `MIN_DETECT_PATCH_PX=150` 的 patch
  2. 将 patch 从原图裁剪并保存到 `outputs/patches/`
  3. 串行调用 VLM（`MAX_CONCURRENT=1`），发送 `DETECT_PROMPT`
  4. VLM 返回 `{"present": true/false, "confidence": 0.0-1.0}`
  5. 过滤 `confidence < 0.15`，按置信度降序排列；截断至 `MAX_PATCHES_PER_ITER=40`
- **限流重试**：遇到 429 指数退避（15s → 30s → 60s → 120s，最多 4 次）

### verify
- **Provider**：Claude（`verify.py:VLM_PROVIDER = "claude"`）
- **输入**：`candidates`（已按置信度降序），`original_image_path`
- **过程**：取前 3 个候选（`TOP_K=3`），每个：
  1. patch 内 bbox → 原图坐标（无精确 bbox 时退化为整个 patch）
  2. 向外扩展 30% padding（最小边长 120px），裁图保存到 `outputs/verify/`
  3. 发给 VLM 使用 `VERIFY_PROMPT` 二次确认，返回 `{"is_waldo": bool, "confidence": float, "reason": str}`
- **输出写入 State**：
  - `candidates`：更新 `verified` / `verify_confidence` / `orig_bbox` / `verify_crop_path`
  - `verified_result`：`is_waldo=True` 中 `verify_confidence` 最高的 `orig_bbox`；无则 `None`

### evaluate + 路由
- `evaluate_node` 仅做 `iteration + 1`
- `route_after_evaluate` 决定路由（按优先级）：
  1. `verified_result is not None` → **visualize**
  2. `iteration >= max_iterations` → **visualize**（兜底退出）
  3. 最佳候选的 `verify_confidence >= 0.85` → **visualize**
  4. 否则 → **calibrate**

### calibrate
- **无 VLM 调用**
- **输入**：`candidates`，`original_image_path`
- **过程**：
  1. 从 `has_waldo=True` 的候选中按 confidence 降序取 top-3（若全为 False 则从全部候选取）
  2. 以每个 patch 的**中心**为圆心，向外扩展到 `EXPAND_TO=400px` 的正方形区域（clamp 至图像边界）
  3. 输出扩展后的区域作为新 `focus_regions`，`grid_size` 固定为 `CALIBRATE_GRID_SIZE=2`
- **输出写入 State**：`focus_regions` / `grid_size=2` / `region_grid_sizes={}`
- 完成后回到 **segment**（不再经过 analyze）

### visualize
- **无 VLM 调用**
- **输入**：`verified_result`（优先），或 candidates 中置信度最高候选的 bbox（兜底）
- **过程**：调用 `tools/visualize.py` 的 `@tool` 在原图上画红框，保存到 `outputs/{basename}_result.jpg`

---

## 3. State 核心字段流转

```
字段                   由谁写入              被谁读取
──────────────────────────────────────────────────────────
original_image_path    initial_state         全部节点
focus_regions          analyze / calibrate   segment
grid_rows              analyze               （记录用）
grid_cols              analyze               （记录用）
grid_size              initial_state(2)      segment
                       calibrate(固定=2)
region_grid_sizes      已废弃，始终为 {}     segment（忽略）
region_complexity      analyze(填0.5)        （不再使用）
candidates             segment(重置)         detect / verify / calibrate / visualize
                       detect(更新)
                       verify(更新)
verified_result        verify                evaluate(路由) / visualize
iteration              initial_state(0)      evaluate(路由)
                       evaluate(+1)
```

---

## 4. VLM 分工一览

| 节点 | Provider | Prompt | 目标 | 策略 |
|------|----------|--------|------|------|
| analyze | Claude | `build_analyze_prompt` | 推荐切割行列数 | 宽松 |
| detect | **GPT-4o** | `DETECT_PROMPT` | 有没有 Waldo + 置信度 | **宽松**（宁可误报，不要漏检） |
| verify | Claude | `VERIFY_PROMPT` | 这真的是 Waldo 吗 | **严格**（必须看到红白条纹） |

---

## 5. 关键设计决策

**200×200px 作为设计锚点**
实验证明 VLM 在 200×200px 的裁图内可以可靠判断 Waldo 是否存在。analyze 推荐行列数的依据是 `dim / 200`，calibrate 扩展 400px 后以 grid_size=2 切分，均保证每个 patch ≈ 200×200px。150px 是两处硬性下限（`MIN_PATCH_PX` / `MIN_DETECT_PATCH_PX`）。

**两阶段检测（detect → verify）**
detect 追求高召回（`DETECT_PROMPT` 明确要求"宁可标 true"，实验验证 GPT-4o 表现更好），verify 追求高精度（`VERIFY_PROMPT` 要求清晰看到红白条纹才确认）。

**calibrate 扩展而非收缩**
旧版 calibrate 直接用 `patch_bbox` 作为下一轮 `focus_region`，叠加 grid_size 递增，导致 patch 指数级缩小。新版以 patch 中心扩展 400px 正方形，保证后续 patch 始终在可识别尺寸。

**bbox 来源的退化链**
detect 不一定返回精确 bbox → verify 退化为整个 patch 坐标 → verify 用加了 padding 的区域发给 VLM → visualize 取 `orig_bbox`（精确）或 `patch_bbox`（兜底）。

---

## 6. 关键可调参数速查

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `max_iterations` | `main.py` | 5 | 最大迭代次数 |
| `THUMBNAIL_MAX` | `nodes/analyze.py` | 900 | 发给 VLM 的缩略图最大边长 |
| `MIN_PATCH_PX` | `nodes/segment.py` | 150 | 跳过过小 patch 的下限 |
| `DETECT_CONFIDENCE_THRESHOLD` | `nodes/detect.py` | 0.15 | detect 过滤阈值 |
| `MIN_DETECT_PATCH_PX` | `nodes/detect.py` | 150 | detect 跳过过小 patch 的下限 |
| `MAX_PATCHES_PER_ITER` | `nodes/detect.py` | 40 | 每轮 patch 硬性上限 |
| `TOP_K` | `nodes/verify.py` | 3 | 送 verify 的候选数 |
| `VERIFY_CONFIDENCE_THRESHOLD` | `nodes/evaluate.py` | 0.85 | 提前退出阈值 |
| `EXPAND_TO` | `nodes/calibrate.py` | 400 | calibrate 扩展区域尺寸（px）|
| `CALIBRATE_GRID_SIZE` | `nodes/calibrate.py` | 2 | calibrate 后 segment 的固定 grid_size |
