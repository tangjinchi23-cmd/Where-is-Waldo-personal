# WhereisWaldoAgent 逻辑链路文档

## 1. 整体流程

当前为**线性流水线**，无迭代回路（evaluate / calibrate 已移除）：

```
START
  │
  ▼
[analyze]  ──────────────────────────────────────────────────────────────
  │  VLM(GPT-4o) 看缩略图，根据复杂度推荐切割行列数（目标每格约 200×200px）
  │  输出：N×M 个 focus_regions（覆盖全图）+ grid_rows + grid_cols
  ▼
[segment]  ──────────────────────────────────────────────────────────────
  │  将 focus_regions 按 grid_size=1 切分为 patch（格子即 patch）
  │  跳过宽或高 < 150px 的区域
  │  输出：candidates（仅含 patch_bbox）
  ▼
[detect]   ──────────────────────────────────────────────────────────────
  │  VLM(GPT-4o) 对每个 patch 判断是否含 Waldo，返回置信度
  │  跳过 < 150px 的 patch，过滤 confidence < 0.15，按置信度降序
  │  输出：candidates（含 has_waldo / confidence）
  ▼
[verify]   ──────────────────────────────────────────────────────────────
  │  取 top-3 候选，从原图裁出（+30% padding），VLM(GPT-4o) 二次确认
  │  输出：candidates（含 verified / verify_confidence）+ verified_result
  ▼
[visualize]────────────────────────────────────────────────────────────────
  │  优先用 verified_result；为空则取最佳候选 bbox，在原图画红框
  ▼
 END
```

> **设计沿革**：原 `evaluate → calibrate → segment` 迭代回路已于 2026-06-10 移除。原因：(1) calibrate 重搜的是已失败的同尺寸（~200px）热区，边际收益低；(2) evaluate 仅做 `iteration+1`，是冗余节点；(3) 旧路由存在缺陷——verify 对"非 Waldo"判断的高 confidence（如 0.96）会误触 `verify_confidence≥0.85` 的提前退出。如后续遇复杂案例需二次细化，再按需接入新节点。

---

## 2. 节点详解

### analyze
- **Provider**：GPT-4o（`analyze.py:VLM_PROVIDER = "gpt4o"`）
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

### segment
- **无 VLM 调用**，纯图像切分
- **输入**：`focus_regions`，`grid_size`（线性流水线下恒为 1）
- **过程**：
  - 对每个 focus_region 按 `grid_size×grid_size` 切分，12% overlap
  - 跳过宽或高 < `MIN_PATCH_PX=150` 的区域（含告警日志）
- **输出写入 State**：
  - `candidates`：**重置**，初始化为 `[{patch_bbox, region_idx, row, col, ...}]`

### detect
- **Provider**：GPT-4o（`detect.py:VLM_PROVIDER = "gpt4o"`）
- **输入**：`candidates`，`original_image_path`
- **过程**：
  1. 跳过宽或高 < `MIN_DETECT_PATCH_PX=150` 的 patch
  2. 将 patch 从原图裁剪并保存到 `outputs/patches/`
  3. 串行调用 VLM（`MAX_CONCURRENT=1`），发送 `DETECT_PROMPT`
  4. VLM 返回 `{"present": true/false, "confidence": 0.0-1.0}`
  5. 过滤 `confidence < 0.15`，按置信度降序排列；超过 `MAX_PATCHES_PER_ITER=80` 则随机采样
- **限流重试**：遇到 429 指数退避（15s → 30s → 60s → 120s，最多 4 次）

### verify
- **Provider**：GPT-4o（`verify.py:VLM_PROVIDER = "gpt4o"`）
- **输入**：`candidates`（已按置信度降序），`original_image_path`
- **过程**：取前 3 个候选（`TOP_K=3`），每个：
  1. patch 内 bbox → 原图坐标（无精确 bbox 时退化为整个 patch）
  2. 向外扩展 30% padding（最小边长 120px），裁图保存到 `outputs/verify/`
  3. 发给 VLM 使用 `VERIFY_PROMPT` 二次确认，返回 `{"is_waldo": bool, "confidence": float, "reason": str}`
- **输出写入 State**：
  - `candidates`：更新 `verified` / `verify_confidence` / `orig_bbox` / `verify_crop_path`
  - `verified_result`：`is_waldo=True` 中 `verify_confidence` 最高的 `orig_bbox`；无则 `None`

### visualize
- **无 VLM 调用**，流水线终点（verify 直接接 visualize）
- **输入**：`verified_result`（优先），或 candidates 中置信度最高候选的 bbox（兜底）
- **过程**：调用 `tools/visualize.py` 的 `@tool` 在原图上画红框，保存到 `outputs/{basename}_result.jpg`

---

## 3. State 核心字段流转

```
字段                   由谁写入              被谁读取
──────────────────────────────────────────────────────────
original_image_path    initial_state         全部节点
focus_regions          analyze               segment
grid_rows              analyze               （记录用）
grid_cols              analyze               （记录用）
grid_size              initial_state(1)      segment
candidates             segment(重置)         detect / verify / visualize
                       detect(更新)
                       verify(更新)
verified_result        verify                visualize
iteration              initial_state(0)      detect / verify（命名输出文件，恒为 0）
```

---

## 4. VLM 分工一览

| 节点 | Provider | Prompt | 目标 | 策略 |
|------|----------|--------|------|------|
| analyze | **GPT-4o** | `build_analyze_prompt` | 推荐切割行列数 | 宽松 |
| detect | **GPT-4o** | `DETECT_PROMPT` | 有没有 Waldo + 置信度 | **宽松**（宁可误报，不要漏检） |
| verify | **GPT-4o** | `VERIFY_PROMPT` | 这真的是 Waldo 吗 | **严格**（必须看到红白条纹） |

---

## 5. 关键设计决策

**200×200px 作为设计锚点**
实验证明 VLM 在 200×200px 的裁图内可以可靠判断 Waldo 是否存在。analyze 推荐行列数的依据是 `dim / 200`，第 1 轮以 grid_size=1 直接用格子，保证每个 patch ≈ 200×200px。150px 是两处硬性下限（`MIN_PATCH_PX` / `MIN_DETECT_PATCH_PX`）。

**两阶段检测（detect → verify）**
detect 追求高召回（`DETECT_PROMPT` 明确要求"宁可标 true"，实验验证 GPT-4o 表现更好），verify 追求高精度（`VERIFY_PROMPT` 要求清晰看到红白条纹才确认）。

**线性流水线（无迭代回路）**
原 evaluate / calibrate 回路已移除：calibrate 重搜的是已失败的同尺寸热区、边际收益低，evaluate 仅做计数属冗余，且旧路由对"非 Waldo"高 confidence 会误触提前退出。当前为单趟 analyze→segment→detect→verify→visualize。

**bbox 来源的退化链**
detect 不一定返回精确 bbox → verify 退化为整个 patch 坐标 → verify 用加了 padding 的区域发给 VLM → visualize 取 `orig_bbox`（精确）或 `patch_bbox`（兜底）。

---

## 6. 关键可调参数速查

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `ANALYZE_MAX_TOKENS` | `nodes/analyze.py` | 128 | VLM 响应的 token 上限（过低会截断 JSON，触发 fallback） |
| `THUMBNAIL_MAX` | `nodes/analyze.py` | 900 | 发给 VLM 的缩略图最大边长 |
| `MIN_PATCH_PX` | `nodes/segment.py` | 150 | 跳过过小 patch 的下限 |
| `DETECT_CONFIDENCE_THRESHOLD` | `nodes/detect.py` | 0.15 | detect 过滤阈值 |
| `MIN_DETECT_PATCH_PX` | `nodes/detect.py` | 150 | detect 跳过过小 patch 的下限 |
| `MAX_PATCHES_PER_ITER` | `nodes/detect.py` | 80 | patch 硬性上限，超出随机采样 |
| `MAX_CONCURRENT` | `nodes/detect.py` | 1 | detect 并发数 |
| `TOP_K` | `nodes/verify.py` | 3 | 送 verify 的候选数 |
