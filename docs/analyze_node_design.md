# analyze 节点设计分析

## 1. 当前做什么

```
原图
  │
  ▼
缩略图（≤900px）+ 叠加 2×2 网格线和格子标签（R0C0…R1C1）
  │
  ▼
Claude VLM → {"grid_scores": [[s00, s01], [s10, s11]]}  （0.0~1.0 复杂度）
  │
  ▼
分数 → grid_size 映射（≥0.7→3，0.4~0.7→2，<0.4→1）
  │
  ▼
输出：4 个 focus_regions（格子原图坐标） + region_grid_sizes + region_complexity
```

**代码关键常量（`nodes/analyze.py`）：**

| 常量 | 值 | 含义 |
|------|----|------|
| `ANALYZE_GRID` | 2 | 2×2 = 4 个格子 |
| `THUMBNAIL_MAX` | 900 | 缩略图最大边长 |
| `VLM_PROVIDER` | `"claude"` | 使用 Claude |
| `ANALYZE_MAX_TOKENS` | 512 | 复杂度 JSON 较大 |

**复杂度 → grid_size 映射：**

```
分数 ≥ 0.7  →  grid_size = 3  →  segment 时切 3×3 = 9 个 patch
分数 0.4~0.7 →  grid_size = 2  →  segment 时切 2×2 = 4 个 patch
分数 < 0.4  →  grid_size = 1  →  segment 时不切分，整块作为 1 个 patch
```

**第一轮 patch 数量的理论上限：**
4 个格子 × 最大 9 patch/格子 = **36 个 patch**（实际 `MAX_PATCHES_PER_ITER=40` 不会触发截断）

---

## 2. analyze 在图中的位置

```
START → [analyze] → [segment] → [detect] → [verify] → [evaluate]
                        ↑                                    │
                   [calibrate] ◄─────────────────────────────┘
```

**关键事实：analyze 只在每次 `run_agent()` 调用时执行一次，calibrate 的回路绕过了它。**

calibrate 执行后，`region_grid_sizes` 被重置为 `{}`，segment 会用全局 `grid_size`（已 +2）作为 fallback——analyze 输出的自适应 grid_size 仅在**第一轮**有效。

---

## 3. 设计问题

### 3.1 输出的缩略图数量

analyze 每次运行只保存 **1 张**缩略图到 `outputs/thumbs/`，不存在"上百张缩略图"的问题。

"上百个文件"更可能来自 `outputs/patches/`：detect 节点每个 patch 保存一张裁图，5 次迭代 × 最多 40 patch = **最多 200 张**。如果之前版本的 analyze 在 calibrate 回路中，或 `ANALYZE_GRID` 更大，数量会进一步增加。

### 3.2 2×2 网格太粗

- 全图只分 4 块，每块面积是原图的 25%
- 如果 Waldo 在格子边界附近，复杂度分数可能被相邻格子"均摊"，导致正确格子分数偏低
- 4 个格子全部都会进入 segment，analyze 的过滤作用几乎为零（4 个格子 = 全图）

### 3.3 "复杂度 = Waldo 可能性"的假设存疑

- VLM 打的是视觉复杂度（人群密集程度），不是 Waldo 存在概率
- Waldo 有时藏在看似简单的边缘区域，简单区域可能被分配 grid_size=1（整块不切分），降低了找到他的概率
- 这个打分对 Waldo 定位的实际贡献目前**没有量化验证**

### 3.4 prompt 没有集中管理

analyze 的 prompt 用了 `_build_prompt(n)` 函数，直接写在 `analyze.py` 里（后来重构到 `prompts.py` 的 `build_analyze_prompt`），与 `DETECT_PROMPT` / `VERIFY_PROMPT` 风格不一致。

### 3.5 provider 与 detect 不一致

analyze 用 Claude，detect 用 GPT-4o。如果未来调整 provider 策略，需要同步修改两处。

---

## 4. 简化方向

### 方案 A：完全移除 analyze，固定全图切分

```
START → [segment(全图, grid_size=3)] → [detect] → ...
```

- 第一轮直接切 3×3 = 9 个 patch，覆盖全图
- 去掉一次 VLM 调用，降低延迟和成本
- 适合场景：全图复杂度均匀，或不信任 VLM 的复杂度判断

### 方案 B：保留 analyze，但简化输出

- 不做 grid_size 差异化，统一用固定粒度
- analyze 只做"哪个格子最可疑"的二元判断（top-2 格子进入下一轮），而不是影响切分粒度
- 减少 analyze 输出对 segment 逻辑的干扰

### 方案 C：扩大网格（如 3×3），取 top-K 高分格子

```
analyze: 3×3 网格 → 取复杂度 top-4 格子 → 只处理这 4 个格子
segment: 4 个格子 × 固定 grid_size=3 = 36 patch（与现在相当）
```

- 覆盖全图的同时真正过滤掉低价值区域
- 但 patch 总量不变，需要权衡复杂度增加是否值得

---

## 5. 现状总结

| 方面 | 评估 |
|------|------|
| 单次运行是否合理 | 是，只调用一次 VLM，1 张缩略图 |
| 对 patch 总量的控制 | 弱（4 个格子全进 segment，等于没过滤）|
| 复杂度打分的有效性 | 未验证，假设可能不成立 |
| 代码复杂度 | 偏高（缩略图生成、字体渲染、fallback 解析） |
| 对后续迭代的影响 | 仅影响第 1 轮，calibrate 后不再使用 |
