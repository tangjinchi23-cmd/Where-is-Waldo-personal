# 运行问题记录 — 2026-06-09

测试图片：`original-images/1.jpg`（2048×1251px）
运行命令：`python main.py original-images/1.jpg`
最终结果：✅ 找到 Waldo，bbox=`[614, 417, 230, 234]`，verify 置信度 0.86，第 0 轮结束

---

## 问题 1：analyze 节点 VLM 返回空字符串，解析失败

### 现象

```
[analyze] Failed to parse grid dims from: '', using fallback 6×10
[analyze] VLM recommended grid: 6×10  (image: 2048×1251)
```

VLM 返回了空字符串，`_parse_grid_dims` 解析失败，降级到 fallback（`max(2, round(dim/200))`）。

### 根因推测

`analyze.py` 中 `ANALYZE_MAX_TOKENS = 64`，而 `{"rows": 6, "cols": 10}` 本身只需约 20 个 token，正常情况下不会被截断。更可能的原因是：
- GPT-4o 对该 prompt 的响应格式不符合预期（如返回了非 JSON 文本，被 strip 后为空）
- 网络或 API 异常导致响应体为空

### 影响

本次 fallback 值（6×10）与图片尺寸吻合，结果正确。但若图片复杂度需要更多格子，fallback 无法感知，可能漏检。

### 建议修复

1. 在 `_parse_grid_dims` 中打印完整的 `raw` 原始响应（非截断版），方便定位是 VLM 输出问题还是解析问题
2. 将 `ANALYZE_MAX_TOKENS` 从 64 提高到 128，避免潜在截断
3. 增加 analyze 阶段的重试逻辑（类似 detect 的指数退避）

---

## 问题 2：main.py 传入的 grid_size=3 与新设计不符

### 现象

```
[segment] Generated 60 patches from 60 regions (grid_size=3)
```

`main.py` 调用 `run_agent(image_path, max_iterations=5, grid_size=3)`，传入了旧的 `grid_size=3`。

### 根因

重构后 analyze 节点已负责切分全图（输出 60 个格子），`grid_size` 在第一轮应为 1（每个格子直接作为一个 patch）。但 `main.py` 仍使用重构前的旧参数 `grid_size=3`。

### 实际行为

`segment_region` 内部检测到 stride（≈68px）< `min_patch_size`（150px），自动将 grid_size 降为 1，最终行为正确（60 regions × 1 patch = 60 patches）。安全网生效，但日志中 `grid_size=3` 具有误导性。

### 建议修复

将 `main.py` 第 20 行改为：

```python
final_state = run_agent(image_path, max_iterations=5, grid_size=1)
```

同时将 `initial_state` 的默认值 `grid_size=2` 改为 `grid_size=1`，与新设计对齐。

---

## 问题 3：detect 截断了 60 个 patch 中的 20 个

### 现象

```
[detect] Truncating 60 → 40 patches
[detect] iter=0, patches=40, workers=1
```

analyze 生成了 60 个格子，但 `MAX_PATCHES_PER_ITER=40` 截断了后 20 个。

### 影响

截断是按列表顺序（region 0 → region 59），后 20 个格子（图片右下角部分区域）完全未被检测。若 Waldo 位于这些区域，第一轮会漏检，需依赖 calibrate 补救。

### 建议修复

- 方案 A：将 `MAX_PATCHES_PER_ITER` 提高到 80~100（接受更高 API 成本）
- 方案 B：将 60 个格子随机打乱后再截断，避免系统性地漏掉某一区域
- 方案 C：按复杂度排序（analyze 已输出 `region_complexity`，当前填充 0.5 未使用），优先处理高复杂度格子

---

## 本次运行总体评价

| 指标 | 数值 |
|------|------|
| 总格子数（analyze 输出） | 60 |
| 实际送入 detect 的 patch 数 | 40（截断） |
| 通过 detect 阈值的 patch 数 | 34 |
| verify 候选数 | 3（top-3） |
| 最终迭代轮次 | 1（第 0 轮即找到） |
| 结果 | ✅ 正确，bbox=[614, 417, 230, 234] |

尽管存在上述三个问题，本次运行因 Waldo 恰好位于前 40 个 patch 覆盖的区域，最终仍正确找到目标。问题 1 和问题 3 存在漏检风险，建议在多图测试中验证。
