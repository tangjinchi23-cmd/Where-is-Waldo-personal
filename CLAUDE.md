# WhereisWaldoAgent

## 项目目标

在复杂的 Where's Waldo 图片中，通过 AI Agent 自动定位 Waldo 并返回原图坐标的 bbox。

> **当前状态**：已完成大规模重构。detect 默认使用 **GPT-4o**（实验验证召回率更高），analyze/verify 使用 **Claude（claude-sonnet-4-6）**。核心设计锚点：**200×200px 是 VLM 可靠识别 Waldo 的最小 patch 尺寸**。

---

## 技术栈

- **Agent 框架**：LangGraph（`StateGraph`，有状态的迭代循环 + 条件路由；保留以备未来并行 detect 分支）
- **多模态模型（VLM）**：GPT-4o（全部节点）；统一接口支持热插拔
- **图像处理**：Pillow
- **并发**：`concurrent.futures.ThreadPoolExecutor`（detect 节点，当前 `MAX_CONCURRENT=1`）

### 依赖（requirements.txt）

```
langgraph>=0.2.0
langchain-core>=0.2.0
anthropic>=0.30.0
openai>=1.30.0
pillow>=10.0.0
# 可选：google-generativeai>=0.7.0
```

### 环境变量（.env）

- `OPENAI_API_KEY` —— GPT-4o 调用所需（全部节点）
- `ANTHROPIC_API_KEY` —— 切换回 Claude 时所需
- 切换 Gemini 时需额外配置 `GOOGLE_API_KEY`

`main.py` 启动时通过 `dotenv.load_dotenv()` 加载（缺失则跳过）。

---

## 运行方式

```bash
python main.py [图片路径]          # 默认 original-images/1.jpg
python draw_graph.py               # 导出 LangGraph 流程图 agent_graph.png
python tests/check_detect_prompt.py  # 测试 DETECT_PROMPT 在已知含 Waldo 图片上的召回率
```

入口 `run_agent(image_path, max_iterations=5, grid_size=2)`，返回最终 `WaldoState`。

---

## 核心流程设计

### 整体流程图

```
START
  ↓
[analyze]   ← VLM(Claude) 看缩略图，根据复杂度推荐切割行列数（目标每格约 200×200px）
              输出：N×M 个 focus_regions（覆盖全图）+ grid_rows/grid_cols
  ↓
[segment]   ← 将 focus_regions 切分为 patch；第1轮 grid_size=1（格子即 patch）；
              calibrate 后 grid_size=2（400px 区域切成 4 个 200px patch）
              跳过宽或高 < 150px 的区域
  ↓
[detect]    ← VLM(GPT-4o) 对每个 patch 判断是否含 Waldo，返回置信度
              跳过 < 150px 的 patch，过滤 confidence < 0.15，按置信度降序
  ↓
[verify]    ← 取 top-3 候选，从原图裁出（带 30% padding），VLM(Claude) 二次确认
  ↓
[evaluate]  ← 仅递增 iteration；路由由 route_after_evaluate 决定
  ├── verified_result 非空          → [visualize] → END
  ├── iteration ≥ max               → [visualize]（取最佳候选）→ END
  ├── 最佳候选 verify_conf ≥ 0.85   → [visualize] → END
  └── 否则                          → [calibrate] → 回到 [segment]
```

### 两阶段检测设计

- **阶段一（detect）**：VLM 看 ~200×200px 的 patch，判断"有没有 Waldo"，返回置信度。prompt 明确要求"宁可误报，不要漏检"（减少 false negative）。实验验证 GPT-4o 在此任务上召回率优于 Claude。
- **阶段二（verify）**：将候选区域从**原图**裁出（加 30% padding、最小 120px），发给 Claude 二次确认"这真的是 Waldo 吗"。

### calibrate 扩展策略

calibrate **不使用 patch_bbox 直接作为下一轮 focus_region**（否则 patch 会指数级缩小）。策略：以命中 patch 的**中心**向外扩展到 `EXPAND_TO=400px` 正方形，再以 `grid_size=2` 切分，保证每个 patch 始终约 200×200px。

---

## LangGraph State 定义（agent/state.py）

```python
class WaldoState(TypedDict):
    original_image_path: str       # 原图路径
    grid_size: int                 # 当前切分粒度（analyze 后=1，calibrate 后=2）
    region_grid_sizes: dict        # 已废弃，保留兼容
    focus_regions: list            # [[x, y, w, h], ...]，当前重点搜索区域（原图坐标）
    grid_rows: int                 # analyze 输出：VLM 推荐行数
    grid_cols: int                 # analyze 输出：VLM 推荐列数
    region_complexity: list        # 保留兼容，填充 0.5
    candidates: list               # [{patch_bbox, confidence, crop_path, verified, ...}, ...]
    verified_result: list | None   # [x, y, w, h]（原图坐标），未找到则 None
    iteration: int
    max_iterations: int
```

`initial_state(image_path, max_iterations=5, grid_size=2)`：以全图作为初始 focus_region，由 analyze 替换。

---

## 各节点职责（agent/nodes/）

| 节点 | Provider | 输入 | 输出 | 说明 |
|------|----------|------|------|------|
| `analyze` | GPT-4o | `original_image_path` | `focus_regions`, `grid_rows`, `grid_cols` | VLM 推荐行列数，全图均匀切 N×M 格（每格约 200×200px） |
| `segment` | — | `focus_regions`, `grid_size` | `candidates`（每轮重置） | 按 grid_size 切分，跳过 < 150px 区域，带 12% overlap |
| `detect` | GPT-4o | `candidates`, `original_image_path` | `candidates`（含 confidence / has_waldo） | 跳过 < 150px patch，过滤低置信度，按置信度降序 |
| `verify` | GPT-4o | `candidates` 中 top-3 | `candidates`（verified 字段）+ `verified_result` | 裁出带 padding 的区域 VLM 二次确认 |
| `evaluate` | — | `candidates`, `iteration` | `iteration + 1` | 仅递增；路由交给 `route_after_evaluate` |
| `calibrate` | — | `candidates`, `original_image_path` | 新 `focus_regions`, `grid_size=2` | 以命中 patch 中心扩展 400px 正方形作为新焦点 |
| `visualize` | — | `verified_result` / 最佳候选 | 标注图片路径 | 调用 `tools/visualize.py` 画红框 |

### 图组装（agent/graph.py）

- 入口：`analyze`
- 线性边：`analyze → segment → detect → verify → evaluate`
- 回路：`calibrate → segment`
- 条件边：`evaluate →(route_after_evaluate)→ visualize | calibrate`
- 终点：`visualize → END`

---

## VLM 抽象层（llm/vlm_client.py）

统一接口，四家 provider 同构实现，工厂函数一键切换：

```python
get_vlm_client(provider="claude")   # "claude" | "gpt4o" | "gemini" | "qwen"
```

| Provider | 类 | 默认 model | 备注 |
|----------|----|-----------|------|
| claude | `ClaudeVLMClient` | `claude-sonnet-4-6` | analyze / verify 默认 |
| gpt4o | `GPT4oVLMClient` | `gpt-5.5` | detect 默认；实验验证召回率更高 |
| gemini | `GeminiVLMClient` | `gemini-1.5-flash` | 可选 |
| qwen | `QwenVLMClient` | `qwen-vl-max` | 走 DashScope OpenAI 兼容接口；需 `DASHSCOPE_API_KEY` |

每个 client 实现三方法：
- `call(image_path, prompt, max_tokens)` —— 发图 + 自定义 prompt，返回原始文本
- `detect(image_path) -> DetectResult(has_waldo, confidence, bbox, raw_response)`
- `verify(image_path) -> VerifyResult(is_waldo, confidence, raw_response)`

`DETECT_PROMPT` / `VERIFY_PROMPT` 定义在 `prompts.py`；`_extract_json()` 容错解析 markdown 代码块。

---

## 分割策略：VLM 驱动的自适应网格 + 中心扩展细化

- **analyze**：VLM 收到图片尺寸信息，推荐 `{"rows": N, "cols": M}`（建议值 = `max(2, round(dim/200))`，VLM 可按复杂度上下调整）
- **segment**：第 1 轮 `grid_size=1`（直接用 analyze 的格子）；calibrate 后 `grid_size=2`（400px 区域切 4 块）；始终跳过 < 150px 的 patch
- **calibrate**：取 `has_waldo=True` 中置信度最高的 top-3，以各 patch **中心扩展 400px** 正方形作为新 focus_regions，`grid_size` 固定为 2
- 每轮 `candidates` 在 segment 节点重置

---

## 关键可调参数

| 位置 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| `main.py` | `max_iterations` | 5 | 迭代上限 |
| `nodes/analyze.py` | `THUMBNAIL_MAX` | 900 | 发给 VLM 的缩略图最大边长 |
| `nodes/segment.py` | `MIN_PATCH_PX` | 150 | 跳过过小 patch 的下限 |
| `nodes/detect.py` | `DETECT_CONFIDENCE_THRESHOLD` | 0.15 | 低于此值的 patch 丢弃 |
| | `MIN_DETECT_PATCH_PX` | 150 | detect 跳过过小 patch 的下限 |
| | `MAX_CONCURRENT` | 1 | 并发数（50 req/min 限制下保守串行） |
| | `MAX_PATCHES_PER_ITER` | 40 | 每轮 patch 硬上限 |
| | `MAX_RETRIES` / `RETRY_BASE_WAIT` | 4 / 15s | 429 限流指数退避：15→30→60→120 |
| `nodes/verify.py` | `TOP_K` | 3 | 送验证的候选数 |
| | `PADDING_RATIO` / `MIN_VERIFY_SIZE` | 0.3 / 120px | 裁剪 padding 与最小尺寸 |
| `nodes/evaluate.py` | `VERIFY_CONFIDENCE_THRESHOLD` | 0.85 | 提前退出阈值 |
| `nodes/calibrate.py` | `EXPAND_TO` | 400 | calibrate 扩展区域尺寸（px） |
| | `CALIBRATE_GRID_SIZE` | 2 | calibrate 后 segment 的固定 grid_size |

---

## 文件结构

```
WhereisWaldoAgent/
├── CLAUDE.md
├── main.py                      # 入口：python main.py [图片路径]
├── draw_graph.py                # 导出 LangGraph 流程图
├── prompts.py                   # 集中存放所有 VLM 提示词
├── requirements.txt
├── .env                         # ANTHROPIC_API_KEY + OPENAI_API_KEY
├── agent/
│   ├── __init__.py              # 导出 run_agent
│   ├── state.py                 # WaldoState + initial_state
│   ├── graph.py                 # build_graph / run_agent
│   └── nodes/
│       ├── __init__.py
│       ├── analyze.py           # VLM 推荐行列数，均匀切分全图
│       ├── segment.py           # 切分 focus_regions 为 patch
│       ├── detect.py            # GPT-4o 判断 patch 是否含 Waldo
│       ├── verify.py            # Claude 二次确认候选
│       ├── evaluate.py          # 递增 iteration + 路由决策
│       ├── calibrate.py         # 中心扩展 400px 作为新焦点
│       └── visualize.py         # 在原图标注结果
├── llm/                         # VLM 适配器层
│   ├── __init__.py
│   ├── vlm_client.py            # 兼容垫片（保留旧 import 路径）
│   ├── base.py                  # BaseVLMClient + _extract_json
│   ├── factory.py               # get_vlm_client 工厂
│   ├── results.py               # DetectResult / VerifyResult
│   └── providers/               # Claude / GPT-4o / Gemini / Qwen 实现
├── vision/                      # 图像处理 + 切分
│   ├── __init__.py
│   ├── image_utils.py           # base64 编码、裁剪、保存
│   └── segment.py               # segment_region / segment_all_regions
├── tools/                       # LangChain @tool
│   ├── __init__.py
│   ├── visualize.py             # 画 bbox（@tool）
│   └── crop_image.py            # 裁剪并保存（@tool）
├── tests/                       # 测试脚本
│   ├── check_claude.py          # Claude API key 验证
│   ├── check_gpt4o.py           # GPT-4o API key 验证
│   ├── check_detect_prompt.py   # DETECT_PROMPT 召回率测试（images_withWaldo/）
│   └── ...
├── docs/                        # 设计文档
│   ├── agent_logic.md           # 完整逻辑链路文档
│   └── analyze_node_design.md   # analyze 节点设计分析
├── original-images/             # 测试图片（1.jpg ~ 19.jpg, OIP.jpg）
├── images_withWaldo/            # 含 Waldo 的标注图片（召回率测试用）
└── outputs/
    ├── patches/                 # 各轮裁出的 patch
    ├── thumbs/                  # analyze 的网格缩略图
    ├── verify/                  # verify 的特写裁剪
    └── [basename]_result.jpg    # 最终标注图
```

---

## 已知 Bug（待修复）

> 详见 `docs/run_issues_2026-06-09.md`

- [ ] **analyze 解析失败**：GPT-4o 偶发返回空字符串，`_parse_grid_dims` 降级到 fallback。需打印完整原始响应定位根因；`ANALYZE_MAX_TOKENS` 从 64 提高到 128；可加重试逻辑
- [ ] **main.py grid_size 参数过时**：`run_agent(..., grid_size=3)` 应改为 `grid_size=1`，`initial_state` 默认值同步改为 1，与新设计对齐（当前靠 min_patch_size 安全网处理，行为正确但日志误导）
- [ ] **detect 截断导致系统性漏检**：`MAX_PATCHES_PER_ITER=40` 会截掉后 20 个格子（图片右下角）。可选方案：提高上限至 80、随机打乱后截断、或按复杂度优先排序

---

## 待确认 / 优化方向

- [ ] **量化评测**：对 `original-images/` 建立 ground truth 标注 + IoU 命中率脚本，每次改动可量化验证效果
- [ ] **并发上限**：当前 `MAX_CONCURRENT=1` 受 50 req/min 限制；系统稳定后可考虑提额 + 并行 detect 分支（LangGraph 已预留接口）
- [ ] **置信度阈值调参**：`DETECT_CONFIDENCE_THRESHOLD=0.15`、`VERIFY_CONFIDENCE_THRESHOLD=0.85` 需用实测数据调优
- [ ] **verify 精修 bbox**：目前 verify 只做 true/false 判断，可让 VLM 同时回传精确 bbox 坐标
