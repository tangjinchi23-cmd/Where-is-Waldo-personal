# WhereisWaldoAgent

## 开发约定

- Git commit messages（标题和正文）必须全部用**英文**书写。

---

## 项目目标

在复杂的 Where's Waldo 图片中，自动定位 Waldo 并返回原图坐标的 bbox。

> **这是一条确定性 workflow（流水线），不是 LLM-driven agent。** 控制流完全由代码决定（一条线性流 + 一个 `len(candidates) > 1` 的确定性分支），VLM 只作「分类器/比较器」用，不自主决策下一步。LangGraph 已于 2026-06-22 移除，改为纯函数 + 生成器编排（`agent/pipeline.py`）。
>
> **流水线** = `segment(确定性切片) → detect(Gemini) → [路由] → verify/visualize`。
> - **segment**：确定性固定尺寸滑窗切片（`TILE_SIZE=256` 可调、末块贴边对齐、`TILE_OVERLAP=0.15`）。不调 VLM。
> - **detect 用 `gemini-3.5-flash`**：全量复验（`docs/工作日志.md`）证明其 present 二元信号最强（召回 94.4% / 误检 4.9%）。⚠️ 其 `confidence` 失效（与 present 矛盾率 77%），故 detect 一律按 **present(has_waldo) 二元信号过滤**，绝不依赖 confidence 排序。
> - **detect 后条件路由**：单候选（或空）直接 visualize、跳过 verify；多候选（少数会冒 false positive 的图）才走 verify 去伪存真。
> - **verify 用 `gemini-3.5-flash` 横向单选**：把全部 present 候选的裁剪图**一次性**发给 Gemini，在候选间相对比较、只选唯一真 Waldo（返回 index + per_image）。实测优于逐张判断——逐张在密集难图上会把多张都判 Yes、且被红白条纹误导（诊断见 `scripts/compare_verify_candidates.py`）。
> - 核心设计锚点：**256×256px 是覆盖含小 Waldo 难图（如 2.jpg）的安全切片下限**（极限测试 `scripts/gemini_limit.py`）。
> ⚠️ 下方「Detect Prompt Engineering 准则」小节保留了部分 gpt-5.5 时期的实测结论，仅作 prompt 调参与可切换 provider 的历史参考——主流程 detect/verify 均走 Gemini。

---

## 服务化与前端（2026-06-18）

把 `main.py` 的一次性测试流程服务化，并配 React + FastAPI 前端。三层单向依赖，流水线层不动：

```
[React SPA (antd)] ──/api,/static (SSE)──> [FastAPI: api/main.py] ──> [service/waldo_service.py] ──> stream_pipeline()
```

- **service 层（`service/waldo_service.py`）**：纯 Python，无 HTTP。
  - `list_cases / get_case`：扫 `original-images/` 与 `outputs/` 配对已有结果。
  - `resolve_image(name)`：`original-images/` 优先、`uploads/` 兜底解析图片路径。
  - `run_detection(image_path)`：把 `stream_pipeline()` 逐节点产出的 `(node, delta)` 翻译成标准事件流 `segment → detect → verify → done`，异常转 `error` 事件；`done` 复刻 main 三态判断（verified / detect-only / not-found）。
- **api 层（`api/main.py`，FastAPI）**：`GET /api/cases`、`POST /api/upload`（存 `uploads/`）、`GET /api/detect?name=`（**SSE** 长连接，逐节点推事件）、`/static/*` 挂载三目录。**启动即 `load_dotenv()`**——否则 server 进程读不到 `GOOGLE_API_KEY`，detect 全 patch 失败、静默退化成「未找到」（2026-06-18 实际踩过此坑）。
- **前端（`frontend/`，React + Vite + Ant Design v6）**：选图/上传 → 运行检测 → `Steps` 逐节点亮灯 + 候选画廊（`Image.PreviewGroup`）+ 结果红框图。逻辑核心 `src/pipeline.js`（纯函数事件归约器，有 vitest 单测）；`src/api.js` 封装 fetch + `EventSource`。
- **批量测试**：`scripts/batch_test_all.py` 跑全量原图、复刻三态判断、**断点续跑**，结果落 `outputs/batch_all_results.json`。

### 运行

```bash
uvicorn api.main:app --reload --port 8000             # 后端
cd frontend && npm install && npm run dev             # 前端（:5173，代理 /api /static → :8000）
pytest tests/test_waldo_service.py tests/test_api.py  # 后端测试
cd frontend && npm test                               # 前端 pipeline 单测
```

### 成本 / 配额（2026-06-18 实测）

- `gemini-3.5-flash` 定价：**输入 $1.50 / 输出 $9.00 每百万 token**；单张图 pipeline（~60 detect + 1 verify）≈ **$0.09**，全量 20 张 ≈ $2。优化阶段付费充 ~$10 足够。
- **免费层对本场景不可用**：`gemini-3.5-flash` 免费层 `GenerateRequestsPerDayPerProjectPerModel-FreeTier` = **20 次/天**，连一张图（~60 次调用）都跑不完。
- 设计文档 / 实现计划见 `docs/superpowers/specs/2026-06-18-*` 与 `docs/superpowers/plans/2026-06-18-*`。

---

## Detect Prompt Engineering 准则（2026-06-11 实测，后续调 prompt 必读）

> 完整实验记录见 `docs/detect_eval_2026-06-11.md`；量化工具见 `tests/quick_detect_check.py`（召回）/ `tests/quick_falsepos_check.py`（误检）/ 根目录 `config.json`（可调 provider/model/temperature/max_tokens/repeats/limit）。

**这些是花了大量 API 成本跑出来的结论，调 `DETECT_PROMPT` 前务必先看，别重复踩坑：**

1. **不要在 prompt 里枚举 Waldo 的特征**。模型自身就认识 Waldo，列特征只会帮倒忙：
   - 列「眼镜 / 红白帽子」→ 这些在人群里到处都是 → 模型**脑补** → 误检爆表（实测 100%）。
   - 列「红白条纹**衫**」→ 衫在 200px patch 里常被遮挡/模糊看不清 → 逼模型过严 → **召回崩**（mini 33%）。
   - ✅ **最佳做法 = 不列特征**：让模型「用自己对 Waldo 的认知去找」+ 一句「他可能小/被遮挡/模糊，仔细看」。gpt-5.5 上召回 **88.9%**、误检 ~20%。
2. **本数据集的领域真相**：红白**条纹帽 + 眼镜**总可见；**条纹衫只偶尔出现且模糊**。所以判别**不能**以条纹衫为闸门。
3. **`confidence` 语义必须明确**：= 「Waldo 存在的概率」，且**必须与 `present` 一致**（present=false → conf 接近 0）。否则模型会把它当「我对答案的确信度」，出现 present=false / conf=0.98 的矛盾，污染 detect 的置信度排序。
4. **gpt-5.5 是推理模型，必有 token 截断坑**：reasoning token 会吃光 `max_completion_tokens`，留给 content 的预算不足 → **返回空响应**被解析成 present=false/conf=0（假失败）。**任何 gpt-5.5 调用（detect/verify/analyze）都要把 max_tokens 调高（≥4096）**。
5. **mini 有天花板**：受 Waldo 绝对像素尺寸（~30-50px）限制，prompt/缩 patch 都救不动判别力；要质变需上 gpt-5.5。**200×200 是下限，往更小走是负收益**。
6. **temperature**：mini 设 **0** 求可复现；gpt-5.5 是推理模型**必须 1**（传其它值 API 报错）。

> **正式流水线与测试的分工**：`DETECT_PROMPT` 保持精简、不要求模型输出 reason（省 token）；prompt engineering 时由 `tests/quick_config.run_repeats` 临时追加「附原因」，只走测试路径。

---

## 技术栈

- **编排**：纯 Python 函数 + 生成器（`agent/pipeline.py`）。无 agent 框架、无 LangGraph——流程是确定性 workflow，顺序调用各节点、`stream_pipeline` 逐节点产出增量供 SSE。
- **多模态模型（VLM）**：`gemini-3.5-flash`（detect + verify 主力）；统一接口支持热插拔（可切回 gpt-5.5 / Claude / Qwen）
- **图像处理**：Pillow（切片 + 裁剪，无 VLM）
- **并发**：`concurrent.futures.ThreadPoolExecutor`（detect 节点，当前 `MAX_CONCURRENT=10`）

### 依赖（requirements.txt）

```
langchain-core>=0.2.0   # 仅 tools/ 的 @tool 装饰器与 vision/segment.py::get_image_size 用
anthropic>=0.30.0
openai>=1.30.0
pillow>=10.0.0
# 可选：google-generativeai>=0.7.0
```

> LangGraph 已移除，`requirements.txt` 里的 `langgraph` 可删。`langchain-core` 仍需保留（`@tool` 装饰器依赖）。

### 环境变量（.env）

- `GOOGLE_API_KEY` —— `gemini-3.5-flash` 调用所需（detect + verify，主力）；`google.generativeai` 自动从环境读取
- `OPENAI_API_KEY` —— 切换回 gpt-5.5 时所需
- `ANTHROPIC_API_KEY` —— 切换回 Claude 时所需

`main.py` 启动时通过 `dotenv.load_dotenv()` 加载（缺失则跳过）。

---

## 运行方式

```bash
python main.py [图片路径]          # 默认 original-images/1.jpg
python tests/check_detect_prompt.py  # 测试 DETECT_PROMPT 在已知含 Waldo 图片上的召回率
```

入口 `run_pipeline(image_path)`（`agent/pipeline.py`），返回最终 `WaldoState`。

---

## 核心流程设计

### 整体流程图

确定性 workflow：`agent/pipeline.py::_run_nodes` 顺序调用各节点，每步 `state.update(delta)` 并产出 `(node, delta)`。第一次 VLM 调用推迟到 detect：

```
[segment]   ← 确定性几何：把整图按 TILE_SIZE=256 固定尺寸滑窗切片，
              末块贴边对齐、TILE_OVERLAP=0.15；跳过 < 150px 的块。不调 VLM
  ↓
[detect]    ← VLM(gemini-3.5-flash) 对每个 patch 判断是否含 Waldo
              按 present(has_waldo) 二元信号过滤（Gemini confidence 失效，不用于排序）
  ↓
 ├─ 候选 > 1 ─→ [verify]    ← 全部候选裁出（带 30% padding），VLM(gemini-3.5-flash) 横向单选唯一真 Waldo
 │                ↓
 └─ 候选 ≤ 1 ─────┴─→ [visualize] ← 优先 verified_result；为空则取候选 patch_bbox，画红框
```

> **路由说明**：`_run_nodes`（`agent/pipeline.py`）里一行 `if len(state["candidates"]) > 1:` —— 多候选走 verify，否则（单候选/空）跳过 verify 直接 visualize。这是**确定性代码分支，不是 LLM 决策**。Gemini detect 高精度，大多数图每张只标 1 个真候选，无需再过 verify；少数冒 false positive 的图才需 verify 去伪存真。
> **历史**：原 `evaluate` / `calibrate` 迭代回路、`analyze` 节点、以及 LangGraph `StateGraph` 编排均已移除。

### 两阶段检测设计

- **阶段一（detect）**：VLM(gemini-3.5-flash) 看固定 256×256px 的 patch，判断"有没有 Waldo"。只用 `present` 二元信号（Gemini confidence 与 present 矛盾率 77%，不可用于排序/阈值）。
- **阶段二（verify）**：仅在多候选时触发；将全部候选区域从**原图**裁出（加 30% padding、最小 120px），**一次性**发给 `gemini-3.5-flash` 做横向单选「这几张里哪张才是 Waldo」（返回 `{choice, confidence, per_image}`）。强制相对比较，避免逐张判断的误检与条纹误导。

---

## State 定义（agent/state.py）

```python
class WaldoState(TypedDict):
    original_image_path: str       # 原图路径
    candidates: list               # [{patch_bbox, confidence, has_waldo, verified, ...}, ...]
    verified_result: list | None   # [x, y, w, h]（原图坐标），未找到则 None
```

`initial_state(image_path)`：只填 `original_image_path`，`candidates=[]`、`verified_result=None`；segment 节点负责把整图切成 candidates。
（旧字段 `focus_regions`/`iteration` 随 LangGraph 移除已一并删除；更早的 `grid_size`/`grid_rows`/`grid_cols` 随 analyze 删除已移除。）

---

## 各节点职责（agent/nodes/）

| 节点 | Provider | 输入 | 输出 | 说明 |
|------|----------|------|------|------|
| `segment`（入口） | — | `original_image_path` | `candidates`（仅含 patch_bbox 等几何字段） | 确定性固定尺寸滑窗切片，TILE_SIZE×TILE_SIZE、末块贴边、跳过 < 150px 块。不调 VLM |
| `detect` | gemini-3.5-flash | `candidates`, `original_image_path` | `candidates`（含 has_waldo / confidence） | 按 present(has_waldo) 二元信号过滤；confidence 透传但无判别意义、不用于排序 |
| `verify` | gemini-3.5-flash | 全部 present 候选（上限 VERIFY_MAX，仅多候选时触发） | `candidates`（verified 字段）+ `verified_result` | 裁出带 padding 的区域，横向单选唯一真 Waldo |
| `visualize` | — | `verified_result` / 最佳候选 | 标注图片路径 | 调用 `tools/visualize.py` 画红框 |

### 流水线组装（agent/pipeline.py）

- `_run_nodes(state)`：顺序运行 `segment → detect →（候选 > 1 才）verify → visualize`，每步 `state.update(delta)` 后 `yield (node, delta)`。
- `run_pipeline(image_path)`：跑完返回最终 state（CLI / 批量测试用）。
- `stream_pipeline(image_path)`：对外暴露 `(node, delta)` 生成器，供 service 层翻译成 SSE。
- 唯一分支是 `if len(state["candidates"]) > 1`（确定性，非 LLM 决策）；evaluate / calibrate / analyze / LangGraph 均已移除。

---

## VLM 抽象层（llm/vlm_client.py）

统一接口，四家 provider 同构实现，工厂函数一键切换：

```python
get_vlm_client(provider="claude")   # "claude" | "gpt4o" | "gemini" | "qwen"
```

| Provider | 类 | 默认 model | 备注 |
|----------|----|-----------|------|
| claude | `ClaudeVLMClient` | `claude-sonnet-4-6` | 可切换备用 |
| gpt4o | `GPT4oVLMClient` | `gpt-5.5` | 推理模型，现已不在主流程默认链上（detect/verify 均走 Gemini）；可切换备用 |
| gemini | `GeminiVLMClient` | `gemini-1.5-flash`（类默认）；**detect/verify 实际用 `gemini-3.5-flash`** | **detect + verify 默认**；detect 用 present 二元信号，verify 用横向单选 `select()` |
| qwen | `QwenVLMClient` | `qwen-vl-max` | 走 DashScope OpenAI 兼容接口；需 `DASHSCOPE_API_KEY` |

每个 client 实现 call / detect 两方法；Gemini 另实现 `select`（横向单选）：
- `call(image_path, prompt, max_tokens)` —— 发图 + 自定义 prompt，返回原始文本
- `detect(image_path) -> DetectResult(has_waldo, confidence, bbox, raw_response)`
- `select(image_paths) -> SelectResult(choice, confidence, per_image, raw_response)` —— 多图横向单选；`BaseVLMClient.select` 默认抛 `NotImplementedError`，仅 Gemini 覆盖

> 逐张确认的 `verify()` 方法与 `VERIFY_PROMPT` / `VerifyResult` 已于 2026-06-22 移除——verify 节点改用横向单选 `select()` 后不再需要逐张判断，provider 接口只留 call / detect / select。

`DETECT_PROMPT` / `SELECT_PROMPT` 定义在 `prompts.py`；`_extract_json()` 容错解析 markdown 代码块。

---

## 分割策略：确定性固定尺寸滑窗切片

- **segment**：对整图 `[0, 0, w, h]` 调 `vision/segment.py::tile_region`，按 `TILE_SIZE` 固定尺寸滑窗切片。
- **切片几何**：每轴起点 `0, stride, 2·stride, …`（`stride = round(TILE_SIZE×(1-TILE_OVERLAP))`），保留所有 `< length-TILE_SIZE` 的起点，再补一个**贴边起点 `length-TILE_SIZE`** → 每块恰好 `TILE_SIZE×TILE_SIZE`、全图无空洞（末排/列多重叠些）。`length ≤ TILE_SIZE` 时退化为单块。
- 跳过宽或高 < `MIN_PATCH_PX` 的块。`candidates` 每次进入 segment 重置。

> `tile_region` 仍接受任意 `region` 参数（不止全图），保留了未来做「粗到细」的几何能力；但当前流水线固定传整图，已不再有 `focus_regions` 状态字段。若将来要加粗到细回路，是在 `_run_nodes` 里加一段确定性循环（代码决定重切哪里），而非引入 LLM 决策。

---

## 关键可调参数

| 位置 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| `nodes/segment.py` | `TILE_SIZE` | 256 | 切片边长（px）；256 覆盖含小 Waldo 难图的安全下限，可调 |
| | `TILE_OVERLAP` | 0.15 | 相邻切片重叠比例，防 Waldo 骑在边界被切两半 |
| | `MIN_PATCH_PX` | 150 | 跳过过小 patch 的下限 |
| `nodes/detect.py` | `DETECT_MAX_TOKENS` | 4096 | detect 客户端 token 上限（Gemini 非推理不需要这么高，保留无害） |
| | `MIN_DETECT_PATCH_PX` | 150 | detect 跳过过小 patch 的下限 |
| | `MAX_CONCURRENT` | 10 | 并发数（付费 Tier 1 ~300 RPM 下提速；免费层 20/天会立刻打满，须降回 1 或换付费） |
| | `MAX_PATCHES_PER_ITER` | 80 | patch 硬上限，超出随机采样（256px 切片下通常远低于此） |
| | `MAX_RETRIES` / `RETRY_BASE_WAIT` | 4 / 15s | 429 限流指数退避：15→30→60→120 |
| `nodes/verify.py` | `VERIFY_MAX` | 12 | 送横向单选的候选数安全上限；全部 present 候选一次性比较，仅多候选路径触发 |
| | `SELECT_MAX_TOKENS` | 1024 | Gemini 横向单选响应 token 上限（含 per_image 数组） |
| | `PADDING_RATIO` / `MIN_VERIFY_SIZE` | 0.3 / 120px | 裁剪 padding 与最小尺寸 |

---

## 文件结构

```
WhereisWaldoAgent/
├── CLAUDE.md
├── main.py                      # 入口：python main.py [图片路径]
├── prompts.py                   # 集中存放所有 VLM 提示词
├── requirements.txt
├── .env                         # ANTHROPIC_API_KEY + OPENAI_API_KEY
├── agent/
│   ├── __init__.py              # 导出 run_pipeline / stream_pipeline
│   ├── state.py                 # WaldoState + initial_state
│   ├── pipeline.py              # run_pipeline / stream_pipeline / _run_nodes（纯函数编排）
│   └── nodes/
│       ├── __init__.py
│       ├── segment.py           # 入口：确定性固定尺寸滑窗切片为 patch
│       ├── detect.py            # gemini-3.5-flash 判断 patch 是否含 Waldo
│       ├── verify.py            # gemini-3.5-flash 横向单选候选（多候选时）
│       └── visualize.py         # 在原图标注结果
├── llm/                         # VLM 适配器层
│   ├── __init__.py
│   ├── vlm_client.py            # 兼容垫片（保留旧 import 路径）
│   ├── base.py                  # BaseVLMClient + _extract_json
│   ├── factory.py               # get_vlm_client 工厂
│   ├── results.py               # DetectResult / SelectResult
│   └── providers/               # Claude / GPT-4o / Gemini / Qwen 实现
├── vision/                      # 图像处理 + 切分
│   ├── __init__.py
│   ├── image_utils.py           # base64 编码、裁剪、保存
│   └── segment.py               # tile_region（固定尺寸滑窗切片）/ get_image_size
├── tools/                       # LangChain @tool
│   ├── __init__.py
│   ├── visualize.py             # 画 bbox（@tool）
│   └── crop_image.py            # 裁剪并保存（@tool）
├── tests/                       # 测试脚本
│   ├── check_claude.py          # Claude API key 验证
│   ├── check_gpt4o.py           # OpenAI API key 验证
│   ├── check_detect_prompt.py   # DETECT_PROMPT 召回率测试（images_withWaldo/）
│   └── ...
├── docs/                        # 设计文档
│   ├── agent_logic.md           # 完整逻辑链路文档
│   └── archive/                 # 已失效文档（描述已删除的 analyze 节点 / 旧 run issues）
├── original-images/             # 测试图片（1.jpg ~ 19.jpg, OIP.jpg）
├── images_withWaldo/            # 含 Waldo 的标注图片（召回率测试用）
└── outputs/
    ├── patches/                 # 各轮裁出的 patch
    ├── verify/                  # verify 的特写裁剪
    └── [basename]_result.jpg    # 最终标注图
```

---

## 已知 Bug（待修复）

> 详见 `docs/archive/run_issues_2026-06-09.md`（已归档）

- [x] **analyze 解析失败（已修复）**：根因 = `gpt-5.5` 是**推理模型**，`max_completion_tokens` 先被 reasoning token 消耗。128 预算被推理 100% 吃光（`finish_reason='length'`, content=`''`），降级到 fallback。实测 reasoning 达 ~232 token。修复：`ANALYZE_MAX_TOKENS` 提至 1024（对齐 detect/verify 默认值）。集成测试 `test_analyze_vlm_response_is_not_empty` 已转绿
- [x] **main.py grid_size 参数过时**：`run_agent` 调用已改为 `grid_size=1`，`initial_state` 默认值同步改为 1
- [x] **detect 截断导致系统性漏检**：`MAX_PATCHES_PER_ITER` 提升至 80，截断改为随机采样，避免系统性漏检右下角

---

## 待确认 / 优化方向

- [ ] **量化评测（头号优先）**：对 `original-images/` 建立 ground truth 标注 + IoU 命中率脚本。当前所有改动只能靠单图肉眼定性验证（密集难图常说不清谁是真 Waldo），这是检验 detect 召回 / verify 横向单选准确率 / bbox 精度的唯一可靠手段。
- [ ] **网络鲁棒性**：Gemini 调用偶发 504/503 超时（实测 2.jpg 跑挂 3 个 patch）；detect 已有 429 退避，但对 503/504/连接错误也应纳入重试。
- [ ] **计费类 429 快速失败**：detect 的重试把**所有** 429 当限流退避（15→30→60→120s），但「额度耗尽 / `credits are depleted` / free-tier 日配额超限」的 429 重试无用，导致每张图空转 ~27 分钟、并把结果污染成假阴性（2026-06-18 image 4 实测）。应识别计费类 429 **直接抛出不重试**，作为 `error` 事件冒到前端。
- [ ] **SSE 长连接心跳**：detect 节点跑数分钟期间 SSE 全程静默，长连接易被 Vite 代理 / 浏览器按空闲超时掐断 → 前端「连接中断」。可加周期性心跳事件。
- [ ] **TILE_SIZE 调参**：默认 256（覆盖小 Waldo 难图）；建立量化评测后正式比较 256 vs 384 的召回/速度权衡。
- [ ] **并发上限**：当前 `MAX_CONCURRENT=10`（付费 Tier 1 ~300 RPM）；如需更高吞吐可考虑用线程池并行各 detect 分支，注意 RPD ~1500/天 的上限。
- [x] **bbox 精修（已完成）**：detect 让 Gemini 在 patch 内回精确 bbox（`waldo_orig_bbox` 映射回原图），verify/单候选路径都能画紧框，无 bbox 时退化整块 patch。
- [x] **verify 抗误检（已完成）**：从 gpt-5.5 逐张判断改为 gemini-3.5-flash 横向单选，摆脱对不可靠 confidence 排序的依赖。

<!-- superpowers-zh:begin (do not edit between these markers) -->
# Superpowers-ZH 中文增强版

本项目已安装 superpowers-zh 技能框架（20 个 skills）。

## 核心规则

1. **收到任务时，先检查是否有匹配的 skill** — 哪怕只有 1% 的可能性也要检查
2. **设计先于编码** — 收到功能需求时，先用 brainstorming skill 做需求分析
3. **测试先于实现** — 写代码前先写测试（TDD）
4. **验证先于完成** — 声称完成前必须运行验证命令

## 可用 Skills

Skills 位于 `.claude/skills/` 目录，每个 skill 有独立的 `SKILL.md` 文件。

- **brainstorming**: 在任何创造性工作之前必须使用此技能——创建功能、构建组件、添加功能或修改行为。在实现之前先探索用户意图、需求和设计。
- **chinese-code-review**: 中文 review 沟通参考——话术模板、分级标注（必须修复/建议修改/仅供参考）、国内团队常见反模式应对。仅在用户显式 /chinese-code-review 时调用，不要根据上下文自动触发。
- **chinese-commit-conventions**: 中文 commit 与 changelog 配置参考——Conventional Commits 中文适配、commitlint/husky/commitizen 中文模板、conventional-changelog 中文配置。仅在用户显式 /chinese-commit-conventions 时调用，不要根据上下文自动触发。
- **chinese-documentation**: 中文文档排版参考——中英文空格、全半角标点、术语保留、链接格式、中文文案排版指北约定。仅在用户显式 /chinese-documentation 时调用，不要根据上下文自动触发。
- **chinese-git-workflow**: 国内 Git 平台配置参考——Gitee、Coding.net、极狐 GitLab、CNB 的 SSH/HTTPS/凭据/CI 接入差异与镜像同步配置。仅在用户显式 /chinese-git-workflow 时调用，不要根据上下文自动触发。
- **dispatching-parallel-agents**: 当面对 2 个以上可以独立进行、无共享状态或顺序依赖的任务时使用
- **executing-plans**: 当你有一份书面实现计划需要在单独的会话中执行，并设有审查检查点时使用
- **finishing-a-development-branch**: 当实现完成、所有测试通过、需要决定如何集成工作时使用——通过提供合并、PR 或清理等结构化选项来引导开发工作的收尾
- **mcp-builder**: MCP 服务器构建方法论 — 系统化构建生产级 MCP 工具，让 AI 助手连接外部能力
- **receiving-code-review**: 收到代码审查反馈后、实施建议之前使用，尤其当反馈不明确或技术上有疑问时——需要技术严谨性和验证，而非敷衍附和或盲目执行
- **requesting-code-review**: 完成任务、实现重要功能或合并前使用，用于验证工作成果是否符合要求
- **subagent-driven-development**: 当在当前会话中执行包含独立任务的实现计划时使用
- **systematic-debugging**: 遇到任何 bug、测试失败或异常行为时使用，在提出修复方案之前执行
- **test-driven-development**: 在实现任何功能或修复 bug 时使用，在编写实现代码之前
- **using-git-worktrees**: 当需要开始与当前工作区隔离的功能开发，或在执行实现计划之前使用——通过原生工具或 git worktree 回退机制确保隔离工作区存在
- **using-superpowers**: 在开始任何对话时使用——确立如何查找和使用技能，要求在任何响应（包括澄清性问题）之前调用 Skill 工具
- **verification-before-completion**: 在宣称工作完成、已修复或测试通过之前使用，在提交或创建 PR 之前——必须运行验证命令并确认输出后才能声称成功；始终用证据支撑断言
- **workflow-runner**: 在 Claude Code / OpenClaw / Cursor 中直接运行 agency-orchestrator YAML 工作流——无需 API key，使用当前会话的 LLM 作为执行引擎。当用户提供 .yaml 工作流文件或要求多角色协作完成任务时触发。
- **writing-plans**: 当你有规格说明或需求用于多步骤任务时使用，在动手写代码之前
- **writing-skills**: 当创建新技能、编辑现有技能或在部署前验证技能是否有效时使用

## 如何使用

当任务匹配某个 skill 时，使用 `Skill` 工具加载对应 skill 并严格遵循其流程。绝不要用 Read 工具读取 SKILL.md 文件。

如果你认为哪怕只有 1% 的可能性某个 skill 适用于你正在做的事情，你必须调用该 skill 检查。
<!-- superpowers-zh:end -->
