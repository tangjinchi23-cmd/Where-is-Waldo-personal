# WhereisWaldoAgent

## 开发约定

- Git commit messages（标题和正文）必须全部用**英文**书写。

---

## 项目目标

在复杂的 Where's Waldo 图片中，通过 AI Agent 自动定位 Waldo 并返回原图坐标的 bbox。

> **当前状态**：已完成大规模重构。**全部节点（analyze / detect / verify）统一使用 GPT-4o**。核心设计锚点：**200×200px 是 VLM 可靠识别 Waldo 的最小 patch 尺寸**。

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

入口 `run_agent(image_path, grid_size=1)`，返回最终 `WaldoState`。

---

## 核心流程设计

### 整体流程图

当前为**线性流水线**，无迭代回路：

```
START
  ↓
[analyze]   ← VLM(GPT-4o) 看缩略图，根据复杂度推荐切割行列数（目标每格约 200×200px）
              输出：N×M 个 focus_regions（覆盖全图）+ grid_rows/grid_cols
  ↓
[segment]   ← 将 focus_regions 按 grid_size=1 切分为 patch（格子即 patch）
              跳过宽或高 < 150px 的区域
  ↓
[detect]    ← VLM(GPT-4o) 对每个 patch 判断是否含 Waldo，返回置信度
              跳过 < 150px 的 patch，过滤 confidence < 0.15，按置信度降序
  ↓
[verify]    ← 取 top-3 候选，从原图裁出（带 30% padding），VLM(GPT-4o) 二次确认
  ↓
[visualize] ← 优先用 verified_result；为空则取最佳候选 bbox，画红框
  ↓
 END
```

> **设计说明**：原 `evaluate` / `calibrate` 迭代回路已移除——评估认为其边际收益低（重搜已失败的同尺寸热区），且 evaluate 仅做 `iteration+1` 属冗余节点。如后续测试遇到复杂案例确需二次细化，再按需在 verify 后接入新节点。

### 两阶段检测设计

- **阶段一（detect）**：VLM 看 ~200×200px 的 patch，判断"有没有 Waldo"，返回置信度。prompt 明确要求"宁可误报，不要漏检"（减少 false negative）。实验验证 GPT-4o 在此任务上召回率优于 Claude。
- **阶段二（verify）**：将候选区域从**原图**裁出（加 30% padding、最小 120px），发给 GPT-4o 二次确认"这真的是 Waldo 吗"。

---

## LangGraph State 定义（agent/state.py）

```python
class WaldoState(TypedDict):
    original_image_path: str       # 原图路径
    grid_size: int                 # 切分粒度（线性流水线下恒为 1）
    focus_regions: list            # [[x, y, w, h], ...]，当前重点搜索区域（原图坐标）
    grid_rows: int                 # analyze 输出：VLM 推荐行数
    grid_cols: int                 # analyze 输出：VLM 推荐列数
    candidates: list               # [{patch_bbox, confidence, crop_path, verified, ...}, ...]
    verified_result: list | None   # [x, y, w, h]（原图坐标），未找到则 None
    iteration: int                 # 恒为 0；detect/verify 用于命名输出文件
```

`initial_state(image_path, grid_size=1)`：以全图作为初始 focus_region，由 analyze 替换。

---

## 各节点职责（agent/nodes/）

| 节点 | Provider | 输入 | 输出 | 说明 |
|------|----------|------|------|------|
| `analyze` | GPT-4o | `original_image_path` | `focus_regions`, `grid_rows`, `grid_cols` | VLM 推荐行列数，全图均匀切 N×M 格（每格约 200×200px） |
| `segment` | — | `focus_regions`, `grid_size` | `candidates`（每轮重置） | 按 grid_size 切分，跳过 < 150px 区域，带 12% overlap |
| `detect` | GPT-4o | `candidates`, `original_image_path` | `candidates`（含 confidence / has_waldo） | 跳过 < 150px patch，过滤低置信度，按置信度降序 |
| `verify` | GPT-4o | `candidates` 中 top-3 | `candidates`（verified 字段）+ `verified_result` | 裁出带 padding 的区域 VLM 二次确认 |
| `visualize` | — | `verified_result` / 最佳候选 | 标注图片路径 | 调用 `tools/visualize.py` 画红框 |

### 图组装（agent/graph.py）

- 入口：`analyze`
- 线性边：`analyze → segment → detect → verify → visualize → END`
- 无回路、无条件路由（evaluate / calibrate 已移除）

---

## VLM 抽象层（llm/vlm_client.py）

统一接口，四家 provider 同构实现，工厂函数一键切换：

```python
get_vlm_client(provider="claude")   # "claude" | "gpt4o" | "gemini" | "qwen"
```

| Provider | 类 | 默认 model | 备注 |
|----------|----|-----------|------|
| claude | `ClaudeVLMClient` | `claude-sonnet-4-6` | 可切换备用 |
| gpt4o | `GPT4oVLMClient` | `gpt-5.5` | **全部节点默认**；实验验证召回率更高 |
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
| `nodes/analyze.py` | `THUMBNAIL_MAX` | 900 | 发给 VLM 的缩略图最大边长 |
| | `ANALYZE_MAX_TOKENS` | 128 | VLM 响应 token 上限（过低会截断 JSON 触发 fallback） |
| `nodes/segment.py` | `MIN_PATCH_PX` | 150 | 跳过过小 patch 的下限 |
| `nodes/detect.py` | `DETECT_CONFIDENCE_THRESHOLD` | 0.15 | 低于此值的 patch 丢弃 |
| | `MIN_DETECT_PATCH_PX` | 150 | detect 跳过过小 patch 的下限 |
| | `MAX_CONCURRENT` | 1 | 并发数（50 req/min 限制下保守串行） |
| | `MAX_PATCHES_PER_ITER` | 80 | patch 硬上限，超出随机采样 |
| | `MAX_RETRIES` / `RETRY_BASE_WAIT` | 4 / 15s | 429 限流指数退避：15→30→60→120 |
| `nodes/verify.py` | `TOP_K` | 3 | 送验证的候选数 |
| | `PADDING_RATIO` / `MIN_VERIFY_SIZE` | 0.3 / 120px | 裁剪 padding 与最小尺寸 |

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
│       ├── verify.py            # GPT-4o 二次确认候选
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

- [ ] **analyze 解析失败（仍存在）**：`ANALYZE_MAX_TOKENS` 已提到 128，但 2026-06-10 实测仍收到**空字符串** `''`，降级到 fallback 6×10。token 上限非根因。下一步：打印完整原始响应 + HTTP 状态定位（疑为限流/model id 问题），加重试逻辑
- [x] **main.py grid_size 参数过时**：`run_agent` 调用已改为 `grid_size=1`，`initial_state` 默认值同步改为 1
- [x] **detect 截断导致系统性漏检**：`MAX_PATCHES_PER_ITER` 提升至 80，截断改为随机采样，避免系统性漏检右下角

---

## 待确认 / 优化方向

- [ ] **量化评测**：对 `original-images/` 建立 ground truth 标注 + IoU 命中率脚本，每次改动可量化验证效果
- [ ] **并发上限**：当前 `MAX_CONCURRENT=1` 受 50 req/min 限制；系统稳定后可考虑提额 + 并行 detect 分支（LangGraph 已预留接口）
- [ ] **置信度阈值调参**：`DETECT_CONFIDENCE_THRESHOLD=0.15`、`VERIFY_CONFIDENCE_THRESHOLD=0.85` 需用实测数据调优
- [ ] **verify 精修 bbox**：目前 verify 只做 true/false 判断，可让 VLM 同时回传精确 bbox 坐标

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
