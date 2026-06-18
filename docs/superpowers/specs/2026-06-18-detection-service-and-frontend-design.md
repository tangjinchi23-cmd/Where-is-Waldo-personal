# 检测服务化 + 前端系统 设计

**日期**：2026-06-18
**状态**：已批准设计，待编写实现计划

## 背景与目标

当前 `main.py` 只能命令行跑一次 `run_agent`，结果打印到终端、标注图落到 `outputs/`。目标是把这条测试流程**服务化**，并配一个**前端系统**，让用户在浏览器里选图/上传图、触发检测、实时看到流水线运行状态与最终结果。

项目 `frontend/`、`service/` 下已有的文件均为**架构占位**（Streamlit 预览页 + 只读 service），本次全部替换/重写，不保留。

## 关键决策（已与用户确认）

| 决策点 | 选择 |
|--------|------|
| 前端技术栈 | React (Vite) + FastAPI 前后端分离 |
| 输入来源 | 既能选 `original-images/` 现有图，也能上传新图 |
| 展示粒度 | 最终结果 + 关键中间态（detect 候选、verify 横向单选结果） |
| 进度推送 | SSE（Server-Sent Events）+ LangGraph `graph.stream()` 原生流式 |
| 占位文件 | 全部删除/重写，不保留 Streamlit 版 |

## 架构总览

三层，单向依赖，边界清晰：

```
[React SPA]  ──HTTP/SSE──>  [FastAPI: api/]  ──调用──>  [service 层]  ──>  run_agent(graph.stream)
  frontend/                    api/main.py            service/waldo_service.py     agent/
```

- **API 层不碰检测逻辑**，只做 HTTP / SSE / 静态文件。
- **service 层不碰 HTTP**，只暴露纯 Python 生成器，可被 FastAPI 或未来别的入口（CLI、批处理）复用。
- **agent 层不变**，service 通过 `graph.stream()` 消费它现有的节点输出。

## ① service 层（重写 `service/waldo_service.py`）

保留并精简 `list_cases / get_case`（用于列现有图），新增：

### `resolve_image(name: str) -> Path | None`
按 `name`（去后缀文件名）解析图片绝对路径：先查 `original-images/`，再查 `uploads/`，都没有返回 `None`。

### `run_detection(image_path: str) -> Iterator[dict]`
核心。把 `run_agent` 的 `graph.invoke(state)` 换成 `graph.stream(state)`（`stream_mode="updates"`，每个节点产出后 yield 一次该节点的状态增量）。内部累积 state，把每个节点增量翻译成标准化事件并 yield：

```python
{"stage": "segment", "patches": 35}
{"stage": "detect",  "count": 3,
 "candidates": [{"crop_path": "...", "confidence": 0.9, "has_waldo": true, "orig_bbox": [x,y,w,h]}, ...]}
{"stage": "verify",  "ran": true, "choice": 2, "per_image": [...],
 "candidates": [...含 verified / verify_confidence / verify_crop_path]}
# verify 被路由跳过（单候选/空）→ 统一 emit {"stage":"verify","ran":false}，前端据此显示「跳过 verify」
{"stage": "done", "found": true, "verify_ran": true,
 "bbox": [x,y,w,h], "result_path": "outputs/<name>_result.jpg"}
```

- `done` 事件**复刻 `main.py` 现有三态判断**，保证语义一致：
  - `verified_result` 非空 → `found=true, verify_ran=true`，bbox = verified_result。
  - 无 verify（单候选路径，靠 `candidates` 里没有 `verify_confidence` 判断）但有候选 → `found=true, verify_ran=false`，bbox = `candidates[0].orig_bbox`（退化取 `patch_bbox`）。
  - 否则 → `found=false`，bbox = None。
- 异常（缺 API key、Gemini 503/超时等）→ yield `{"stage": "error", "message": "..."}` 并结束。
- 所有 `*_path` 字段是相对项目根的路径，API 层负责转成静态 URL。

## ② API 层（新建 `api/main.py`，FastAPI）

| 方法 | 路径 | 作用 |
|------|------|------|
| `GET` | `/api/cases` | 列 `original-images/` 现有图（复用 `list_cases`）：`[{name, image_url, has_result, result_url}]` |
| `POST` | `/api/upload` | 上传新图存到 `uploads/`，校验后缀（jpg/jpeg/png）与大小上限，返回 `{name, image_url}` |
| `GET` | `/api/detect?name=<name>` | **SSE 长连接**：迭代 `service.run_detection`，每事件 `yield f"data: {json}\n\n"`，`media_type="text/event-stream"` |
| `mount` | `/static/original-images`、`/static/outputs`、`/static/uploads` | `StaticFiles` 挂载三目录，前端直接按 URL 取原图/结果图/patch 裁剪图 |

设计要点：
- SSE 受浏览器 `EventSource` 限制必须是 GET，所以「选/传图」与「开始检测」分两步：先 `POST /upload` 或从 `/cases` 选图拿到 `name`，再 `GET /detect?name=`。
- `graph.stream` 是同步阻塞生成器；FastAPI 用 `def`（非 async）生成器返回 `StreamingResponse` 时会在 threadpool 里跑，不阻塞事件循环。
- service 产出的相对路径在 API 层映射成 `/static/...` URL 后再下发前端。
- 允许跨域（开发期前端 dev server 与 API 不同端口），加 `CORSMiddleware`。

## ③ 前端（React + Vite，重写 `frontend/`）

删除 Streamlit 版 `app.py / requirements.txt / README.md`，换 React 工程：

```
frontend/
├── package.json / vite.config.js / index.html
└── src/
    ├── api.js              # fetch /cases /upload + EventSource(/detect)
    ├── App.jsx             # 整体布局 + 状态管理
    └── components/
        ├── ImagePicker.jsx       # 选现有图 + 上传新图
        ├── PipelineProgress.jsx  # segment→detect→verify→done 四盏灯逐节点亮
        ├── CandidateGallery.jsx  # detect 候选 patch 裁剪图 + confidence / has_waldo
        └── ResultView.jsx        # 原图 | 红框结果图 + 最终 bbox 坐标
```

交互流：
1. 进入页面 → `GET /api/cases` 填充图片选择器。
2. 选现有图，或上传新图（`POST /api/upload`）→ 得到 `name`。
3. 点「运行检测」→ 开 `EventSource(/api/detect?name=)`。
4. 逐事件更新：`segment` 亮第一盏灯 + 显示 patch 数 → `detect` 铺候选画廊 → `verify` 标出被选中候选（或显示「单候选跳过 verify」）→ `done` 显示红框结果图与最终 bbox。
5. `error` 事件 → 进度灯停在出错节点，弹出错误信息。

## 错误处理

- 缺 `GOOGLE_API_KEY`、Gemini 503/超时、上传非法文件 → service 内捕获，经 SSE `error` 事件传到前端。
- 前端收到 `error` → 进度停在当前节点并提示，不崩页。
- 上传：限制后缀为 jpg/jpeg/png，设大小上限（如 20MB）；非法直接 400。
- `resolve_image` 返回 None（name 不存在）→ API 返回 404 / SSE error。

## 测试策略（TDD，先写测试）

- **service**：
  - `run_detection`：用 monkeypatch 替掉 `build_graph`（返回假图，不打真实 API），断言事件序列（segment→detect→[verify]→done）与 `done` 三态分支正确。
  - `resolve_image`：original-images 优先于 uploads；不存在返回 None。
  - `list_cases`：保留现有行为（已有覆盖可沿用）。
- **api**：`TestClient` 测 `/api/cases`、`/api/upload`（含非法文件 400）；`/api/detect` 用 mock 的 service 生成器断言 SSE 帧格式与顺序。
- **前端**：仅对 `api.js` 的 SSE 帧解析做一个 vitest 轻测；UI 交互手动验证（YAGNI，不引入 e2e）。

## 依赖变更

- `requirements.txt` 新增：`fastapi`、`uvicorn[standard]`、`python-multipart`（上传）。
- 前端：独立 npm 工程（React + Vite + vitest），不进 `requirements.txt`。

## 运行方式（实现后）

```bash
# 后端
uvicorn api.main:app --reload --port 8000
# 前端
cd frontend && npm install && npm run dev   # 默认 5173，dev 代理到 :8000
```

## 非目标（YAGNI，本期不做）

- 运行中途取消任务（SSE 单向，需另加 cancel 接口，本期不做）。
- 并发跑多张图、任务队列。
- 用户鉴权、多用户隔离。
- 量化评测 / IoU 命中率（属另一条独立路线）。
