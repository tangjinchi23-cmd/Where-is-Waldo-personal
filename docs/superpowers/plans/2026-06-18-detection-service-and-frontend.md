# 检测服务化 + React/FastAPI 前端 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 把 `main.py` 的一次性 `run_agent` 测试流程服务化，配一个能选图/上传图、SSE 实时看流水线运行状态与结果的 React 前端。

**架构：** 三层单向依赖——`service`（纯 Python 生成器，把 `graph.stream()` 翻译成标准事件）← `api`（FastAPI，SSE + 静态文件）← `frontend`（React+Vite，EventSource 消费）。agent 层不动。

**技术栈：** Python / FastAPI / uvicorn / LangGraph `graph.stream` / React 18 / Vite / vitest / pytest。

参考规格：`docs/superpowers/specs/2026-06-18-detection-service-and-frontend-design.md`

---

## 文件结构

**创建：**
- `api/__init__.py` — 包标识（空）
- `api/main.py` — FastAPI 应用：`/api/cases`、`/api/upload`、`/api/detect`(SSE)、静态挂载
- `frontend/package.json`、`frontend/vite.config.js`、`frontend/index.html` — 前端工程配置
- `frontend/src/main.jsx` — React 入口
- `frontend/src/pipeline.js` — 纯函数事件归约器（UI 状态机，可单测）
- `frontend/src/api.js` — fetch + EventSource 封装
- `frontend/src/App.jsx` — 整体布局与状态管理
- `frontend/src/components/ImagePicker.jsx`、`PipelineProgress.jsx`、`CandidateGallery.jsx`、`ResultView.jsx`
- `frontend/test/pipeline.test.js` — pipeline.js 的 vitest 测试
- `tests/test_api.py` — API 层 pytest

**修改：**
- `service/waldo_service.py` — 保留 `list_cases/get_case`，新增 `UPLOADS_DIR`、`resolve_image`、`run_detection`
- `tests/test_waldo_service.py` — 追加 `resolve_image` / `run_detection` 测试
- `requirements.txt` — 加 `fastapi`、`uvicorn[standard]`、`python-multipart`

**删除（Streamlit 占位）：**
- `frontend/app.py`、`frontend/requirements.txt`、`frontend/README.md`

---

## 任务 1：service 层——resolve_image + run_detection

**文件：**
- 修改：`service/waldo_service.py`
- 测试：`tests/test_waldo_service.py`

- [ ] **步骤 1：在 `tests/test_waldo_service.py` 末尾追加失败测试**

```python
# ── resolve_image ────────────────────────────────────────────────
from service.waldo_service import resolve_image, run_detection


def test_resolve_image_prefers_original_then_uploads(tmp_path):
    images = tmp_path / "original-images"
    uploads = tmp_path / "uploads"
    images.mkdir()
    uploads.mkdir()
    (images / "1.jpg").write_bytes(b"x")
    (uploads / "2.png").write_bytes(b"x")

    assert resolve_image("1", images_dir=images, uploads_dir=uploads) == images / "1.jpg"
    assert resolve_image("2", images_dir=images, uploads_dir=uploads) == uploads / "2.png"
    assert resolve_image("zzz", images_dir=images, uploads_dir=uploads) is None


# ── run_detection ────────────────────────────────────────────────
class _FakeGraph:
    def __init__(self, updates):
        self._updates = updates

    def stream(self, state):
        yield from self._updates


def _patch_graph(monkeypatch, updates):
    monkeypatch.setattr("service.waldo_service.initial_state", lambda p: {"original_image_path": p})
    monkeypatch.setattr("service.waldo_service.build_graph", lambda: _FakeGraph(updates))


def test_run_detection_full_pipeline_with_verify(tmp_path, monkeypatch):
    updates = [
        {"segment": {"candidates": [{}, {}, {}]}},
        {"detect": {"candidates": [
            {"crop_path": "a", "confidence": 0.9, "has_waldo": True, "orig_bbox": [1, 2, 3, 4]},
            {"crop_path": "b", "confidence": 0.8, "has_waldo": True, "orig_bbox": [5, 6, 7, 8]},
        ]}},
        {"verify": {"candidates": [
            {"verified": True, "verify_crop_path": "va", "verify_looks_waldo": True, "orig_bbox": [1, 2, 3, 4]},
            {"verified": False, "verify_crop_path": "vb", "verify_looks_waldo": False, "orig_bbox": [5, 6, 7, 8]},
        ], "verified_result": [1, 2, 3, 4]}},
        {"visualize": {}},
    ]
    _patch_graph(monkeypatch, updates)

    events = list(run_detection("foo.jpg"))
    assert [e["stage"] for e in events] == ["segment", "detect", "verify", "done"]
    assert events[0]["patches"] == 3
    assert events[1]["count"] == 2
    assert events[2]["ran"] is True and events[2]["choice"] == 0
    assert events[-1]["found"] is True
    assert events[-1]["verify_ran"] is True
    assert events[-1]["bbox"] == [1, 2, 3, 4]


def test_run_detection_skips_verify_single_candidate(tmp_path, monkeypatch):
    updates = [
        {"segment": {"candidates": [{}]}},
        {"detect": {"candidates": [
            {"crop_path": "a", "confidence": 0.9, "has_waldo": True, "orig_bbox": [1, 2, 3, 4]},
        ]}},
        {"visualize": {}},
    ]
    _patch_graph(monkeypatch, updates)

    events = list(run_detection("foo.jpg"))
    assert [e["stage"] for e in events] == ["segment", "detect", "verify", "done"]
    assert events[2]["ran"] is False
    assert events[-1]["found"] is True
    assert events[-1]["verify_ran"] is False
    assert events[-1]["bbox"] == [1, 2, 3, 4]


def test_run_detection_emits_error_event(monkeypatch):
    class _BoomGraph:
        def stream(self, state):
            raise RuntimeError("no api key")

    monkeypatch.setattr("service.waldo_service.initial_state", lambda p: {"original_image_path": p})
    monkeypatch.setattr("service.waldo_service.build_graph", lambda: _BoomGraph())

    events = list(run_detection("foo.jpg"))
    assert events[-1]["stage"] == "error"
    assert "no api key" in events[-1]["message"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_waldo_service.py -v`
预期：FAIL，`ImportError: cannot import name 'resolve_image'`（及 run_detection）。

- [ ] **步骤 3：实现 service 新增项**

在 `service/waldo_service.py` 顶部 import 区下方加入模块级 import 与常量；保持 `WaldoCase / list_cases / get_case` 不变，在文件末尾追加新函数。

文件顶部（`from pathlib import Path` 之后）补：

```python
from typing import Iterator

# 模块级导入，便于测试 monkeypatch（agent 层无回环依赖 service）
from agent.graph import build_graph
from agent.state import initial_state
```

常量区（`OUTPUTS_DIR = ...` 之后）补：

```python
UPLOADS_DIR = PROJECT_ROOT / "uploads"
```

文件末尾追加：

```python
def resolve_image(
    name: str,
    images_dir: Path = IMAGES_DIR,
    uploads_dir: Path = UPLOADS_DIR,
) -> Path | None:
    """按去后缀文件名解析图片路径：original-images 优先，uploads 兜底；都没有返回 None。"""
    for d in (images_dir, uploads_dir):
        for ext in IMAGE_EXTS:
            p = d / f"{name}{ext}"
            if p.is_file():
                return p
    return None


def _cand_brief(c: dict) -> dict:
    """把候选裁成前端需要的精简形状。"""
    return {
        "crop_path": c.get("verify_crop_path") or c.get("crop_path"),
        "confidence": c.get("confidence"),
        "has_waldo": c.get("has_waldo"),
        "verified": c.get("verified", False),
        "verify_looks_waldo": c.get("verify_looks_waldo"),
        "orig_bbox": c.get("orig_bbox") or c.get("patch_bbox"),
    }


def _build_done(image_path: str, candidates: list, verify_ran: bool, verified_result) -> dict:
    """复刻 main.py 三态判断，组装 done 事件。"""
    name = Path(image_path).stem
    result_file = OUTPUTS_DIR / f"{name}_result.jpg"
    result_path = str(result_file) if result_file.is_file() else None

    if verified_result:
        return {"stage": "done", "found": True, "verify_ran": True,
                "bbox": verified_result, "result_path": result_path}
    if candidates and not verify_ran:
        best = candidates[0]
        bbox = best.get("orig_bbox") or best.get("patch_bbox")
        return {"stage": "done", "found": True, "verify_ran": False,
                "bbox": bbox, "result_path": result_path}
    return {"stage": "done", "found": False, "verify_ran": verify_ran,
            "bbox": None, "result_path": result_path}


def run_detection(image_path: str) -> Iterator[dict]:
    """把 graph.stream 的逐节点增量翻译成标准化事件流。

    事件 stage ∈ {segment, detect, verify, done, error}；详见设计文档。
    所有 *_path 为相对项目根的真实路径，由 API 层转 URL。
    """
    graph = build_graph()
    state = initial_state(image_path)
    candidates: list = []
    verify_ran = False
    verified_result = None
    try:
        for update in graph.stream(state):
            for node, delta in update.items():
                if node == "segment":
                    yield {"stage": "segment", "patches": len(delta.get("candidates", []))}
                elif node == "detect":
                    candidates = delta.get("candidates", [])
                    yield {"stage": "detect", "count": len(candidates),
                           "candidates": [_cand_brief(c) for c in candidates]}
                elif node == "verify":
                    verify_ran = True
                    candidates = delta.get("candidates", candidates)
                    verified_result = delta.get("verified_result")
                    choice = next((i for i, c in enumerate(candidates) if c.get("verified")), -1)
                    yield {"stage": "verify", "ran": True, "choice": choice,
                           "per_image": [c.get("verify_looks_waldo") for c in candidates],
                           "candidates": [_cand_brief(c) for c in candidates]}
                # visualize 节点返回 {}，无需事件
        if not verify_ran:
            yield {"stage": "verify", "ran": False}
        yield _build_done(image_path, candidates, verify_ran, verified_result)
    except Exception as exc:  # 缺 key / 503 / 超时等统一转 error 事件
        yield {"stage": "error", "message": str(exc)}
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_waldo_service.py -v`
预期：PASS（含原有 3 个契约测试 + 新增 4 个）。

- [ ] **步骤 5：Commit**

```bash
git add service/waldo_service.py tests/test_waldo_service.py
git commit -m "feat(service): add resolve_image and run_detection event stream"
```

---

## 任务 2：API 层——FastAPI 应用

**文件：**
- 创建：`api/__init__.py`、`api/main.py`
- 修改：`requirements.txt`
- 测试：`tests/test_api.py`

- [ ] **步骤 1：补依赖并安装**

`requirements.txt` 在 `pillow>=10.0.0` 之后追加：

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
```

运行：`pip install -r requirements.txt`
预期：fastapi / uvicorn / python-multipart 安装成功。

- [ ] **步骤 2：编写失败测试 `tests/test_api.py`**

```python
"""API 层测试：用 TestClient + monkeypatch service，不打真实 VLM。"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from service.waldo_service import WaldoCase, IMAGES_DIR, OUTPUTS_DIR


def test_cases_endpoint_maps_paths_to_urls(monkeypatch):
    fake = [WaldoCase(
        name="1",
        image_path=str(IMAGES_DIR / "1.jpg"),
        result_path=str(OUTPUTS_DIR / "1_result.jpg"),
        has_result=True,
    )]
    monkeypatch.setattr("api.main.list_cases", lambda: fake)

    client = TestClient(app)
    r = client.get("/api/cases")
    assert r.status_code == 200
    body = r.json()
    assert body[0]["name"] == "1"
    assert body[0]["image_url"] == "/static/original-images/1.jpg"
    assert body[0]["result_url"] == "/static/outputs/1_result.jpg"
    assert body[0]["has_result"] is True


def test_upload_rejects_bad_extension():
    client = TestClient(app)
    r = client.post("/api/upload", files={"file": ("x.txt", b"hi", "text/plain")})
    assert r.status_code == 400


def test_detect_404_when_image_missing(monkeypatch):
    monkeypatch.setattr("api.main.resolve_image", lambda name: None)
    client = TestClient(app)
    r = client.get("/api/detect", params={"name": "nope"})
    assert r.status_code == 404


def test_detect_streams_sse_frames(monkeypatch):
    monkeypatch.setattr("api.main.resolve_image", lambda name: Path("fake.jpg"))

    def fake_run(path):
        yield {"stage": "segment", "patches": 3}
        yield {"stage": "done", "found": False, "verify_ran": False, "bbox": None, "result_path": None}

    monkeypatch.setattr("api.main.run_detection", fake_run)

    client = TestClient(app)
    with client.stream("GET", "/api/detect", params={"name": "foo"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    assert '"stage": "segment"' in text
    assert '"stage": "done"' in text
    assert text.count("data:") == 2
```

- [ ] **步骤 3：运行测试验证失败**

运行：`pytest tests/test_api.py -v`
预期：FAIL，`ModuleNotFoundError: No module named 'api'`。

- [ ] **步骤 4：实现 `api/__init__.py` 与 `api/main.py`**

`api/__init__.py`：空文件。

`api/main.py`：

```python
"""FastAPI 入口：把 service.run_detection 暴露成 SSE，并提供选图/上传/静态文件。

运行：uvicorn api.main:app --reload --port 8000
"""

import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from service.waldo_service import (
    IMAGE_EXTS,
    IMAGES_DIR,
    OUTPUTS_DIR,
    UPLOADS_DIR,
    list_cases,
    resolve_image,
    run_detection,
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="WhereisWaldo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态目录：前端按 URL 直接取原图/结果图/patch 裁剪图
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
app.mount("/static/original-images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/static/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/static/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

_MOUNTS = [
    (IMAGES_DIR, "/static/original-images"),
    (OUTPUTS_DIR, "/static/outputs"),
    (UPLOADS_DIR, "/static/uploads"),
]


def _to_url(path: str | None) -> str | None:
    """把项目内绝对/相对路径映射成 /static/... URL；无法映射返回 None。"""
    if not path:
        return None
    p = Path(path).resolve()
    for base, mount in _MOUNTS:
        try:
            rel = p.relative_to(base.resolve())
            return f"{mount}/{rel.as_posix()}"
        except ValueError:
            continue
    return None


def _map_event(ev: dict) -> dict:
    """把 service 事件里的 *_path 字段补成前端用的 *_url。"""
    ev = dict(ev)
    if "result_path" in ev:
        ev["result_url"] = _to_url(ev.pop("result_path"))
    if "candidates" in ev:
        ev["candidates"] = [
            {**c, "crop_url": _to_url(c.get("crop_path"))} for c in ev["candidates"]
        ]
    return ev


@app.get("/api/cases")
def get_cases():
    return [
        {
            "name": c.name,
            "image_url": _to_url(c.image_path),
            "has_result": c.has_result,
            "result_url": _to_url(c.result_path),
        }
        for c in list_cases()
    ]


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (>20MB)")
    UPLOADS_DIR.mkdir(exist_ok=True)
    dest = UPLOADS_DIR / Path(file.filename).name
    dest.write_bytes(data)
    return {"name": dest.stem, "image_url": _to_url(str(dest))}


@app.get("/api/detect")
def detect(name: str):
    image = resolve_image(name)
    if image is None:
        raise HTTPException(status_code=404, detail=f"Image not found: {name}")

    def event_stream():
        for event in run_detection(str(image)):
            yield f"data: {json.dumps(_map_event(event))}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **步骤 5：运行测试验证通过**

运行：`pytest tests/test_api.py -v`
预期：PASS（4 个测试）。

- [ ] **步骤 6：Commit**

```bash
git add api/__init__.py api/main.py tests/test_api.py requirements.txt
git commit -m "feat(api): add FastAPI SSE detect endpoint with cases/upload"
```

---

## 任务 3：前端工程脚手架 + pipeline 状态机

**文件：**
- 创建：`frontend/package.json`、`frontend/vite.config.js`、`frontend/index.html`、`frontend/src/main.jsx`、`frontend/src/pipeline.js`、`frontend/test/pipeline.test.js`

- [ ] **步骤 1：写工程配置文件**

`frontend/package.json`：

```json
{
  "name": "whereiswaldo-frontend",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest run"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.1",
    "vite": "^5.4.0",
    "vitest": "^2.0.5"
  }
}
```

`frontend/vite.config.js`（dev 代理到 :8000，vitest 用 node 环境）：

```javascript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/static": "http://localhost:8000",
    },
  },
  test: {
    environment: "node",
  },
});
```

`frontend/index.html`：

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Where's Waldo Agent</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

`frontend/src/main.jsx`：

```jsx
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

- [ ] **步骤 2：写失败测试 `frontend/test/pipeline.test.js`**

```javascript
import { describe, it, expect } from "vitest";
import { INITIAL, applyEvent } from "../src/pipeline.js";

describe("applyEvent", () => {
  it("segment marks segment done and detect running", () => {
    const s = applyEvent(INITIAL, { stage: "segment", patches: 5 });
    expect(s.stages.segment).toBe("done");
    expect(s.stages.detect).toBe("running");
    expect(s.patches).toBe(5);
  });

  it("detect stores candidates and starts verify", () => {
    const ev = { stage: "detect", count: 2, candidates: [{ crop_url: "/a" }, { crop_url: "/b" }] };
    const s = applyEvent(INITIAL, ev);
    expect(s.stages.detect).toBe("done");
    expect(s.stages.verify).toBe("running");
    expect(s.candidates).toHaveLength(2);
  });

  it("verify ran=false marks skipped", () => {
    const s = applyEvent(INITIAL, { stage: "verify", ran: false });
    expect(s.stages.verify).toBe("done");
    expect(s.verify.ran).toBe(false);
  });

  it("done sets result", () => {
    const s = applyEvent(INITIAL, {
      stage: "done", found: true, verify_ran: false, bbox: [1, 2, 3, 4], result_url: "/x",
    });
    expect(s.stages.done).toBe("done");
    expect(s.result.found).toBe(true);
    expect(s.result.bbox).toEqual([1, 2, 3, 4]);
  });

  it("error flags running stage as error", () => {
    const mid = applyEvent(INITIAL, { stage: "segment", patches: 1 }); // detect=running
    const s = applyEvent(mid, { stage: "error", message: "boom" });
    expect(s.error).toBe("boom");
    expect(s.stages.detect).toBe("error");
  });
});
```

- [ ] **步骤 3：运行测试验证失败**

运行：`cd frontend && npm install && npm test`
预期：FAIL，找不到 `../src/pipeline.js`。

- [ ] **步骤 4：实现 `frontend/src/pipeline.js`**

```javascript
// 纯函数事件归约器：把 SSE 事件序列折叠成 UI 状态。无副作用，便于单测。

export const INITIAL = {
  stages: { segment: "pending", detect: "pending", verify: "pending", done: "pending" },
  patches: null,
  count: null,
  candidates: [],
  verify: null,
  result: null,
  error: null,
};

export function applyEvent(state, ev) {
  const s = { ...state, stages: { ...state.stages } };
  switch (ev.stage) {
    case "segment":
      s.stages.segment = "done";
      s.stages.detect = "running";
      s.patches = ev.patches;
      break;
    case "detect":
      s.stages.detect = "done";
      s.stages.verify = "running";
      s.count = ev.count;
      s.candidates = ev.candidates || [];
      break;
    case "verify":
      s.stages.verify = "done";
      s.stages.done = "running";
      s.verify = { ran: ev.ran, choice: ev.choice ?? -1 };
      if (ev.candidates) s.candidates = ev.candidates;
      break;
    case "done":
      s.stages.done = "done";
      s.result = {
        found: ev.found,
        verifyRan: ev.verify_ran,
        bbox: ev.bbox,
        resultUrl: ev.result_url,
      };
      break;
    case "error":
      s.error = ev.message;
      for (const k of Object.keys(s.stages)) {
        if (s.stages[k] === "running") s.stages[k] = "error";
      }
      break;
    default:
      break;
  }
  return s;
}
```

- [ ] **步骤 5：运行测试验证通过**

运行：`cd frontend && npm test`
预期：PASS（5 个用例）。

- [ ] **步骤 6：Commit**

```bash
git add frontend/package.json frontend/vite.config.js frontend/index.html frontend/src/main.jsx frontend/src/pipeline.js frontend/test/pipeline.test.js
git commit -m "feat(frontend): scaffold Vite project and pipeline state reducer"
```

---

## 任务 4：前端 api.js + 组件 + App

**文件：**
- 创建：`frontend/src/api.js`、`frontend/src/App.jsx`、`frontend/src/components/ImagePicker.jsx`、`PipelineProgress.jsx`、`CandidateGallery.jsx`、`ResultView.jsx`

> 本任务为 UI 组装，逻辑核心（pipeline reducer）已在任务 3 单测覆盖；这里按手动验证（任务 5）把关。

- [ ] **步骤 1：实现 `frontend/src/api.js`**

```javascript
// 后端通信封装。dev 下 /api、/static 由 vite 代理到 :8000。
const BASE = import.meta.env.VITE_API_BASE || "";

export async function fetchCases() {
  const r = await fetch(`${BASE}/api/cases`);
  if (!r.ok) throw new Error("加载图片列表失败");
  return r.json();
}

export async function uploadImage(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${BASE}/api/upload`, { method: "POST", body: fd });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail || "上传失败");
  }
  return r.json();
}

// 打开 SSE，逐事件回调；done/error 自动关闭。返回取消函数。
export function subscribeDetect(name, { onEvent, onError }) {
  const es = new EventSource(`${BASE}/api/detect?name=${encodeURIComponent(name)}`);
  es.onmessage = (e) => {
    const data = JSON.parse(e.data);
    onEvent(data);
    if (data.stage === "done" || data.stage === "error") es.close();
  };
  es.onerror = () => {
    es.close();
    onError && onError();
  };
  return () => es.close();
}

export function staticUrl(u) {
  return u ? `${BASE}${u}` : null;
}
```

- [ ] **步骤 2：实现 `frontend/src/components/PipelineProgress.jsx`**

```jsx
const LABELS = { segment: "切片", detect: "检测", verify: "验证", done: "完成" };
const COLOR = { pending: "#ccc", running: "#f5a623", done: "#2ecc71", error: "#e74c3c" };

export default function PipelineProgress({ stages, patches, count, verify }) {
  return (
    <div style={{ display: "flex", gap: 12, alignItems: "center", margin: "12px 0" }}>
      {Object.keys(LABELS).map((k) => (
        <div key={k} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 12, height: 12, borderRadius: "50%", background: COLOR[stages[k]] }} />
          <span>{LABELS[k]}</span>
          {k === "segment" && patches != null && <small>({patches} patch)</small>}
          {k === "detect" && count != null && <small>({count} 候选)</small>}
          {k === "verify" && verify && <small>({verify.ran ? `选中 #${verify.choice}` : "跳过"})</small>}
        </div>
      ))}
    </div>
  );
}
```

- [ ] **步骤 3：实现 `frontend/src/components/CandidateGallery.jsx`**

```jsx
import { staticUrl } from "../api.js";

export default function CandidateGallery({ candidates }) {
  if (!candidates.length) return null;
  return (
    <div>
      <h3>检测候选（{candidates.length}）</h3>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
        {candidates.map((c, i) => (
          <div
            key={i}
            style={{
              border: c.verified ? "3px solid #2ecc71" : "1px solid #ddd",
              padding: 4, borderRadius: 4, width: 140,
            }}
          >
            {c.crop_url && (
              <img src={staticUrl(c.crop_url)} alt={`候选 ${i}`} style={{ width: "100%", display: "block" }} />
            )}
            <small>
              #{i} conf={c.confidence != null ? c.confidence.toFixed(2) : "—"}
              {c.verified && " ✅"}
            </small>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **步骤 4：实现 `frontend/src/components/ResultView.jsx`**

```jsx
import { staticUrl } from "../api.js";

export default function ResultView({ imageUrl, result }) {
  return (
    <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
      <div>
        <h3>原图</h3>
        {imageUrl && <img src={staticUrl(imageUrl)} alt="原图" style={{ maxWidth: 480, display: "block" }} />}
      </div>
      <div>
        <h3>结果</h3>
        {!result && <p>尚未检测。</p>}
        {result && !result.found && <p>未找到 Waldo。</p>}
        {result && result.found && (
          <>
            {result.resultUrl ? (
              <img src={staticUrl(result.resultUrl)} alt="结果" style={{ maxWidth: 480, display: "block" }} />
            ) : (
              <p>（结果图未生成）</p>
            )}
            <p>
              bbox: {JSON.stringify(result.bbox)}
              {result.verifyRan ? "（verify 确认）" : "（detect 单候选，跳过 verify）"}
            </p>
          </>
        )}
      </div>
    </div>
  );
}
```

- [ ] **步骤 5：实现 `frontend/src/components/ImagePicker.jsx`**

```jsx
import { useState } from "react";

export default function ImagePicker({ cases, selected, onSelect, onUpload, disabled }) {
  const [busy, setBusy] = useState(false);

  async function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    setBusy(true);
    try {
      await onUpload(file);
    } finally {
      setBusy(false);
      e.target.value = "";
    }
  }

  return (
    <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
      <select value={selected || ""} onChange={(e) => onSelect(e.target.value)} disabled={disabled}>
        <option value="" disabled>选择图片…</option>
        {cases.map((c) => (
          <option key={c.name} value={c.name}>
            {c.name} {c.has_result ? "✅" : ""}
          </option>
        ))}
      </select>
      <label style={{ cursor: "pointer", color: "#2a6" }}>
        {busy ? "上传中…" : "上传新图"}
        <input type="file" accept="image/*" onChange={handleFile} disabled={disabled || busy} style={{ display: "none" }} />
      </label>
    </div>
  );
}
```

- [ ] **步骤 6：实现 `frontend/src/App.jsx`**

```jsx
import { useEffect, useState } from "react";
import { fetchCases, uploadImage, subscribeDetect, staticUrl } from "./api.js";
import { INITIAL, applyEvent } from "./pipeline.js";
import ImagePicker from "./components/ImagePicker.jsx";
import PipelineProgress from "./components/PipelineProgress.jsx";
import CandidateGallery from "./components/CandidateGallery.jsx";
import ResultView from "./components/ResultView.jsx";

export default function App() {
  const [cases, setCases] = useState([]);
  const [selected, setSelected] = useState("");
  const [imageUrl, setImageUrl] = useState(null);
  const [pipe, setPipe] = useState(INITIAL);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    fetchCases().then(setCases).catch((e) => console.error(e));
  }, []);

  function pickCase(name) {
    setSelected(name);
    const c = cases.find((x) => x.name === name);
    setImageUrl(c ? c.image_url : null);
    setPipe(INITIAL);
  }

  async function handleUpload(file) {
    const { name, image_url } = await uploadImage(file);
    setCases((prev) => [...prev.filter((c) => c.name !== name), { name, image_url, has_result: false }]);
    setSelected(name);
    setImageUrl(image_url);
    setPipe(INITIAL);
  }

  function runDetect() {
    if (!selected) return;
    setPipe(INITIAL);
    setRunning(true);
    subscribeDetect(selected, {
      onEvent: (ev) => {
        setPipe((prev) => applyEvent(prev, ev));
        if (ev.stage === "done" || ev.stage === "error") setRunning(false);
      },
      onError: () => {
        setPipe((prev) => applyEvent(prev, { stage: "error", message: "连接中断" }));
        setRunning(false);
      },
    });
  }

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: 24, maxWidth: 1100, margin: "0 auto" }}>
      <h1>Where's Waldo Agent</h1>
      <ImagePicker
        cases={cases}
        selected={selected}
        onSelect={pickCase}
        onUpload={handleUpload}
        disabled={running}
      />
      <button onClick={runDetect} disabled={!selected || running} style={{ marginTop: 12 }}>
        {running ? "检测中…" : "运行检测"}
      </button>

      <PipelineProgress stages={pipe.stages} patches={pipe.patches} count={pipe.count} verify={pipe.verify} />
      {pipe.error && <p style={{ color: "#e74c3c" }}>错误：{pipe.error}</p>}

      <CandidateGallery candidates={pipe.candidates} />
      <ResultView imageUrl={imageUrl} result={pipe.result} />
    </div>
  );
}
```

- [ ] **步骤 7：构建验证（无类型/语法错误）**

运行：`cd frontend && npm run build`
预期：构建成功，生成 `frontend/dist/`。

- [ ] **步骤 8：Commit**

```bash
git add frontend/src/api.js frontend/src/App.jsx frontend/src/components
git commit -m "feat(frontend): add api client, pipeline UI components and App"
```

---

## 任务 5：清理占位文件 + 端到端手动验证

**文件：**
- 删除：`frontend/app.py`、`frontend/requirements.txt`、`frontend/README.md`
- 创建：`frontend/README.md`（React 版说明）
- 修改：`.gitignore`（忽略 `uploads/`、`frontend/node_modules/`、`frontend/dist/`）

- [ ] **步骤 1：删除 Streamlit 占位文件**

```bash
git rm frontend/app.py frontend/requirements.txt frontend/README.md
```

- [ ] **步骤 2：写新的 `frontend/README.md`**

```markdown
# WhereisWaldoAgent 前端（React + Vite）

浏览器里选图/上传图 → 触发 `run_agent` 检测 → SSE 实时看 segment→detect→verify→done 流水线与结果。

## 运行

```bash
# 1) 后端（项目根）
uvicorn api.main:app --reload --port 8000

# 2) 前端
cd frontend
npm install
npm run dev        # 默认 http://localhost:5173，/api 与 /static 代理到 :8000
```

## 测试

```bash
npm test           # vitest：pipeline 状态机单测
```

## 架构

```
frontend/ (React)  ──/api, /static──>  api/main.py (FastAPI)  ──>  service/waldo_service.py  ──>  agent/
```
```

- [ ] **步骤 3：更新 `.gitignore`**

在文件末尾追加（若条目已存在则跳过）：

```
uploads/
frontend/node_modules/
frontend/dist/
```

- [ ] **步骤 4：跑全量后端测试**

运行：`pytest tests/test_waldo_service.py tests/test_api.py -v`
预期：全部 PASS。

- [ ] **步骤 5：端到端手动验证（需 `GOOGLE_API_KEY`）**

1. 终端 A：`uvicorn api.main:app --reload --port 8000`
2. 终端 B：`cd frontend && npm run dev`
3. 浏览器开 `http://localhost:5173`，从下拉选 `1`（或上传一张图），点「运行检测」。
4. 观察：四盏灯依次 segment→detect→verify→done 变绿；detect 后铺出候选裁剪图；done 后右侧显示红框结果图与 bbox。
5. 异常路径自检：临时改名 `.env` 里的 key 重启后端 → 前端应收到 error 事件、进度灯停在 detect 并标红。

预期：与上述一致。把实际观察（成功/失败 + 关键输出）记录到 commit 说明或回报。

- [ ] **步骤 6：Commit**

```bash
git add -A
git commit -m "chore(frontend): replace Streamlit placeholder with React docs and gitignore"
```

---

## 自检结果

**1. 规格覆盖度：**
- service `resolve_image` / `run_detection` / 三态 done → 任务 1 ✅
- API `/api/cases` / `/api/upload`（含非法文件 400）/ `/api/detect` SSE / 静态挂载 / CORS → 任务 2 ✅
- 进度推送 SSE + LangGraph stream → 任务 1（事件流）+ 任务 2（StreamingResponse）+ 任务 4（EventSource）✅
- 前端 ImagePicker/PipelineProgress/CandidateGallery/ResultView + 上传/选图/逐节点亮灯 → 任务 3、4 ✅
- verify 跳过统一 `ran:false` → 任务 1 步骤 3 + 任务 3 reducer ✅
- 错误处理（缺 key/503/超时/非法上传）→ service error 事件（任务 1）、上传 400（任务 2）、前端 error 灯（任务 3/4）✅
- 测试策略（service monkeypatch / api TestClient / 前端 vitest）→ 任务 1、2、3 ✅
- 删除 Streamlit 占位 → 任务 5 ✅
- 依赖变更 fastapi/uvicorn/python-multipart → 任务 2 ✅

**2. 占位符扫描：** 无 TODO/待定；每个代码步骤均含完整代码与精确命令。

**3. 类型一致性：** service 事件字段（stage/patches/count/candidates/crop_path/orig_bbox/ran/choice/found/verify_ran/bbox/result_path）↔ API `_map_event`（crop_url/result_url）↔ 前端 reducer（stages/candidates/verify/result）↔ 组件 props 全程一致；`subscribeDetect`、`applyEvent`、`INITIAL`、`staticUrl` 在定义与调用处签名一致。
