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

- `src/pipeline.js`：纯函数事件归约器（SSE 事件 → UI 状态），逻辑核心，有单测。
- `src/api.js`：`fetchCases` / `uploadImage` / `subscribeDetect`（EventSource）。
- `src/components/`：ImagePicker / PipelineProgress / CandidateGallery / ResultView。
