# WhereisWaldoAgent 最小前端

基于 Streamlit 的结果预览页面（**架构占位版**）。当前仅预览 `outputs/` 中已有的检测结果，不实时调用 agent —— 实时检测将在将来的 FastAPI 版接入。

> 本目录与 `service/` 都只是架构占位，重点在分层与边界，功能将随真实需求迭代。

## 运行

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

浏览器打开提示的本地地址（默认 http://localhost:8501）。左侧选择图片，右侧查看红框标注结果；无结果的图会提示"尚未检测"。

## 架构

```
frontend/app.py            UI 层（将来换成 React）
        │  仅依赖
        ▼
service/waldo_service.py   共享服务层（将来被 FastAPI 复用）
        │  读取
        ▼
original-images/ + outputs/
```

UI 只依赖 `service.waldo_service.list_cases()`，与检测逻辑解耦。过渡路径：`service/` 不变 → 新增 `api/`（FastAPI 调 service）→ `frontend/` 换 React 调 API。
