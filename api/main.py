"""FastAPI 入口：把 service.run_detection 暴露成 SSE，并提供选图/上传/静态文件。

运行：uvicorn api.main:app --reload --port 8000
"""

import json
from pathlib import Path

# 优先从 .env 加载环境变量（GOOGLE_API_KEY 等）；缺失则跳过
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
