"""WhereisWaldoAgent 共享服务层（架构占位版）。

定位：前端与检测逻辑之间的解耦边界。当前只提供"已有结果"的查询能力，
供 Streamlit 前端消费；将来 FastAPI 可直接复用 list_cases / get_case 包成
REST 接口，而无需改动前端契约。

注意：本文件是架构占位，配对/过滤逻辑保持最小够用，后续真实需求再细化。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# 模块级导入，便于测试 monkeypatch（agent 层无回环依赖 service）
from agent.graph import build_graph
from agent.state import initial_state

# 相对本文件解析项目根：service/ 的上一级即项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "original-images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class WaldoCase:
    """前端与 service 之间的数据契约。"""

    name: str                  # 原图文件名去后缀，如 "1"
    image_path: str            # 原图绝对路径
    result_path: str | None    # outputs/{name}_result.jpg；无则 None
    has_result: bool


def _is_source_image(p: Path) -> bool:
    """是否为可用作 case 的源图：文件 + 图片后缀 + 非 *_annotated 标注图。"""
    return (
        p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not p.stem.endswith("_annotated")
    )


def list_cases(
    images_dir: Path = IMAGES_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
) -> list[WaldoCase]:
    """扫描 images_dir 下的源图，与 outputs_dir/{name}_result.jpg 配对。

    目录不存在时返回空列表，不抛异常。结果按 name 升序。
    """
    if not images_dir.exists():
        return []

    cases: list[WaldoCase] = []
    for p in sorted(images_dir.iterdir(), key=lambda x: x.name):
        if not _is_source_image(p):
            continue
        name = p.stem
        result = outputs_dir / f"{name}_result.jpg"
        has_result = result.is_file()
        cases.append(
            WaldoCase(
                name=name,
                image_path=str(p),
                result_path=str(result) if has_result else None,
                has_result=has_result,
            )
        )
    return cases


def get_case(
    name: str,
    images_dir: Path = IMAGES_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
) -> WaldoCase | None:
    """按 name 返回单个 case；不存在返回 None。"""
    for case in list_cases(images_dir=images_dir, outputs_dir=outputs_dir):
        if case.name == name:
            return case
    return None


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
