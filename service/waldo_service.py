"""WhereisWaldoAgent 共享服务层（架构占位版）。

定位：前端与检测逻辑之间的解耦边界。当前只提供"已有结果"的查询能力，
供 Streamlit 前端消费；将来 FastAPI 可直接复用 list_cases / get_case 包成
REST 接口，而无需改动前端契约。

注意：本文件是架构占位，配对/过滤逻辑保持最小够用，后续真实需求再细化。
"""

from dataclasses import dataclass
from pathlib import Path

# 相对本文件解析项目根：service/ 的上一级即项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "original-images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
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
