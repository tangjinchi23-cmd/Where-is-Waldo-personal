import base64
import io
import os
from PIL import Image


def image_to_base64(image_path: str, max_size: int = 1568) -> str:
    """将图片读取并编码为 base64 字符串（自动缩放至 VLM 安全尺寸）。

    Args:
        image_path: 图片路径。
        max_size: 长边最大像素数。Claude/GPT-4o 建议不超过 1568。

    Returns:
        base64 编码字符串（不含 data: 前缀）。
    """
    img = Image.open(image_path).convert("RGB")
    img = _resize_keep_aspect(img, max_size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def save_patch(img: Image.Image, output_path: str) -> str:
    """保存 PIL Image 到指定路径，自动创建目录。"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img.save(output_path)
    return output_path


def crop_to_pil(image_path: str, bbox: list[int]) -> Image.Image:
    """从原图裁出 bbox 区域，返回 PIL Image（不保存）。

    Args:
        image_path: 原图路径。
        bbox: [x, y, w, h]。

    Returns:
        裁剪后的 PIL Image。
    """
    img = Image.open(image_path).convert("RGB")
    x, y, w, h = bbox
    img_w, img_h = img.size
    return img.crop((max(0, x), max(0, y), min(img_w, x + w), min(img_h, y + h)))


def _resize_keep_aspect(img: Image.Image, max_size: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)