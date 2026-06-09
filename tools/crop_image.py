import os
from PIL import Image
from langchain_core.tools import tool


@tool
def crop_image(image_path: str, bbox: list[int], output_path: str) -> str:
    """按 bbox 裁剪原图并保存到指定路径。

    Args:
        image_path: 原图路径。
        bbox: 裁剪区域，格式 [x, y, width, height]（左上角坐标 + 宽高）。
        output_path: 裁剪结果保存路径。

    Returns:
        保存结果的文件路径。
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    x, y, w, h = bbox
    img_w, img_h = img.size
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    cropped = img.crop((x1, y1, x2, y2))
    cropped.save(output_path)
    return output_path