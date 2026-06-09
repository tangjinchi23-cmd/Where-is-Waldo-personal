from PIL import Image, ImageDraw
from langchain_core.tools import tool


@tool
def visualize_result(image_path: str, bbox: list[int], output_path: str = "bbox_result.jpg") -> str:
    """在图片上绘制边界框并保存。

    Args:
        image_path: 输入图片的文件路径。
        bbox: 边界框，格式为 [x, y, width, height]（左上角坐标 + 宽高）。
        output_path: 结果保存路径。

    Returns:
        保存结果的文件路径。
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    x, y, w, h = bbox
    rect = [x, y, x + w, y + h]   # 转成 PIL 需要的 [x_min, y_min, x_max, y_max]

    draw.rectangle(rect, outline="red", width=2)
    img.save(output_path)
    return output_path