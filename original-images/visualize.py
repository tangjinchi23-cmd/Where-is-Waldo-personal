#!/usr/bin/env python3
"""
Waldo bbox 可视化工具
---------------------
把 Claude 输出的 bbox 画到图片上，方便你检查准确度并校正。

支持两种坐标格式：
  - "pixel"  : [x_min, y_min, x_max, y_max]  （像素，绝对坐标）
  - "yolo"   : [x_center, y_center, w, h]    （归一化 0~1）

依赖：
  pip install pillow

用法示例见文件底部 __main__。
"""

from PIL import Image, ImageDraw, ImageFont
import os


def _to_pixel_xyxy(box, fmt, img_w, img_h):
    """把任意格式的 box 统一转换成像素 [x_min, y_min, x_max, y_max]。"""
    if fmt == "pixel":
        x_min, y_min, x_max, y_max = box
    elif fmt == "yolo":
        xc, yc, w, h = box
        x_min = (xc - w / 2) * img_w
        y_min = (yc - h / 2) * img_h
        x_max = (xc + w / 2) * img_w
        y_max = (yc + h / 2) * img_h
    else:
        raise ValueError(f"未知格式: {fmt}，应为 'pixel' 或 'yolo'")
    return [int(round(v)) for v in (x_min, y_min, x_max, y_max)]


def draw_boxes(
    image_path,
    boxes,
    fmt="pixel",
    labels=None,
    output_path=None,
    color="red",
    width=4,
    show=False,
):
    """
    在图片上绘制一个或多个 bbox。

    参数：
        image_path  : 图片路径
        boxes       : 单个 box（list/tuple）或 box 列表（list of list）
        fmt         : "pixel" 或 "yolo"
        labels      : 与 boxes 对应的标签文字列表（可选）
        output_path : 保存路径；为 None 时默认在原文件名后加 _annotated
        color       : 框颜色
        width       : 框线宽
        show        : 是否调用系统查看器打开（本地有 GUI 时有用）

    返回：
        保存后的输出路径
    """
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    # 统一成 [[...], [...]] 形式
    if boxes and isinstance(boxes[0], (int, float)):
        boxes = [boxes]
    if labels is None:
        labels = [f"#{i}" for i in range(len(boxes))]

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(14, img_h // 60))
    except Exception:
        font = ImageFont.load_default()

    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = _to_pixel_xyxy(box, fmt, img_w, img_h)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)

        # 画一个带底色的标签
        text = str(label)
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y_min - th - 4)
        draw.rectangle([x_min, ty, x_min + tw + 6, ty + th + 4], fill=color)
        draw.text((x_min + 3, ty + 2), text, fill="white", font=font)

    if output_path is None:
        root, ext = os.path.splitext(image_path)
        output_path = f"{root}_annotated{ext}"

    img.save(output_path)
    print(f"图片尺寸: {img_w} x {img_h}")
    print(f"已保存标注图: {output_path}")

    if show:
        img.show()

    return output_path


if __name__ == "__main__":
    # ===== 使用示例 =====
    # 1) 像素坐标格式 [x_min, y_min, x_max, y_max]
    draw_boxes(
        image_path="18.jpg",          # 改成你的图片路径
        boxes=[1250, 70, 1300, 120],        # Claude 给你的 bbox
        fmt="pixel",
        labels=["Waldo"],
        color="red",
        width=2,
        show=False,                          # 本地有桌面环境可设 True
    )

    # 2) 多个框 + YOLO 归一化格式（取消注释使用）
    # draw_boxes(
    #     image_path="waldo_02.jpg",
    #     boxes=[[0.43, 0.61, 0.03, 0.05],
    #            [0.71, 0.22, 0.02, 0.04]],
    #     fmt="yolo",
    #     labels=["Waldo", "Wizard"],
    # )