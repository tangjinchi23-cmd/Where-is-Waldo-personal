
from PIL import Image
from langchain_core.tools import tool


def segment_region(
    region: list[int],
    grid_size: int,
    image_size: tuple[int, int],
    min_patch_size: int = 64,
    overlap: float = 0.12,
) -> list[dict]:
    """将一个区域按 grid_size×grid_size 切分，返回带 overlap 的 patch 列表（原图坐标）。

    Args:
        region: [x, y, w, h]，原图坐标的目标区域。
        grid_size: 切分粒度，生成 grid_size×grid_size 个 patch。
        image_size: (img_width, img_height)，用于边界裁剪。
        min_patch_size: patch 宽/高最小像素数，低于此值不再切分。
        overlap: 相邻 patch 重叠比例（相对于 stride），防止 Waldo 被切在边界。

    Returns:
        patch 字典列表，每项：
            {
                "bbox": [x, y, w, h],   # 原图坐标（含 overlap 扩展）
                "row": int,
                "col": int,
            }
    """
    rx, ry, rw, rh = region
    img_w, img_h = image_size

    # stride：相邻 patch 起点之间的间距
    stride_w = rw / grid_size
    stride_h = rh / grid_size

    if stride_w < min_patch_size or stride_h < min_patch_size:
        actual = max(1, min(
            int(rw // min_patch_size),
            int(rh // min_patch_size),
            grid_size,
        ))
        stride_w = rw / actual
        stride_h = rh / actual
        grid_size = actual

    # 每个 patch 向右/下额外延伸 overlap × stride
    extend_w = stride_w * overlap
    extend_h = stride_h * overlap

    patches = []
    for row in range(grid_size):
        for col in range(grid_size):
            x1 = int(round(rx + col * stride_w))
            y1 = int(round(ry + row * stride_h))
            x2 = min(img_w, int(round(rx + (col + 1) * stride_w + extend_w)))
            y2 = min(img_h, int(round(ry + (row + 1) * stride_h + extend_h)))
            patches.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "row": row,
                "col": col,
            })
    return patches


def segment_all_regions(
    focus_regions: list[list[int]],
    region_grid_sizes: dict[int, int],
    default_grid_size: int,
    image_size: tuple[int, int],
    min_patch_size: int = 64,
    overlap: float = 0.12,
) -> list[dict]:
    """对所有 focus_regions 分别切分，合并成完整 patch 列表。

    Args:
        focus_regions: 区域列表，每项 [x, y, w, h]。
        region_grid_sizes: {region_idx: grid_size}，缺省用 default_grid_size。
        default_grid_size: 默认切分粒度。
        image_size: (img_width, img_height)。
        min_patch_size: 最小 patch 像素尺寸。
        overlap: 相邻 patch 重叠比例。

    Returns:
        所有区域的 patch 列表，每项额外含 region_idx 字段。
    """
    all_patches = []
    for idx, region in enumerate(focus_regions):
        grid_size = region_grid_sizes.get(idx, default_grid_size)
        patches = segment_region(region, grid_size, image_size, min_patch_size, overlap)
        for p in patches:
            p["region_idx"] = idx
        all_patches.extend(patches)
    return all_patches


@tool
def get_image_size(image_path: str) -> list[int]:
    """获取图片的宽高（像素）。

    Args:
        image_path: 图片文件路径。

    Returns:
        [width, height]。
    """
    img = Image.open(image_path)
    return list(img.size)