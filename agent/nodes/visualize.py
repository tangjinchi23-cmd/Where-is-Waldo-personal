"""visualize 节点：将最终结果标注在原图上并保存。"""

import os

from agent.state import WaldoState
from tools.visualize import visualize_result

OUTPUT_DIR = "outputs"


def visualize_node(state: WaldoState) -> dict:
    """
    输入：original_image_path, verified_result, candidates
    输出：（无 state 更新，副作用为保存标注图片）

    策略：
    - 优先使用 verified_result
    - 若为 None（超迭代上限退出），取 candidates 中置信度最高的 bbox
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bbox = state["verified_result"]
    if bbox is None:
        bbox = _best_candidate_bbox(state["candidates"])

    if bbox is None:
        print("[visualize] No candidate found. Waldo not located.")
        return {}

    base = os.path.splitext(os.path.basename(state["original_image_path"]))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base}_result.jpg")

    saved = visualize_result.invoke({
        "image_path": state["original_image_path"],
        "bbox": bbox,
        "output_path": output_path,
    })
    print(f"[visualize] Result saved → {saved}  bbox={bbox}")
    return {}


def _best_candidate_bbox(candidates: list) -> list[int] | None:
    if not candidates:
        return None
    best = max(candidates, key=lambda c: c.get("verify_confidence") or c.get("confidence", 0.0))
    return best.get("orig_bbox") or best.get("patch_bbox")
