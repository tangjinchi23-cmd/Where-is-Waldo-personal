"""evaluate 节点：判断是否找到 Waldo，决定下一步路由。"""

from typing import Literal

from agent.state import WaldoState

VERIFY_CONFIDENCE_THRESHOLD = 0.85
Route = Literal["visualize", "calibrate"]


def evaluate_node(state: WaldoState) -> dict:
    """
    输入：candidates, verified_result, iteration, max_iterations
    输出：iteration + 1（仅递增迭代计数）

    路由决策由 route_after_evaluate 完成（LangGraph conditional edge）。
    """
    return {"iteration": state["iteration"] + 1}


def route_after_evaluate(state: WaldoState) -> Route:
    """
    路由逻辑（作为 LangGraph conditional_edge 的条件函数）：

    - verified_result 非空                → "visualize"（已确认找到）
    - 超过最大迭代次数                    → "visualize"（取最佳候选）
    - 否则                                → "calibrate"（继续搜索）
    """
    if state["verified_result"] is not None:
        return "visualize"

    if state["iteration"] >= state["max_iterations"]:
        return "visualize"

    top = _get_best_candidate(state["candidates"])
    if top and top.get("verify_confidence", 0.0) >= VERIFY_CONFIDENCE_THRESHOLD:
        return "visualize"

    return "calibrate"


def _get_best_candidate(candidates: list) -> dict | None:
    verified = [c for c in candidates if c.get("verified")]
    if verified:
        return max(verified, key=lambda c: c.get("verify_confidence", 0.0))
    if candidates:
        return max(candidates, key=lambda c: c.get("confidence", 0.0))
    return None
