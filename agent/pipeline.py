"""检测流水线：segment → detect →（多候选才）verify → visualize 的纯函数编排。

历史上这里用 LangGraph 的 StateGraph 编排，但实际只有一条线性流 + 一个确定性
分支（候选 > 1 才走 verify），没有任何 LLM 自主控制流——本质是 workflow 而非
agent。故改为纯函数顺序调用 + 生成器流式，去掉 LangGraph 依赖。

- `run_pipeline(image_path)`  端到端跑完，返回最终 state（CLI / 批量测试用）。
- `stream_pipeline(image_path)`  逐节点产出 `(node, delta)`，供 service 层翻译成 SSE。
"""

from agent.state import WaldoState, initial_state
from agent.nodes import (
    segment_node,
    detect_node,
    verify_node,
    visualize_node,
)


def run_pipeline(image_path: str) -> WaldoState:
    """端到端运行检测流水线，返回最终 state。

    Args:
        image_path: 待检测图片路径。

    Returns:
        最终 WaldoState，verified_result 字段为检测结果。
    """
    state = initial_state(image_path)
    for _node, _delta in _run_nodes(state):
        pass
    return state


def stream_pipeline(image_path: str):
    """逐节点运行，每个节点完成后产出 `(node_name, delta)`。

    复刻原 LangGraph `graph.stream()` 的事件契约，供 service 层翻译成标准事件流。
    """
    state = initial_state(image_path)
    yield from _run_nodes(state)


def _run_nodes(state: WaldoState):
    """按序运行各节点，原地更新 `state`，每步产出 `(node, delta)`。

    路由：detect 后候选 > 1 才走 verify（多候选去伪存真），否则跳过直接 visualize。
    """
    delta = segment_node(state)
    state.update(delta)
    yield "segment", delta

    delta = detect_node(state)
    state.update(delta)
    yield "detect", delta

    if len(state["candidates"]) > 1:
        delta = verify_node(state)
        state.update(delta)
        yield "verify", delta

    delta = visualize_node(state)
    state.update(delta)
    yield "visualize", delta
