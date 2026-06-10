"""LangGraph 图组装：定义节点、边，编译为可执行图。

当前为线性流水线：analyze → segment → detect → verify → visualize → END。
evaluate / calibrate 迭代回路已移除；如遇复杂案例需要二次细化，可在此重新接入。
"""

from langgraph.graph import StateGraph, END

from agent.state import WaldoState, initial_state
from agent.nodes import (
    analyze_node,
    segment_node,
    detect_node,
    verify_node,
    visualize_node,
)


def build_graph() -> StateGraph:
    """构造并编译 WaldoAgent 状态图。"""
    g = StateGraph(WaldoState)

    # ── 注册节点 ──────────────────────────────────────
    g.add_node("analyze",   analyze_node)
    g.add_node("segment",   segment_node)
    g.add_node("detect",    detect_node)
    g.add_node("verify",    verify_node)
    g.add_node("visualize", visualize_node)

    # ── 线性边 ────────────────────────────────────────
    g.set_entry_point("analyze")
    g.add_edge("analyze",   "segment")
    g.add_edge("segment",   "detect")
    g.add_edge("detect",    "verify")
    g.add_edge("verify",    "visualize")
    g.add_edge("visualize", END)

    return g.compile()


def run_agent(image_path: str, grid_size: int = 1) -> WaldoState:
    """端到端运行 WaldoAgent，返回最终 State。

    Args:
        image_path: 待检测图片路径。
        grid_size: 初始网格粒度（默认 1，即 analyze 的格子直接作为 patch）。

    Returns:
        最终 WaldoState，verified_result 字段为检测结果。
    """
    graph = build_graph()
    state = initial_state(image_path, grid_size=grid_size)
    return graph.invoke(state)
