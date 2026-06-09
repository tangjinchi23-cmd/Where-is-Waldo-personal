"""LangGraph 图组装：定义节点、边、条件路由，编译为可执行图。"""

from langgraph.graph import StateGraph, END

from agent.state import WaldoState, initial_state
from agent.nodes import (
    analyze_node,
    segment_node,
    detect_node,
    verify_node,
    evaluate_node,
    route_after_evaluate,
    calibrate_node,
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
    g.add_node("evaluate",  evaluate_node)
    g.add_node("calibrate", calibrate_node)
    g.add_node("visualize", visualize_node)

    # ── 线性边 ────────────────────────────────────────
    g.set_entry_point("analyze")
    g.add_edge("analyze",   "segment")
    g.add_edge("segment",   "detect")
    g.add_edge("detect",    "verify")
    g.add_edge("verify",    "evaluate")
    g.add_edge("calibrate", "segment")   # 迭代回路
    g.add_edge("visualize", END)

    # ── 条件路由（evaluate → visualize | calibrate） ──
    g.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "visualize": "visualize",
            "calibrate": "calibrate",
        },
    )

    return g.compile()


def run_agent(image_path: str, max_iterations: int = 5, grid_size: int = 3) -> WaldoState:
    """端到端运行 WaldoAgent，返回最终 State。

    Args:
        image_path: 待检测图片路径。
        max_iterations: 最大迭代次数。
        grid_size: 初始网格粒度。

    Returns:
        最终 WaldoState，verified_result 字段为检测结果。
    """
    graph = build_graph()
    state = initial_state(image_path, max_iterations=max_iterations, grid_size=grid_size)
    return graph.invoke(state)
