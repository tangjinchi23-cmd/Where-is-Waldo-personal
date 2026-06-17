"""LangGraph 图组装：定义节点、边，编译为可执行图。

当前为线性流水线：segment → detect → verify → visualize → END。
evaluate / calibrate 迭代回路已移除；如遇复杂案例需要二次细化，可在此重新接入。
"""

from langgraph.graph import StateGraph, END

from agent.state import WaldoState, initial_state
from agent.nodes import (
    segment_node,
    detect_node,
    verify_node,
    visualize_node,
)


def build_graph() -> StateGraph:
    """构造并编译 WaldoAgent 状态图。"""
    g = StateGraph(WaldoState)

    # ── 注册节点 ──────────────────────────────────────
    g.add_node("segment",   segment_node)
    g.add_node("detect",    detect_node)
    g.add_node("verify",    verify_node)
    g.add_node("visualize", visualize_node)

    # ── 边 ────────────────────────────────────────────
    g.set_entry_point("segment")
    g.add_edge("segment",   "detect")
    # detect 后条件路由：单候选（或空）直接 visualize，跳过多余的 verify；
    # 多候选（少数会冒 false positive 的图）才走 verify 去伪存真。
    g.add_conditional_edges(
        "detect",
        route_after_detect,
        {"verify": "verify", "visualize": "visualize"},
    )
    g.add_edge("verify",    "visualize")
    g.add_edge("visualize", END)

    return g.compile()


def route_after_detect(state: WaldoState) -> str:
    """detect 后路由：候选 >1 走 verify，否则（单候选或空）直接 visualize。

    detect 改用 Gemini 后 confidence 失效、候选按 present(has_waldo) 过滤；
    大多数图只剩 1 个真候选，无需再过一遍 verify。
    """
    return "verify" if len(state["candidates"]) > 1 else "visualize"


def run_agent(image_path: str) -> WaldoState:
    """端到端运行 WaldoAgent，返回最终 State。

    Args:
        image_path: 待检测图片路径。

    Returns:
        最终 WaldoState，verified_result 字段为检测结果。
    """
    graph = build_graph()
    state = initial_state(image_path)
    return graph.invoke(state)
