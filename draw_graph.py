"""生成 WaldoAgent 的 LangGraph 结构图，保存为 PNG。"""

import sys # d大大大大大
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_PATH = "agent_graph.png"


def main():
    from agent.graph import build_graph

    print("Building graph...")
    graph = build_graph()

    # 方案一：调用 Mermaid Ink 在线 API 生成 PNG（需要联网，无需本地依赖）
    try:
        from langchain_core.runnables.graph import MermaidDrawMethod

        png_bytes = graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
            background_color="white",
            padding=16,
        )
        Path(OUTPUT_PATH).write_bytes(png_bytes)
        print(f"Saved → {OUTPUT_PATH}")
        return
    except Exception as e:
        print(f"Mermaid API failed: {e}")

    # 方案二：输出 Mermaid 源码，供手动渲染
    mermaid_src = graph.get_graph().draw_mermaid()
    mermaid_path = "agent_graph.md"
    Path(mermaid_path).write_text(f"```mermaid\n{mermaid_src}\n```", encoding="utf-8")
    print(f"Mermaid source saved → {mermaid_path}")
    print("\n--- Mermaid source ---")
    print(mermaid_src)


if __name__ == "__main__":
    main()
