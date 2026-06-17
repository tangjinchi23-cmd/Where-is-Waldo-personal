"""入口脚本：指定图片路径，运行 WaldoAgent。"""

 # 大撒旦sjkdhIH  
import sys

# 优先从 .env 文件加载环境变量（有则加载，无则跳过）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent import run_agent


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "original-images/1.jpg"
    print(f"[main] Running WaldoAgent on: {image_path}")

    final_state = run_agent(image_path)

    result = final_state.get("verified_result")
    candidates = final_state.get("candidates") or []
    # detect 后单候选会跳过 verify（见 agent/graph.py:route_after_detect），
    # 此时 verified_result 为 None 但仍是「相信 detect 高精度」的有信心结果，
    # 不应与「verify 全否决/无候选」混为一谈。用 candidate 是否带 "verified"
    # 字段判断 verify 实际是否跑过。
    verify_ran = any("verified" in c for c in candidates)
    if result:
        print(f"[main] Waldo confirmed (verified) at bbox: {result}")
    elif candidates and not verify_ran:
        best = candidates[0]
        bbox = best.get("orig_bbox") or best.get("patch_bbox")
        print(f"[main] Waldo located (detect-only, verify skipped) at bbox: {bbox}")
    else:
        print("[main] Waldo not found.")


if __name__ == "__main__":
    main()
