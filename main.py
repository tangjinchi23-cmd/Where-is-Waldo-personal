"""入口脚本：指定图片路径，运行 WaldoAgent。"""


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

    final_state = run_agent(image_path, grid_size=1)
    # This is a test to the new branch
    result = final_state.get("verified_result")
    if result:
        print(f"[main] Waldo confirmed at bbox: {result}")
    else:
        print("[main] Waldo not confirmed — best-guess bbox saved to outputs/ (if any candidates exist)")


if __name__ == "__main__":
    main()
