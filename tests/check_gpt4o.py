"""GPT 视觉模型对照实验：推理 vs 非推理模型在 detect 任务上的准确率/速度对比。

背景：Bug 1 定位发现 gpt-5.5 是推理模型，每次调用先消耗大量 reasoning token，
detect 阶段 60+ patch 串行因此极慢。detect 只是"有没有 Waldo"的二分判断，未必需要
重推理。本脚本在带标注的快速验证集 images_quicktests/ 上对比两个模型的：

    - 准确率 / 召回率 / 误报率（对比 ground truth）
    - 平均单次 detect 耗时

数据集标签 GROUND_TRUTH：按 img_{i}.png 顺序，1=含 Waldo，0=不含。

    gpt-5.5        — 推理模型（当前 detect 默认）
    gpt-5.4-mini   — 非推理模型（候选，预期更快）

直接在 PyCharm 右键 Run，或：python tests/check_gpt4o.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

MODELS = ["gpt-5.5", "gpt-5.4-mini"]

# 快速验证集：img_0.png ~ img_6.png，标签按序对应
GROUND_TRUTH = "1100111"
DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "images_quicktests"
)


def _check_key_and_sdk() -> bool:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[FAIL] OPENAI_API_KEY 未设置，请检查 .env 文件")
        return False
    print(f"[OK]   Key 已读取: ...{api_key[-6:]}")
    try:
        import openai
        print(f"[OK]   openai SDK 版本: {openai.__version__}")
    except ImportError:
        print("[FAIL] openai 未安装，请执行: pip install openai")
        return False
    return True


def _load_dataset() -> list[tuple[str, int]]:
    """返回 [(image_path, label), ...]，label 为 0/1。"""
    samples = []
    for i, ch in enumerate(GROUND_TRUTH):
        path = os.path.join(DATASET_DIR, f"img_{i}.png")
        if not os.path.exists(path):
            print(f"[WARN] 缺失图片，跳过: {path}")
            continue
        samples.append((path, int(ch)))
    return samples


def _run_model(model: str, samples: list[tuple[str, int]]) -> None:
    """对单个模型在整个数据集上跑 detect，打印每图结果与汇总指标。"""
    print(f"\n{'=' * 56}\n模型: {model}\n{'=' * 56}")
    try:
        from llm.vlm_client import GPT4oVLMClient
        vlm = GPT4oVLMClient(model=model)
    except Exception as e:
        print(f"[FAIL] 初始化失败: {e}")
        return

    tp = fp = tn = fn = 0
    elapsed_list = []

    for path, gt in samples:
        name = os.path.basename(path)
        try:
            t0 = time.perf_counter()
            result = vlm.detect(path)
            dt = time.perf_counter() - t0
            elapsed_list.append(dt)
            pred = 1 if result.has_waldo else 0

            if gt == 1 and pred == 1:
                tp += 1; tag = "TP"
            elif gt == 0 and pred == 1:
                fp += 1; tag = "FP ←误报"
            elif gt == 0 and pred == 0:
                tn += 1; tag = "TN"
            else:
                fn += 1; tag = "FN ←漏检"

            mark = "✓" if pred == gt else "✗"
            print(
                f"  {mark} {name}  gt={gt} pred={pred} [{tag:8}] "
                f"conf={result.confidence:.2f}  {dt:5.2f}s"
            )
        except Exception as e:
            print(f"  ✗ {name}  [FAIL] {e}")

    total = tp + fp + tn + fn
    if total == 0:
        return
    acc = (tp + tn) / total
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    avg = sum(elapsed_list) / len(elapsed_list) if elapsed_list else 0.0

    print(f"  {'-' * 52}")
    print(f"  准确率 {acc:.0%}  召回 {recall:.0%}  精确 {precision:.0%}  "
          f"| TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  平均耗时 {avg:.2f}s/张  (min {min(elapsed_list):.2f} / max {max(elapsed_list):.2f})")


def main():
    print("=" * 56)
    print("GPT detect 对照实验：推理 vs 非推理模型（带标注验证集）")
    print("=" * 56)

    if not _check_key_and_sdk():
        return

    samples = _load_dataset()
    if not samples:
        print(f"\n[SKIP] 数据集为空: {DATASET_DIR}")
        return
    pos = sum(g for _, g in samples)
    print(f"\n数据集: {DATASET_DIR}")
    print(f"样本 {len(samples)} 张（含 Waldo {pos} / 不含 {len(samples) - pos}）")

    for model in MODELS:
        _run_model(model, samples)

    print("\n[DONE] 若非推理模型准确率不低于推理模型且速度明显更优，"
          "可将 detect 的模型切换之。")


if __name__ == "__main__":
    main()
