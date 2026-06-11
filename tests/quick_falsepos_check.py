"""误检（false positive）自查：把一批「不含 Waldo」的负样本 patch 依次过 detect，
统计误检率 + 列出被错报的样本。给「人」用，手工调 prompt 降低误检率时看效果。

不依赖 bbox 真值：直接读「已经裁好的负样本图片文件」（每张都不含 Waldo），
所以 detect 判 present=true 即为误检（false positive）。

负样本文件命名约定：名字里含 `_neg`（如 `10_neg3.jpg`），默认从 NEG_DIR 读取。

用法（手动运行，脚本不自动执行）：
    python tests/quick_falsepos_check.py               # 跑默认 NEG_DIR 下全部负样本
    python tests/quick_falsepos_check.py 10            # 只测第 10 张图的负样本（10_neg*）
    python tests/quick_falsepos_check.py outputs/foo   # 指定别的负样本目录

可调参数集中在根目录 config.json（provider / model / temperature / repeats / limit），
改 json 即可，无需动本脚本。
"""

import os
import random
import sys

# Windows 控制台默认 GBK，会编码不了 ✗/中文；强制 utf-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(ROOT, ".env"))

from tests.quick_config import load_config, build_vlm, run_repeats

# ── 从 config.json 读可调参数 ───────────────────────────────────
_CFG = load_config()
REPEATS = _CFG["repeats"]    # 每张重复次数，>1 时取多数票
LIMIT = _CFG["limit"]        # 0=全部；>0 则随机抽这么多张（固定种子可复现）
SEED = 42
NEG_DIR = os.path.join(ROOT, "outputs", "eval_patches")
NEG_MARK = "_neg"            # 负样本文件名标识


def _is_neg(f: str) -> bool:
    return NEG_MARK in f and f.lower().endswith((".jpg", ".jpeg", ".png"))


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    # 传图号 → 只测该图的负样本（如 "10" 匹配 10_neg*）；传路径 → 当目录用
    if arg is not None and arg.isdigit():
        neg_dir = NEG_DIR
        only_id = arg
    else:
        neg_dir = arg or NEG_DIR
        neg_dir = neg_dir if os.path.isabs(neg_dir) else os.path.join(ROOT, neg_dir)
        only_id = None

    files = sorted(
        os.path.join(neg_dir, f) for f in os.listdir(neg_dir)
        if _is_neg(f) and (only_id is None or f.split("_")[0] == only_id)
    ) if os.path.isdir(neg_dir) else []

    if not files:
        scope = f"图号 {only_id} 的" if only_id else ""
        print(f"未在 {neg_dir} 找到{scope}含 '{NEG_MARK}' 的负样本文件。")
        return

    if LIMIT and LIMIT < len(files):
        files = random.Random(SEED).sample(files, LIMIT)
        files.sort()

    print("=" * 70)
    print(f"误检自查  provider={_CFG['provider']} model={_CFG['model']} "
          f"temp={_CFG['temperature']} repeats={REPEATS}  n={len(files)}")
    print("=" * 70)

    vlm = build_vlm(_CFG)

    fired_files: list[str] = []
    for path in files:
        fired, conf, details = run_repeats(vlm, path, REPEATS)
        if fired:
            fired_files.append(os.path.basename(path))
            print(f"  [✗ 误检] {os.path.basename(path):<16} conf={conf:.3f}")
            for present, c, reason in details:  # prompt engineering：打印误检原因
                print(f"          · present={present} conf={c:.2f} | {reason}")

    total = len(files)
    fp = len(fired_files)
    print("-" * 70)
    if fp == 0:
        print("没有误检，全部负样本都判为「无 Waldo」。")
    print(f"误检率 (false positive) : {fp}/{total} = {fp / total * 100:.1f}%")
    print(f"被误检的负样本          : {fired_files}")


if __name__ == "__main__":
    main()
