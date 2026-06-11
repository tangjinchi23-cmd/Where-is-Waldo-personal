"""快速召回（recall）自查：把一批已裁好的正样本 patch 依次过 detect，
统计召回率 + 列出失败图号。设计给「人」用，凭经验判断优化空间。

不依赖 bbox 真值：直接读「已经裁好的正样本图片文件」（每张都含 Waldo），
所以 detect 判 present=false 即为漏检（false negative）。

正样本文件命名约定：`<图号>_pos.jpg`（如 `18_pos.jpg`），默认从 POS_DIR 读取。

用法（手动运行，脚本不自动执行）：
    python tests/quick_detect_check.py                 # 跑默认 POS_DIR 下全部正样本
    python tests/quick_detect_check.py outputs/foo     # 指定别的正样本目录

可调参数集中在根目录 config.json（provider / model / temperature / repeats），
改 json 即可，无需动本脚本。
"""

import os
import sys

# Windows 控制台默认 GBK，会编码不了 ✓/中文；强制 utf-8 输出
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
POS_DIR = os.path.join(ROOT, "outputs", "eval_patches")
POS_SUFFIX = "_pos.jpg"      # 正样本文件名后缀


def _img_id(filename: str) -> int:
    """从 `18_pos.jpg` 取图号 18，用于排序与输出。"""
    return int(os.path.basename(filename).split("_")[0])


def main():
    pos_dir = sys.argv[1] if len(sys.argv) > 1 else POS_DIR
    pos_dir = pos_dir if os.path.isabs(pos_dir) else os.path.join(ROOT, pos_dir)

    files = sorted(
        (os.path.join(pos_dir, f) for f in os.listdir(pos_dir)
         if f.endswith(POS_SUFFIX)),
        key=_img_id,
    ) if os.path.isdir(pos_dir) else []

    if not files:
        print(f"未在 {pos_dir} 找到 *{POS_SUFFIX} 正样本文件。")
        return

    print("=" * 70)
    print(f"召回自查  provider={_CFG['provider']} model={_CFG['model']} "
          f"temp={_CFG['temperature']} repeats={REPEATS}  n={len(files)}")
    print("=" * 70)

    vlm = build_vlm(_CFG)

    failed: list[int] = []
    for path in files:
        n = _img_id(path)
        recalled, conf, details = run_repeats(vlm, path, REPEATS)
        mark = "✓ OK  " if recalled else "✗ FAIL"
        if not recalled:
            failed.append(n)
        print(f"  [{mark}] {n}.jpg   conf={conf:.3f}")
        for present, c, reason in details:  # prompt engineering：打印决策原因
            print(f"          · present={present} conf={c:.2f} | {reason}")

    total = len(files)
    hit = total - len(failed)
    print("-" * 70)
    print(f"召回率 (recall) : {hit}/{total} = {hit / total * 100:.1f}%")
    print(f"失败图片序号    : {failed}")


if __name__ == "__main__":
    main()
