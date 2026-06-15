"""Grounding DINO —— Where's Waldo detect 评测脚本。

针对 200x200 小图中 ~30x30 的小目标，上采样后做开放词汇检测。两种模式共用同一份
已加载的模型（加载是最贵的一步，绝不重复加载）：

  1. 全量评测（默认，无参数）—— 在 `outputs/eval_patches/` 上跑：
       · 召回率：18 张 `*_pos.jpg`（含 Waldo），检出任意框即命中
       · 误检率：162 张 `*_neg*.jpg`（不含 Waldo），检出任意框即误检
     输出口径与 `tests/quick_detect_check.py` / `quick_falsepos_check.py` 一致，
     可直接和 gpt-5.5 / gpt-5.4-mini / qwen-vl 的召回/误检数字同表对比。

  2. 单图调试 —— `python tests/Grounding_DINO_test.py <图片路径>`：
     在该图上推理 + 画框 + 存 result.png，肉眼看框得准不准。

判定口径：patch 内出现 ≥ BOX_THRESHOLD 的框即 present=true（与 VLM 的 present/absent
对齐，召回/误检才可比）。GDINO 不认识 "Waldo" 这个专有名词，故 prompt 用视觉属性描述。
"""

import inspect
import os
import sys

# Windows 控制台默认 GBK，编码不了 ✓/中文；强制 utf-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ 配置区（按需修改） ============
MODEL_ID = "IDEA-Research/grounding-dino-base"   # 想更快可换 "...-tiny"

# 用视觉属性描述，而不是 "Waldo" 这个名字（模型不认识这个专有名词）。
# GDINO 用 "." 分隔短语；prompt engineering 时改这里即可。
TEXT_PROMPT = "a small person with glasses and a red and white hat ."

UPSCALE = 4            # 上采样倍数：30x30 -> 120x120，细节更可分
BOX_THRESHOLD = 0.15  # 小目标 + 单一目标，阈值放低保召回
TEXT_THRESHOLD = 0.15

EVAL_DIR = os.path.join(ROOT, "outputs", "eval_patches")
POS_SUFFIX = "_pos.jpg"      # 正样本（含 Waldo）
NEG_MARK = "_neg"            # 负样本（不含 Waldo）标识
OUTPUT_PATH = os.path.join(ROOT, "tests", "result.png")
ONLY_BEST = True             # 单图调试：True=只画最高分框（适合"框里就一个Waldo"）
# ==========================================


def post_process_compat(processor, outputs, input_ids, box_thr, text_thr, target_sizes):
    """兼容不同 transformers 版本的参数名：新版用 threshold，旧版用 box_threshold。"""
    fn = processor.post_process_grounded_object_detection
    params = inspect.signature(fn).parameters
    kwargs = {
        "input_ids": input_ids,
        "text_threshold": text_thr,
        "target_sizes": target_sizes,
    }
    if "threshold" in params:
        kwargs["threshold"] = box_thr          # 新版
    elif "box_threshold" in params:
        kwargs["box_threshold"] = box_thr      # 旧版
    return fn(outputs, **kwargs)


def load_model():
    """加载 processor + model 到 GPU/CPU（只调一次，全程复用）。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
    return processor, model, device


def detect(processor, model, device, image_path):
    """对单张 patch 做检测，返回 (present, best_score, dets)。

    dets = [(x0, y0, x1, y1, score, label), ...]，坐标已换算回原始分辨率。
    present = 是否存在 ≥ BOX_THRESHOLD 的框（post_process 已按阈值过滤）。
    """
    image = Image.open(image_path).convert("RGB")
    image_up = image.resize(
        (image.width * UPSCALE, image.height * UPSCALE), Image.Resampling.LANCZOS
    )

    inputs = processor(images=image_up, text=TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = post_process_compat(
        processor, outputs, inputs.input_ids,
        BOX_THRESHOLD, TEXT_THRESHOLD,
        target_sizes=[image_up.size[::-1]],   # (height, width)
    )[0]

    dets = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x0, y0, x1, y1 = [v / UPSCALE for v in box.tolist()]
        dets.append((x0, y0, x1, y1, float(score), label))

    best = max((d[4] for d in dets), default=0.0)
    return len(dets) > 0, best, dets


# ── 全量评测 ───────────────────────────────────────────────────────────

def _img_id(filename: str) -> int:
    """从 `18_pos.jpg` / `10_neg3.jpg` 取图号，用于排序。"""
    return int(os.path.basename(filename).split("_")[0])


def run_eval(processor, model, device):
    """在 eval_patches 上跑召回 + 误检，输出与 quick_*_check.py 同口径的汇总。"""
    if not os.path.isdir(EVAL_DIR):
        print(f"未找到评测目录：{EVAL_DIR}")
        return

    pos = sorted(
        (f for f in os.listdir(EVAL_DIR) if f.endswith(POS_SUFFIX)), key=_img_id
    )
    neg = sorted(
        (f for f in os.listdir(EVAL_DIR)
         if NEG_MARK in f and f.lower().endswith((".jpg", ".jpeg", ".png"))),
        key=_img_id,
    )

    print("=" * 70)
    print(f"Grounding DINO 评测  model={MODEL_ID.split('/')[-1]}  "
          f"upscale={UPSCALE}  box_thr={BOX_THRESHOLD}  text_thr={TEXT_THRESHOLD}")
    print(f"prompt: {TEXT_PROMPT!r}")
    print("=" * 70)

    # 召回：正样本检出任意框即命中
    print(f"\n── 召回（{len(pos)} 张正样本）" + "─" * 40)
    pos_fail, pos_scores = [], []
    for f in pos:
        present, best, _ = detect(processor, model, device, os.path.join(EVAL_DIR, f))
        pos_scores.append(best)
        if not present:
            pos_fail.append(_img_id(f))
        print(f"  [{'✓ OK  ' if present else '✗ FAIL'}] {f:<14} best_score={best:.3f}")

    # 误检：负样本检出任意框即误检
    print(f"\n── 误检（{len(neg)} 张负样本，仅列被误检的）" + "─" * 18)
    fp_files, neg_scores = [], []
    for f in neg:
        present, best, _ = detect(processor, model, device, os.path.join(EVAL_DIR, f))
        neg_scores.append(best)
        if present:
            fp_files.append(f)
            print(f"  [✗ 误检] {f:<14} best_score={best:.3f}")

    # 汇总
    n_pos, n_neg = len(pos), len(neg)
    recall = (n_pos - len(pos_fail)) / n_pos * 100 if n_pos else 0.0
    fp_rate = len(fp_files) / n_neg * 100 if n_neg else 0.0
    avg_pos = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
    avg_neg = sum(neg_scores) / len(neg_scores) if neg_scores else 0.0

    print("\n" + "=" * 70)
    print(f"召回率 (recall)          : {n_pos - len(pos_fail)}/{n_pos} = {recall:.1f}%")
    print(f"误检率 (false positive)  : {len(fp_files)}/{n_neg} = {fp_rate:.1f}%")
    print(f"判别力 (召回 − 误检)      : {recall - fp_rate:.1f}")
    print(f"平均 best_score  正样本={avg_pos:.3f}  负样本={avg_neg:.3f}  "
          f"gap={avg_pos - avg_neg:+.3f}")
    print(f"召回失败图号             : {pos_fail}")
    print("=" * 70)


# ── 单图调试 ───────────────────────────────────────────────────────────

def run_single(processor, model, device, image_path):
    """单图：推理 + 画框 + 存 result.png，肉眼检查框得准不准。"""
    image = Image.open(image_path).convert("RGB")
    present, best, dets = detect(processor, model, device, image_path)

    if not present:
        print("未检测到目标。试试：调低 BOX_THRESHOLD（如 0.1）、加大 UPSCALE、"
              "或换更具体的 TEXT_PROMPT。")
        return

    if ONLY_BEST:
        dets = [max(dets, key=lambda d: d[4])]

    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for x0, y0, x1, y1, score, label in dets:
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        draw.text((x0, max(0, y0 - 12)), f"{label} {score:.2f}",
                  fill=(255, 0, 0), font=font)
        print(f"{label} ({score:.2f}): [{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}]")

    draw_img.save(OUTPUT_PATH)
    print(f"\n结果已保存到: {OUTPUT_PATH}")


def main():
    processor, model, device = load_model()
    # 传图片路径 → 单图调试；无参数 → 全量评测
    if len(sys.argv) > 1:
        run_single(processor, model, device, sys.argv[1])
    else:
        run_eval(processor, model, device)


if __name__ == "__main__":
    main()
