"""detect 排序评测：Top-3 命中率为主指标。

每张含真值的图构造 1 个正样本 patch（含真 Waldo）+ N_NEG 个负样本 patch
（同图其它位置、与真值零重叠），跑 detect 打分，看正样本能否排进 top-3。

用法：
    python tests/eval_detect_ranking.py
顶部常量可调 provider / model / 负样本数 / 随机种子。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(ROOT, ".env"))

import random

from PIL import Image

from tests.patch_sampler import (
    parse_bbox_file,
    positive_patch_bbox,
    negative_patch_bboxes,
    pessimistic_rank,
)
from vision.image_utils import crop_to_pil, save_patch
from llm.vlm_client import get_vlm_client

# ── 可调参数（对齐 detect 节点）────────────────────────────────
VLM_PROVIDER = "gpt4o"
VLM_MODEL = "gpt-5.4-mini"
N_NEG = 9
PATCH_PX = 200
SEED = 42
TOP_K = 3

IMAGE_DIR = os.path.join(ROOT, "original-images")
BBOX_FILE = os.path.join(IMAGE_DIR, "bbox")
EVAL_PATCH_DIR = os.path.join(ROOT, "outputs", "eval_patches")


def _score_patch(vlm, image_path: str, bbox: list[int], tag: str) -> tuple[bool, float]:
    """裁出 patch 存盘后调 detect，返回 (present, confidence)。"""
    crop = crop_to_pil(image_path, bbox)
    path = os.path.join(EVAL_PATCH_DIR, f"{tag}.jpg")
    save_patch(crop, path)
    r = vlm.detect(path)
    return r.has_waldo, r.confidence


def main():
    print("=" * 70)
    print(f"detect 排序评测  provider={VLM_PROVIDER} model={VLM_MODEL} "
          f"N_NEG={N_NEG} seed={SEED}")
    print("=" * 70)

    boxes = parse_bbox_file(BBOX_FILE)
    vlm = get_vlm_client(VLM_PROVIDER, model=VLM_MODEL)
    rng = random.Random(SEED)

    hits = 0
    evaluated = 0
    pos_present_count = 0
    neg_present_count = 0
    neg_total = 0
    pos_conf_sum = 0.0
    neg_conf_sum = 0.0

    for n in sorted(boxes):
        gt = boxes[n]
        if gt is None:
            continue
        image_path = os.path.join(IMAGE_DIR, f"{n}.jpg")
        if not os.path.exists(image_path):
            print(f"  [skip] {n}.jpg 不存在")
            continue

        img_w, img_h = Image.open(image_path).size
        pos_box = positive_patch_bbox(img_w, img_h, gt, PATCH_PX)
        neg_boxes = negative_patch_bboxes(img_w, img_h, gt, PATCH_PX, N_NEG, rng)

        pos_present, pos_conf = _score_patch(vlm, image_path, pos_box, f"{n}_pos")
        neg_confs = []
        for j, nb in enumerate(neg_boxes):
            present, conf = _score_patch(vlm, image_path, nb, f"{n}_neg{j}")
            neg_confs.append(conf)
            neg_present_count += int(present)
        neg_total += len(neg_boxes)

        rank = pessimistic_rank(pos_conf, neg_confs)
        hit = rank <= TOP_K
        hits += int(hit)
        evaluated += 1
        pos_present_count += int(pos_present)
        pos_conf_sum += pos_conf
        neg_conf_sum += sum(neg_confs)

        flag = "HIT " if hit else "miss"
        print(f"  [{flag}] {n}.jpg  pos_conf={pos_conf:.2f} rank={rank}/"
              f"{N_NEG + 1}  neg_max={max(neg_confs):.2f} neg_mean={sum(neg_confs) / len(neg_confs):.2f}")

    print("-" * 70)
    if evaluated == 0:
        print("没有可评测的图。")
        return
    top3 = hits / evaluated * 100
    recall = pos_present_count / evaluated * 100
    neg_fp = neg_present_count / neg_total * 100
    gap = pos_conf_sum / evaluated - neg_conf_sum / neg_total
    print(f"评测图数        : {evaluated}")
    print(f"Top-{TOP_K} 命中率(主) : {top3:.1f}%   ({hits}/{evaluated})")
    print(f"召回率(正样本)  : {recall:.1f}%")
    print(f"负样本误报率    : {neg_fp:.1f}%")
    print(f"置信度 gap      : {gap:+.3f}  (正均值 - 负均值)")
    print("-" * 70)


if __name__ == "__main__":
    main()