# 判别式 DETECT_PROMPT + Top-3 排序评测 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 重写 detect 阶段的 `DETECT_PROMPT` 使其置信度成为有效排序信号（真 Waldo patch 能进 top-3），并配套一个基于 ground truth 的 Top-3 命中率评测脚本来量化改进。

**架构：** 三步。(1) 纯函数模块 `tests/patch_sampler.py` 从 `original-images/bbox` 真值生成「1 正 + 9 负」patch 框、计算悲观平局排名——可脱离 API 单元测试。(2) 评测脚本 `tests/eval_detect_ranking.py` 调 VLM 给每个 patch 打分，汇总 Top-3 命中率/召回率/负样本误报率，先用旧 prompt 跑出基线。(3) 重写 `prompts.py` 的 `DETECT_PROMPT`，重跑评测对比。

**技术栈：** Python 3.10、Pillow（裁剪/尺寸）、pytest（纯函数单测）、现有 `llm` VLM 抽象层（`get_vlm_client` / `DetectResult`）、`vision.image_utils`。

**设计依据：** `docs/superpowers/specs/2026-06-11-detect-discriminative-prompt-design.md`

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `tests/patch_sampler.py` | 创建 | 纯函数：解析 bbox 真值文件、生成正/负 patch 框、悲观平局排名。无 API、无 I/O（除读 bbox 文件）。 |
| `tests/test_patch_sampler.py` | 创建 | `patch_sampler` 的 pytest 单元测试（不调 API）。 |
| `tests/eval_detect_ranking.py` | 创建 | 评测 runner：组装 patch、调 VLM、算指标、打印。脚本式（`__main__`）。 |
| `prompts.py` | 修改（`DETECT_PROMPT`，第 13-30 行） | 判别式 prompt 重写。 |

**关键既有接口（计划中直接复用，勿改签名）：**
- `vision.image_utils.crop_to_pil(image_path, bbox)` —— `bbox=[x,y,w,h]`，返回 PIL Image。
- `vision.image_utils.save_patch(img, output_path)` —— 保存并自动建目录，返回路径。
- `llm.vlm_client.get_vlm_client(provider, model=...)` —— 返回带 `.detect(path) -> DetectResult` 的 client。
- `DetectResult(has_waldo: bool, confidence: float, bbox, raw_response)`。
- `prompts.DETECT_PROMPT` 被 `llm/base.py:9` 导入为 `BaseVLMClient.DETECT_PROMPT`，改 `prompts.py` 即全局生效。
- `tests/conftest.py` 已把项目根加入 `sys.path` 并加载 `.env`，故 pytest 内可 `from tests.patch_sampler import ...`。

---

## 任务 1：纯函数模块 `patch_sampler`

**文件：**
- 创建：`tests/patch_sampler.py`
- 测试：`tests/test_patch_sampler.py`

- [ ] **步骤 1：编写失败的测试**

创建 `tests/test_patch_sampler.py`：

```python
"""patch_sampler 纯函数单元测试（不调用 API）。"""

import os
import random

from tests.patch_sampler import (
    parse_bbox_file,
    positive_patch_bbox,
    negative_patch_bboxes,
    pessimistic_rank,
)

_BBOX_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "original-images", "bbox"
)


# ---------- parse_bbox_file ----------

def test_parse_bbox_file_line1():
    boxes = parse_bbox_file(_BBOX_FILE)
    # 第 1 行对应 1.jpg
    assert boxes[1] == [690, 520, 740, 610]


def test_parse_bbox_file_line16_is_none():
    boxes = parse_bbox_file(_BBOX_FILE)
    # 第 16 行为空 → 无真值
    assert boxes[16] is None


def test_parse_bbox_file_has_18_valid_entries():
    boxes = parse_bbox_file(_BBOX_FILE)
    valid = [k for k, v in boxes.items() if v is not None]
    assert len(valid) == 18


# ---------- positive_patch_bbox ----------

def test_positive_patch_is_patch_px_square_and_contains_gt():
    gt = [690, 520, 740, 610]  # center (715, 565)
    x, y, w, h = positive_patch_bbox(2048, 1251, gt, patch_px=200)
    assert w == 200 and h == 200
    # patch 完整包含真值框
    assert x <= gt[0] and y <= gt[1]
    assert x + w >= gt[2] and y + h >= gt[3]
    # 在图像内
    assert 0 <= x and 0 <= y and x + w <= 2048 and y + h <= 1251


def test_positive_patch_clamps_at_top_left_corner():
    gt = [10, 10, 40, 40]  # center (25,25) → 居中会越界，应 clamp 到 0
    x, y, w, h = positive_patch_bbox(2048, 1251, gt, patch_px=200)
    assert x == 0 and y == 0 and w == 200 and h == 200


# ---------- negative_patch_bboxes ----------

def test_negatives_count_and_no_overlap_with_gt():
    gt = [690, 520, 740, 610]
    rng = random.Random(42)
    negs = negative_patch_bboxes(2048, 1251, gt, patch_px=200, n=9, rng=rng)
    assert len(negs) == 9
    gx1, gy1, gx2, gy2 = gt
    for x, y, w, h in negs:
        # 与真值框零重叠
        no_overlap = (x >= gx2) or (x + w <= gx1) or (y >= gy2) or (y + h <= gy1)
        assert no_overlap, f"patch {[x, y, w, h]} overlaps gt {gt}"
        # 在图像内
        assert 0 <= x and 0 <= y and x + w <= 2048 and y + h <= 1251


def test_negatives_reproducible_with_same_seed():
    gt = [690, 520, 740, 610]
    a = negative_patch_bboxes(2048, 1251, gt, 200, 9, random.Random(42))
    b = negative_patch_bboxes(2048, 1251, gt, 200, 9, random.Random(42))
    assert a == b


# ---------- pessimistic_rank ----------

def test_rank_positive_clearly_first():
    assert pessimistic_rank(0.9, [0.5, 0.5, 0.1]) == 1


def test_rank_pessimistic_ties_push_positive_last():
    # 2 个负样本与正样本同分 → 正样本排第 3
    assert pessimistic_rank(0.99, [0.99, 0.99, 0.5]) == 3


def test_rank_counts_strictly_greater_negatives():
    # 2 个负样本严格大于 → 正样本排第 3
    assert pessimistic_rank(0.5, [0.9, 0.7, 0.4]) == 3
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_patch_sampler.py -v`
预期：FAIL（`ModuleNotFoundError: No module named 'tests.patch_sampler'` 或 import 错误）

- [ ] **步骤 3：编写最少实现代码**

创建 `tests/patch_sampler.py`：

```python
"""评测用纯函数：从 ground truth 生成正/负 patch 框 + 悲观平局排名。

无 API、无图像 I/O（除读取 bbox 真值文件）；可独立单元测试。
坐标约定：
  - 真值框 gt = [x1, y1, x2, y2]（左上角 + 右下角）
  - patch 框 = [x, y, w, h]（左上角 + 宽高），与 crop_to_pil 一致
"""

from __future__ import annotations

import random


def parse_bbox_file(path: str) -> dict[int, list[int] | None]:
    """读取 bbox 真值文件，第 N 行 → N.jpg 的 [x1,y1,x2,y2]；空行 → None。

    返回 {行号(从1起): [x1,y1,x2,y2] | None}。
    """
    result: dict[int, list[int] | None] = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip().strip("[]")
            if not s:
                result[i] = None
                continue
            parts = [int(p) for p in s.replace(" ", "").split(",") if p != ""]
            result[i] = parts if len(parts) == 4 else None
    return result


def positive_patch_bbox(
    img_w: int, img_h: int, gt: list[int], patch_px: int = 200
) -> list[int]:
    """以真值框中心裁一个 patch_px×patch_px 窗口，越界则 clamp 回界内。"""
    cx = (gt[0] + gt[2]) // 2
    cy = (gt[1] + gt[3]) // 2
    x = cx - patch_px // 2
    y = cy - patch_px // 2
    x = max(0, min(x, img_w - patch_px))
    y = max(0, min(y, img_h - patch_px))
    return [x, y, patch_px, patch_px]


def _overlaps(patch: list[int], gt: list[int]) -> bool:
    """patch=[x,y,w,h] 与 gt=[x1,y1,x2,y2] 是否有非零交集。"""
    px1, py1, pw, ph = patch
    px2, py2 = px1 + pw, py1 + ph
    gx1, gy1, gx2, gy2 = gt
    return px1 < gx2 and px2 > gx1 and py1 < gy2 and py2 > gy1


def negative_patch_bboxes(
    img_w: int,
    img_h: int,
    gt: list[int],
    patch_px: int,
    n: int,
    rng: random.Random,
    max_attempts: int = 10000,
) -> list[list[int]]:
    """随机采样 n 个 patch_px 窗口，均与真值框零重叠且在图像内。

    用传入的 rng 保证可复现。采样不足 max_attempts 仍凑不齐则抛错。
    """
    if img_w < patch_px or img_h < patch_px:
        raise ValueError(f"image {img_w}x{img_h} smaller than patch {patch_px}")
    negs: list[list[int]] = []
    attempts = 0
    while len(negs) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"failed to sample {n} negatives after {max_attempts} tries")
        x = rng.randint(0, img_w - patch_px)
        y = rng.randint(0, img_h - patch_px)
        patch = [x, y, patch_px, patch_px]
        if not _overlaps(patch, gt):
            negs.append(patch)
    return negs


def pessimistic_rank(pos_conf: float, neg_confs: list[float]) -> int:
    """正样本在「正+负」集合中的排名，平局时正样本排在并列者之后（悲观）。

    rank = 1 + #(neg > pos) + #(neg == pos)；命中 top-k 即 rank <= k。
    """
    greater = sum(1 for c in neg_confs if c > pos_conf)
    equal = sum(1 for c in neg_confs if c == pos_conf)
    return 1 + greater + equal
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_patch_sampler.py -v`
预期：PASS（10 个测试全绿）

- [ ] **步骤 5：Commit**

```bash
git add tests/patch_sampler.py tests/test_patch_sampler.py
git commit -m "test: add patch_sampler pure functions for detect ranking eval

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 任务 2：评测 runner + 采集旧 prompt 基线

**文件：**
- 创建：`tests/eval_detect_ranking.py`

> 本任务无单元测试（runner 是调 API 的胶水脚本，纯逻辑已在任务 1 覆盖）。验证方式 = 实际运行脚本采集**旧 prompt 基线**，确认输出指标结构正确。

- [ ] **步骤 1：编写 runner 脚本**

创建 `tests/eval_detect_ranking.py`：

```python
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
```

- [ ] **步骤 2：运行 runner 采集旧 prompt 基线**

运行：`python tests/eval_detect_ranking.py`
预期：跑完 18 张图（跳过 16），打印每图 HIT/miss 行 + 汇总。**记录这组数字为基线**——按诊断预期：Top-3 命中率接近随机基线（约 30%，即 ~5-6/18）、负样本误报率高（>50%）、gap 接近 0。
（注：这是真实 API 调用，约 180 次，受 50 req/min 限流，耗时数分钟属正常。）

- [ ] **步骤 3：把基线写进设计文档备查**

把步骤 2 的汇总数字（Top-3 命中率 / 召回率 / 负样本误报率 / gap）以一行追加到设计文档末尾的「基线」小节。运行：

打开 `docs/superpowers/specs/2026-06-11-detect-discriminative-prompt-design.md`，在文件末尾追加：

```markdown

## 旧 prompt 基线（实测）

- Top-3 命中率：<填实测>%
- 召回率：<填实测>%
- 负样本误报率：<填实测>%
- 置信度 gap：<填实测>
```

将 `<填实测>` 替换为步骤 2 的真实数字。

- [ ] **步骤 4：Commit**

```bash
git add tests/eval_detect_ranking.py docs/superpowers/specs/2026-06-11-detect-discriminative-prompt-design.md
git commit -m "test: add detect ranking eval harness and record old-prompt baseline

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 任务 3：重写 DETECT_PROMPT 并对比验证

**文件：**
- 修改：`prompts.py:13-30`（`DETECT_PROMPT`）

- [ ] **步骤 1：重写 DETECT_PROMPT**

将 `prompts.py` 中现有 `DETECT_PROMPT`（第 13-30 行）整体替换为：

```python
DETECT_PROMPT = (
    "You are scanning ONE small patch of a dense 'Where's Waldo' crowd scene.\n"
    "Decide whether Waldo (Wally) is in THIS patch.\n\n"
    "Waldo's ONLY reliable signature is a RED-AND-WHITE HORIZONTALLY STRIPED shirt\n"
    "on a SINGLE person's torso. Supporting (not sufficient alone) cues: a red-and-white\n"
    "bobble hat, round wire-frame glasses, slim build.\n\n"
    "CRITICAL — this is a crowd scene packed with decoys: red flags, white banners,\n"
    "red hats, striped patterns on unrelated objects, and many people wearing red OR white.\n"
    "Red pixels alone, white pixels alone, or red and white on DIFFERENT objects do NOT\n"
    "count. Require red AND white alternating horizontal stripes co-located on the SAME\n"
    "person before you treat it as a strong match. Do NOT answer present=true just because\n"
    "the patch contains red or white somewhere.\n\n"
    "Reply with ONLY this JSON, no markdown:\n"
    '{\"present\": true/false, \"confidence\": 0.0-1.0}\n\n'
    "Calibrate confidence to what you actually see:\n"
    "  0.8-1.0  clear red+white horizontal stripes on one person's torso\n"
    "           (extra confidence if bobble hat / round glasses also present)\n"
    "  0.4-0.7  a single figure with partial / occluded / cropped stripe pattern,\n"
    "           plausibly Waldo but not certain\n"
    "  0.1-0.3  only weak hints (red top without clear white stripes, or a striped\n"
    "           pattern that is NOT on a person)\n"
    "  0.0      no co-located red+white horizontal stripes anywhere in the patch\n\n"
    "Set present=true when confidence >= 0.4. Still report a genuine but partial Waldo\n"
    "(do not drop recall) — just give it the lower, honest confidence above.\n"
)
```

- [ ] **步骤 2：运行纯函数单测确认未被破坏**

运行：`pytest tests/test_patch_sampler.py -v`
预期：PASS（prompt 改动不影响纯函数，应仍全绿）

- [ ] **步骤 3：运行评测对比新 prompt**

运行：`python tests/eval_detect_ranking.py`
预期：与任务 2 基线相比，**Top-3 命中率显著上升**、**负样本误报率明显下降**、**gap 转为明显正值**，且**召回率不低于基线**。（同一 SEED=42，负样本与基线完全一致，可比。）

- [ ] **步骤 4：判定并记录结果**

对照设计文档「通过条件」：Top-3 命中率显著高于基线 **且** 召回率不低于基线。
- 若通过：在设计文档末尾追加「## 新 prompt 实测」小节，填入新数字，并标注「✅ 通过，方案 1 生效」。
- 若不通过：追加同样小节并标注「❌ 未达标，建议升级方案 3（detect 换 gpt-5.5）」，**停下来交给用户决策**，不要擅自换模型。

- [ ] **步骤 5：Commit**

```bash
git add prompts.py docs/superpowers/specs/2026-06-11-detect-discriminative-prompt-design.md
git commit -m "feat(detect): rewrite DETECT_PROMPT to be discriminative on red-white stripes

Replace the permissive ANY-feature rule (which made 90%+ patches return 0.99
and destroyed confidence ranking) with a rule requiring co-located red+white
horizontal stripes on a single person, plus calibrated confidence anchors.
Preserves recall via lower honest confidence for partial Waldo. Verified with
tests/eval_detect_ranking.py: Top-3 hit rate up, negative false-positive down.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 完成标准

- `pytest tests/test_patch_sampler.py -v` 全绿。
- `python tests/eval_detect_ranking.py` 在新 prompt 下 Top-3 命中率显著高于旧 prompt 基线、召回率不降。
- 设计文档记录了新旧两组实测数字。
- 若新 prompt 未达标：已停下并把方案 3 的决策交回用户，未擅自改模型/架构（遵守 YAGNI 与范围约束）。
