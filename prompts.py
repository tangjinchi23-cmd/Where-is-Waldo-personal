"""集中存放发给 VLM 的任务提示词（Waldo 检测相关）。

与 provider 无关、与编排无关，单独抽出便于统一调试与迭代：
    - DETECT_PROMPT          detect 阶段：判断 patch 是否含 Waldo
    - SELECT_PROMPT          verify 阶段：在多张候选裁剪图间横向单选唯一真 Waldo
"""

from __future__ import annotations

# ---------- detect 阶段 ----------

DETECT_PROMPT = (
    "This is a patch from a 'Where's Waldo' puzzle. Decide whether Waldo (Wally) is in it.\n"
    "Use your own knowledge of what Waldo looks like. He is often small, partially hidden,\n"
    "or blurry — look carefully before deciding.\n\n"
    "Reply with ONLY this JSON, no markdown:\n"
    '{"present": true/false, "confidence": 0.0-1.0, "bbox": [x, y, w, h]}\n'
    "  - present: true if Waldo is in this patch\n"
    "  - confidence: probability Waldo IS in this patch (0.0 = not here, 1.0 = clearly\n"
    "    here); must agree with present.\n"
    "  - bbox: a TIGHT box around Waldo in PIXELS within THIS patch image,\n"
    "    [x, y, width, height] with (0,0) at the top-left corner. Use null if present\n"
    "    is false.\n"
)


# ---------- verify 阶段（横向单选） ----------

# 把多张候选裁剪图一次性发给 VLM，强制在候选间横向比较、只选一张真 Waldo。
# 实测优于逐张独立判断：逐张判断在密集难图上会把多张都判 Yes、且易被红白条纹误导。
SELECT_PROMPT = (
    "These are several close-up crops from a 'Where's Waldo' puzzle, each suspected to "
    "contain Waldo (Wally). They are given in order, indexed from 0. At most ONE of them "
    "is the real Waldo. Use your own knowledge of what Waldo looks like (red-and-white "
    "striped bobble hat, round glasses, red-white striped shirt, slim build). Do not be "
    "fooled by red-white stripes alone — many decoys have stripes.\n\n"
    "Reply with ONLY this JSON, no markdown:\n"
    '{"choice": <index of the real Waldo, or -1 if none is Waldo>, '
    '"confidence": 0.0-1.0, "per_image": [true/false, ...]}\n'
    "  - choice: 0-based index of the crop that is the real Waldo (-1 if none)\n"
    "  - per_image: for each crop in the given order, true if that crop looks like Waldo\n"
)
