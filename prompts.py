"""集中存放发给 VLM 的任务提示词（Waldo 检测相关）。

与 provider 无关、与编排无关，单独抽出便于统一调试与迭代：
    - DETECT_PROMPT          detect 阶段：判断 patch 是否含 Waldo
    - VERIFY_PROMPT          verify 阶段：对候选区域二次确认
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


# ---------- verify 阶段 ----------

VERIFY_PROMPT = (
    "You are performing a final verification step in a Where's Waldo search.\n"
    "This is a close-up crop centered on a suspected Waldo location.\n\n"
    "Waldo's distinguishing features:\n"
    "  - Red and white HORIZONTAL striped shirt (most reliable identifier)\n"
    "  - Round wire-frame glasses\n"
    "  - Red and white bobble hat (may be partially cropped)\n"
    "  - Slim build, often partially occluded by other characters\n\n"
    "Is the person shown definitely Waldo?\n\n"
    "Respond ONLY with a JSON object, no markdown, no extra text:\n"
    '{"is_waldo": true/false, "confidence": 0.0-1.0, "reason": "one short sentence"}\n\n'
    "  - is_waldo: true only if you clearly see the red-white horizontal stripes\n"
    "  - confidence: 0.0 = definitely not Waldo, 1.0 = absolutely certain it is Waldo\n"
    "  - reason: brief evidence (e.g. 'red-white stripes clearly visible on shirt')"
)
