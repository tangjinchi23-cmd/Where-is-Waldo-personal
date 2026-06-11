"""集中存放发给 VLM 的任务提示词（Waldo 检测相关）。

与 provider 无关、与编排无关，单独抽出便于统一调试与迭代：
    - DETECT_PROMPT          detect 阶段：判断 patch 是否含 Waldo
    - VERIFY_PROMPT          verify 阶段：对候选区域二次确认
    - build_analyze_prompt   analyze 阶段：根据图片尺寸动态生成网格推荐提示词
"""

from __future__ import annotations

# ---------- detect 阶段 ----------

DETECT_PROMPT = (
    "This is a patch from a 'Where's Waldo' puzzle. Decide whether Waldo (Wally) is in it.\n"
    "Use your own knowledge of what Waldo looks like. He is often small, partially hidden,\n"
    "or blurry — look carefully before deciding.\n\n"
    "Reply with ONLY this JSON, no markdown:\n"
    '{"present": true/false, "confidence": 0.0-1.0}\n'
    "  - present: true if Waldo is in this patch\n"
    "  - confidence: probability Waldo IS in this patch (0.0 = not here, 1.0 = clearly\n"
    "    here); must agree with present.\n"
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


# ---------- analyze 阶段 ----------

def build_analyze_prompt(img_w: int, img_h: int) -> str:
    """根据图片尺寸生成网格推荐提示词（analyze 节点用）。

    VLM 根据图片复杂度推荐切割行列数，目标每格约 200×200px。
    """
    target = 200
    suggest_cols = max(2, round(img_w / target))
    suggest_rows = max(2, round(img_h / target))
    return (
        f"This is a Where's Waldo puzzle image ({img_w}×{img_h} pixels).\n"
        f"Waldo is a small figure (~30-50px tall) hidden in a dense crowd.\n\n"
        f"Recommend how many rows and columns to split this image into for a grid search.\n"
        f"Each cell should be roughly 200×200 pixels so Waldo is large enough to identify.\n\n"
        f"Suggested starting point: {suggest_rows} rows × {suggest_cols} cols.\n"
        f"Adjust higher if the image is very dense or complex; lower if sparse.\n\n"
        "Reply ONLY with JSON, no markdown:\n"
        '{"rows": N, "cols": M}'
    )
