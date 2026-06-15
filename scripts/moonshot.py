"""Moonshot Kimi —— Where's Waldo detect 召回 + 误检自查（独立脚本）。

Moonshot 走 OpenAI 兼容接口，故复用 `openai` SDK（与 scripts/QWEN.py 同源），
评测套路对齐 scripts/GEMINI.py：

  1. 同时测 **召回（_pos.jpg）** 和 **误检（_neg*.jpg）** —— 只看召回会被骗：
     之前 qwen-vl-plus 召回 88.9% 却误检 80%，纯属"无脑说有"。结论必须两者合看。
  2. prompt 用项目实测对 gpt-5.5 召回最优的 feature-free 英文版（不列特征）。
  3. .env 用相对脚本文件的绝对路径加载，从任何目录运行都能找到 key。

⚠️ Moonshot 国内站 (api.moonshot.cn) 与国际站 (api.moonshot.ai) 是两套独立账号，
   key 不通用。下方 BASE_URL 按你的注册站点选，选错会报 401。

用法（需先在项目根 .env 配 MOONSHOT_API_KEY）：
    pip install openai
    python scripts/moonshot.py --list-models  # 先列当前账号可用模型，挑一个填到 MODEL_NAME
    python scripts/moonshot.py                # 跑全量召回 + 误检
    python scripts/moonshot.py --pos-only     # 只跑召回（prompt 迭代用）
"""

import base64
import json
import os
import random
import re
import sys
import time

# Windows 控制台默认 GBK，编码不了 ✓/中文；强制 utf-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
from openai import OpenAI

# ============ 配置区（按需修改） ============
# 项目根目录 = 本文件的上上级，.env 与 outputs 都在这
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

API_KEY = os.getenv("MOONSHOT_API_KEY")

# ⚠️ 按你的注册站点二选一（国内站 .cn / 国际站 .ai，key 不通用）：
BASE_URL = "https://api.moonshot.cn/v1"
# BASE_URL = "https://api.moonshot.ai/v1"

IMAGE_DIR = os.path.join(ROOT, "outputs", "eval_patches")

# ⚠️ 先跑 --list-models 确认当前账号可用的视觉模型，再改这里（避免撞 404/无视觉能力）。
# kimi-k2.6 为当前原生多模态旗舰；旧账号可能只有 moonshot-v1-8k-vision-preview 之类。
MODEL_NAME = "kimi-k2.6"

TEMPERATURE = 0.0     # 非推理模型(moonshot-v1-*)用 0 求可复现；推理模型(kimi-k2.x)会在 main 里自动改成 1.0
MAX_TOKENS = 4096     # kimi-k2.x 推理模型须 ≥4096 防 reasoning token 吃光致空响应假漏检；非推理用不了这么多但无害
NEG_LIMIT = 0         # 误检随机抽查张数；0 = 全部 162 张（全量确认 FP）
SEED = 42             # 固定随机种子，抽样可复现

# 限流防护
REQUEST_INTERVAL = 1.0  # 每次请求间主动间隔秒数；撞 429 会自动退避重试，太激进就回调
MAX_RETRIES = 5         # 撞 429 后的重试次数
# ==========================================

# feature-free 英文版（项目实测对 gpt-5.5 召回最优）：不列具体特征，让模型用自身
# 对 Waldo 的认知去找，并提示目标可能小/被遮挡/模糊，以提升召回。
PROMPT = """This is a "Where's Waldo?" (also known as "Where's Wally?") style illustration.
Using your own knowledge of what the Waldo character looks like, carefully decide whether
Waldo is present in this image. He may appear small, partially hidden, or blurry, so look
very carefully before deciding.

Return strictly this JSON and nothing else:
{"found": true or false}
"""


def encode_image(image_path):
    """将本地图片转换为 Base64 编码。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def list_models(client):
    """列出当前账号可用模型（OpenAI 兼容 /models 接口；视觉能力需自行确认）。"""
    print("当前账号可用模型：")
    for m in client.models.list().data:
        print(f"  · {m.id}")


def query_kimi(client, image_path):
    """对单张图调用 Kimi，返回 (found: bool | None, error: str | None)。

    撞 429 限流时指数退避重试；其余异常直接返回错误。
    """
    b64 = encode_image(image_path)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},  # 强制 JSON，省去清洗 markdown
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                return None, "空响应（可能 token 截断或被安全策略拦截）"
            # 容错：万一模型仍包了 markdown 代码块
            if "```" in content:
                content = content.split("```")[1].removeprefix("json").strip()
            return bool(json.loads(content).get("found", False)), None
        except json.JSONDecodeError as e:
            return None, f"JSON 解析失败: {e} | 原始: {content!r}"
        except Exception as e:
            msg = str(e)
            is_429 = "429" in msg or "rate_limit" in msg.lower()
            if is_429 and attempt < MAX_RETRIES - 1:
                m = re.search(r"(\d+)\s*s", msg)
                wait = (int(m.group(1)) + 2) if m else 5 * (2 ** attempt)
                print(f"      ⏳ 限流，等待 {wait}s 重试 ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            return None, f"API 调用异常: {e}"


def _img_id(filename):
    """从 `18_pos.jpg` / `10_neg3.jpg` 取图号，用于排序。"""
    return int(os.path.basename(filename).split("_")[0])


def main():
    if not API_KEY:
        print("❌ 未找到 MOONSHOT_API_KEY，请在项目根 .env 添加：MOONSHOT_API_KEY=你的key")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # CLI: --model NAME 覆盖默认模型，方便横向对比不同模型而不必改文件
    global MODEL_NAME, TEMPERATURE
    if "--model" in sys.argv:
        MODEL_NAME = sys.argv[sys.argv.index("--model") + 1]
    # 推理模型(kimi-k2.x)只接受 temperature=1；非推理(moonshot-v1-*)沿用配置值
    if MODEL_NAME.startswith("kimi-k2"):
        TEMPERATURE = 1.0

    if "--list-models" in sys.argv:
        list_models(client)
        return

    if not os.path.isdir(IMAGE_DIR):
        print(f"❌ 未找到评测目录：{IMAGE_DIR}")
        return

    pos = sorted(
        (f for f in os.listdir(IMAGE_DIR) if f.endswith("_pos.jpg")), key=_img_id
    )
    neg = sorted(
        (f for f in os.listdir(IMAGE_DIR)
         if "_neg" in f and f.lower().endswith((".jpg", ".jpeg", ".png"))),
        key=_img_id,
    )
    if NEG_LIMIT and NEG_LIMIT < len(neg):
        neg = sorted(random.Random(SEED).sample(neg, NEG_LIMIT), key=_img_id)

    print("=" * 70)
    print(f"Kimi 评测  model={MODEL_NAME}  temp={TEMPERATURE}  "
          f"pos={len(pos)}  neg={len(neg)}")
    print("=" * 70)

    # ── 计时：墙钟总耗时 + 纯 API 耗时（含重试退避，不含主动限速 sleep）──
    t_wall_start = time.perf_counter()
    api_secs, n_calls = 0.0, 0

    # ── 召回：正样本判 found=false 即漏检 ──
    print(f"\n── 召回（{len(pos)} 张正样本）" + "─" * 40)
    pos_fail, errors = [], 0
    for f in pos:
        t0 = time.perf_counter()
        found, err = query_kimi(client, os.path.join(IMAGE_DIR, f))
        api_secs += time.perf_counter() - t0
        n_calls += 1
        if err:
            print(f"  [⚠️ ERR ] {f:<14} {err}")
            errors += 1
            continue
        if not found:
            pos_fail.append(_img_id(f))
        print(f"  [{'✓ OK  ' if found else '✗ FAIL'}] {f}")
        time.sleep(REQUEST_INTERVAL)  # 主动限速，避开每分钟配额

    # ── 误检：负样本判 found=true 即误检（--pos-only 时跳过，prompt 迭代只看召回）──
    pos_only = "--pos-only" in sys.argv
    fp_files = []
    if pos_only:
        neg = []
        print("\n（--pos-only：跳过负样本/误检测试）")
    else:
        print(f"\n── 误检（{len(neg)} 张负样本，仅列被误检的）" + "─" * 18)
        for f in neg:
            t0 = time.perf_counter()
            found, err = query_kimi(client, os.path.join(IMAGE_DIR, f))
            api_secs += time.perf_counter() - t0
            n_calls += 1
            if err:
                print(f"  [⚠️ ERR ] {f:<14} {err}")
                errors += 1
                continue
            if found:
                fp_files.append(f)
                print(f"  [✗ 误检] {f}")
            time.sleep(REQUEST_INTERVAL)  # 主动限速，避开每分钟配额

    # ── 汇总 ──
    n_pos, n_neg = len(pos), len(neg)
    recall = (n_pos - len(pos_fail)) / n_pos * 100 if n_pos else 0.0
    fp_rate = len(fp_files) / n_neg * 100 if n_neg else 0.0
    wall = time.perf_counter() - t_wall_start

    print("\n" + "=" * 70)
    print(f"召回率 (recall)          : {n_pos - len(pos_fail)}/{n_pos} = {recall:.1f}%")
    if not pos_only:
        print(f"误检率 (false positive)  : {len(fp_files)}/{n_neg} = {fp_rate:.1f}%")
        print(f"判别力 (召回 − 误检)      : {recall - fp_rate:.1f}")
    print(f"召回失败图号             : {pos_fail}")
    print(f"API/解析错误             : {errors}")
    print(f"总耗时 (墙钟)            : {wall:.1f}s ({wall / 60:.1f}min)")
    print(f"纯 API 耗时             : {api_secs:.1f}s  | 平均 "
          f"{api_secs / n_calls:.2f}s/张（共 {n_calls} 次调用）"
          if n_calls else "纯 API 耗时             : 无调用")
    print("=" * 70)
    print("⚠️ 提醒：高召回 + 高误检 = 无脑说有，不可用（参考 qwen-vl-plus）。"
          "判别力(召回−误检)才是关键，gpt-5.5 基线为 ~69。")


if __name__ == "__main__":
    main()
