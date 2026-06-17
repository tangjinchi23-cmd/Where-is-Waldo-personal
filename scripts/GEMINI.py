"""Google Gemini —— Where's Waldo detect 召回 + 误检 + 置信度自查（独立脚本）。

用官方现行 `google-genai` SDK（旧的 `google-generativeai` 已停止支持）。与
scripts/QWEN.py 同一套路（不走项目 llm/ 抽象层），但已对齐正式 agent 的 detect 逻辑：
  1. **复用正式 `DETECT_PROMPT`（从 prompts.py import，单一真源）**：评测脚本测的就是
     agent 实跑的那条 prompt，结论可直接迁移到 `agent/nodes/detect.py`。
  2. **解析 present + confidence（与 detect 节点一致）**：detect 用 confidence 做阈值
     过滤 + top-3 排序，所以这里不只看「有没有」，还要查 Gemini 的 confidence 是否可靠
     ——是否与 present 一致、正负样本能否拉开 gap。mini / Qwen 都栽在 confidence 失效
     上（present=false / conf=0.97），Gemini 落地前必须确认这一点。
  3. 同时测 **召回（_pos.jpg）** 和 **误检（_neg*.jpg）** —— 只看召回会被骗：
     之前 qwen-vl-plus 召回 88.9% 却误检 80%，纯属"无脑说有"。结论必须两者合看。
  4. .env 用相对脚本文件的绝对路径加载，从任何目录运行都能找到 key。

用法（需先在项目根 .env 配 GOOGLE_API_KEY）：
    pip install google-genai
    python scripts/GEMINI.py --list-models   # 先列当前账号可用模型，挑一个填到 MODEL_NAME
    python scripts/GEMINI.py                 # 跑全量召回 + 误检 + 置信度统计
    python scripts/GEMINI.py --pos-only      # 只跑召回（prompt 迭代时）
"""

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

import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ============ 配置区（按需修改） ============
# 项目根目录 = 本文件的上上级，.env 与 outputs 都在这
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

# 把项目根加入 path，复用正式 agent 的 DETECT_PROMPT（单一真源）：
# 评测脚本测的就是 agent 实跑的那条 prompt，结论可直接迁移到 detect 节点。
sys.path.insert(0, ROOT)
from prompts import DETECT_PROMPT  # noqa: E402

API_KEY = os.getenv("GOOGLE_API_KEY")

IMAGE_DIR = os.path.join(ROOT, "outputs", "eval_patches")

# ⚠️ 先跑 --list-models 确认当前账号可用模型，再改这里（避免撞 404/已退役）。
# 注：本账号 3.5-pro 暂未开放；最新可用 pro 为 gemini-3.1-pro-preview（2026-06-15 实测
#     免费层即可调用，0 额度错误——此前「pro 免费额度=0」结论对本账号不成立）。
# 评测结论（2026-06-15）：gemini-3.5-flash 判别力 86.4 登顶，已设为默认。可用 --model 横比。
MODEL_NAME = "gemini-3.5-flash"

TEMPERATURE = 0.0     # 求可复现（Gemini 接受 temp=0，非 OpenAI 推理模型那种必须 1 的限制）
MAX_TOKENS = 2048     # 调高防 thinking 模型 reasoning token 吃光导致空响应假漏检
NEG_LIMIT = 0         # 误检随机抽查张数；0 = 全部 162 张（全量确认 FP）
SEED = 42             # 固定随机种子，抽样可复现

# present 与 confidence 一致性判定阈值：present=true 却 conf<此值，或 present=false 却
# conf>此值，记为「矛盾」——detect 排序依赖 conf 与 present 同向，矛盾即污染置信度排序。
CONF_CONSISTENCY_THRESHOLD = 0.5

# 免费层限流防护
REQUEST_INTERVAL = 1.0  # 每次请求间主动间隔秒数（flash 系列 RPM 较高，可调小；撞 429 会自动退避重试，太激进就回调）
MAX_RETRIES = 5         # 撞 429 后的重试次数
# ==========================================


def list_models(client):
    """列出当前账号下支持 generateContent（即可用于视觉问答）的 Gemini 模型。"""
    print("当前可用模型（支持 generateContent）：")
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            print(f"  · {m.name}")


def query_gemini(client, image_path):
    """对单张图调用 Gemini，返回 (present, confidence, error)。

    用正式 DETECT_PROMPT，解析 present + confidence（与 detect 节点一致）：
      - present 缺失时回退读 found / has_waldo（兼容旧键名）。
      - confidence 缺失时回退：present=true → 0.8，present=false → 0.0（同 quick_config）。
    成功返回 (bool, float, None)；失败返回 (None, None, 错误串)。
    撞 429 限流时按服务器给的 retryDelay（缺省指数退避）等待重试，免费层下能跑完全量。
    """
    img = PIL.Image.open(image_path)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[DETECT_PROMPT, img],
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS,
                    response_mime_type="application/json",  # 强制 JSON，省去清洗 markdown
                ),
            )
            content = (response.text or "").strip()
            if not content:
                return None, None, "空响应（可能 token 截断或被安全策略拦截）"
            data = json.loads(content)
            present = bool(data.get("present", data.get("found", data.get("has_waldo", False))))
            conf = data.get("confidence")
            conf = float(conf) if conf is not None else (0.8 if present else 0.0)
            return present, conf, None
        except json.JSONDecodeError as e:
            return None, None, f"JSON 解析失败: {e} | 原始: {content!r}"
        except Exception as e:
            msg = str(e)
            is_429 = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            if is_429 and attempt < MAX_RETRIES - 1:
                m = re.search(r"retryDelay.*?(\d+)\s*s", msg)
                wait = (int(m.group(1)) + 2) if m else 5 * (2 ** attempt)
                print(f"      ⏳ 限流，等待 {wait}s 重试 ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            return None, None, f"API 调用异常: {e}"


def _img_id(filename):
    """从 `18_pos.jpg` / `10_neg3.jpg` 取图号，用于排序。"""
    return int(os.path.basename(filename).split("_")[0])


def _is_inconsistent(present, conf):
    """present 与 confidence 是否矛盾（污染 detect 排序的信号）。"""
    if present and conf < CONF_CONSISTENCY_THRESHOLD:
        return True
    if not present and conf > CONF_CONSISTENCY_THRESHOLD:
        return True
    return False


def _pct(sorted_vals, q):
    """sorted_vals 的 q 分位（q∈[0,1]，最近秩法）。空列表返回 None。"""
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def _conf_separation_report(records):
    """核心分析：模型说「没有」(present=false) 时，漏检(FN) 的 conf 能否和真负样本(TN) 分开？

    若能分开 → 可对低分的 present=false 样本「再捞一遍」救回漏检；
    若分不开 → 模型说没有时无论对错都一样自信，conf 对救漏检无用。

    同时给出 detect 真正会用的「派生 Waldo 分数」= present? conf : (1-conf)，
    看它能否把真有 Waldo 的 patch 排在真没有的之上（即 conf 对排序是否有救）。
    """
    # 按真值 × present 四象限拆分
    fn = [(f, c) for g, f, p, c in records if g == "pos" and not p]   # 漏检：真有却说没有
    tn = [c for g, f, p, c in records if g == "neg" and not p]        # 真负：真没且说没有
    tp = [c for g, f, p, c in records if g == "pos" and p]            # 真有且说有
    fp = [c for g, f, p, c in records if g == "neg" and p]            # 误检：真没却说有

    print("\n" + "─" * 70)
    print("【漏检 vs 真负样本：conf 能否分开】(回答「低 conf 再捞一遍能否救漏检」)")
    if not fn:
        print("  本次无漏检（present=false 的正样本），无法评估——召回越高这块样本越少。")
    elif not tn:
        print("  本次无真负样本（可能 --pos-only 或负样本全 ERR），无法对比。")
    else:
        tn_sorted = sorted(tn)
        print(f"  漏检(FN) {len(fn)} 个，conf：" +
              ", ".join(f"{f}={c:.2f}" for f, c in fn))
        print(f"  真负(TN) {len(tn)} 个，conf 分布："
              f"min={min(tn):.2f}  中位={_pct(tn_sorted, 0.5):.2f}  "
              f"max={max(tn):.2f}  均值={sum(tn) / len(tn):.3f}")
        # 关键判定：要救回某个 FN，需把 conf ≤ 它的 present=false 都重检；
        # 代价 = 同时被卷入的真负样本数。卷入越多 → 越分不开。
        for f, c in fn:
            swept = sum(1 for t in tn if t <= c)
            verdict = "✅ 可分（卷入少）" if swept <= max(1, len(tn) * 0.1) else "❌ 分不开（淹没在真负中）"
            print(f"    · 设阈值救 {f}(conf≤{c:.2f})：会同时卷入 {swept}/{len(tn)} 个真负样本 → {verdict}")

    # 派生分数：detect 排序真正会用的口径
    if tp or tn or fn or fp:
        def score(present, conf):
            return conf if present else (1 - conf)
        s_waldo = [score(True, c) for c in tp] + [score(False, c) for _, c in fn]  # 真有 Waldo 的 patch
        s_nowaldo = [score(False, c) for c in tn] + [score(True, c) for c in fp]    # 真没 Waldo 的 patch
        if s_waldo and s_nowaldo:
            print("\n【派生 Waldo 分数 = present?conf:(1-conf)：detect 排序能否分开真有/真没】")
            print(f"  真有 Waldo 的 patch（n={len(s_waldo)}）：均值={sum(s_waldo)/len(s_waldo):.3f}  "
                  f"min={min(s_waldo):.2f}  max={max(s_waldo):.2f}")
            print(f"  真没 Waldo 的 patch（n={len(s_nowaldo)}）：均值={sum(s_nowaldo)/len(s_nowaldo):.3f}  "
                  f"min={min(s_nowaldo):.2f}  max={max(s_nowaldo):.2f}")
            print(f"  分数 gap（真有 − 真没 均值）："
                  f"{sum(s_waldo)/len(s_waldo) - sum(s_nowaldo)/len(s_nowaldo):+.3f}"
                  f"（>0 且越大，排序越能把真 Waldo 顶上去）")
    print("─" * 70)


def main():
    if not API_KEY:
        print("❌ 未找到 GOOGLE_API_KEY，请在项目根 .env 添加：GOOGLE_API_KEY=你的key")
        return

    client = genai.Client(api_key=API_KEY)

    # CLI: --model NAME 覆盖默认模型，方便横向对比不同模型而不必改文件
    global MODEL_NAME
    if "--model" in sys.argv:
        MODEL_NAME = sys.argv[sys.argv.index("--model") + 1]

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
    print(f"Gemini 评测  model={MODEL_NAME}  temp={TEMPERATURE}  "
          f"pos={len(pos)}  neg={len(neg)}")
    print("=" * 70)

    # ── 计时：墙钟总耗时 + 纯 API 耗时（含重试退避，不含主动限速 sleep）──
    t_wall_start = time.perf_counter()
    api_secs, n_calls = 0.0, 0

    # 置信度统计容器：正/负样本的 conf 列表 + present↔conf 矛盾明细
    pos_confs, neg_confs = [], []
    inconsistent = []  # [(文件名, present, conf), ...]
    # 逐样本明细（带真值标签），用于「漏检 vs 真负样本 conf 能否分开」分析
    records = []  # [(group, 文件名, present, conf), ...]  group ∈ {"pos","neg"}

    # ── 召回：正样本判 present=false 即漏检 ──
    print(f"\n── 召回（{len(pos)} 张正样本）" + "─" * 40)
    pos_fail, errors = [], 0
    for f in pos:
        t0 = time.perf_counter()
        present, conf, err = query_gemini(client, os.path.join(IMAGE_DIR, f))
        api_secs += time.perf_counter() - t0
        n_calls += 1
        if err:
            print(f"  [⚠️ ERR ] {f:<14} {err}")
            errors += 1
            continue
        pos_confs.append(conf)
        records.append(("pos", f, present, conf))
        if _is_inconsistent(present, conf):
            inconsistent.append((f, present, conf))
        if not present:
            pos_fail.append(_img_id(f))
        tag = "✓ OK  " if present else "✗ FAIL"
        print(f"  [{tag}] {f:<14} conf={conf:.2f}")
        time.sleep(REQUEST_INTERVAL)  # 主动限速，避开每分钟配额

    # ── 误检：负样本判 present=true 即误检（--pos-only 时跳过，prompt 迭代只看召回）──
    pos_only = "--pos-only" in sys.argv
    fp_files = []
    if pos_only:
        neg = []
        print("\n（--pos-only：跳过负样本/误检测试）")
    else:
        print(f"\n── 误检（{len(neg)} 张负样本，仅列被误检的）" + "─" * 18)
        for f in neg:
            t0 = time.perf_counter()
            present, conf, err = query_gemini(client, os.path.join(IMAGE_DIR, f))
            api_secs += time.perf_counter() - t0
            n_calls += 1
            if err:
                print(f"  [⚠️ ERR ] {f:<14} {err}")
                errors += 1
                continue
            neg_confs.append(conf)
            records.append(("neg", f, present, conf))
            if _is_inconsistent(present, conf):
                inconsistent.append((f, present, conf))
            if present:
                fp_files.append(f)
                print(f"  [✗ 误检] {f:<14} conf={conf:.2f}")
            time.sleep(REQUEST_INTERVAL)  # 主动限速，避开每分钟配额

    # ── 汇总 ──
    # ⚠️ 分母用「实际判定成功数」而非文件总数：断网/限流导致的 ERR 样本既不进分子也
    #    不进分母，否则会系统性高估召回、低估误检（断网那次 2/162 即假象）。
    n_pos, n_neg = len(pos), len(neg)            # 文件总数（仅用于显示覆盖度）
    n_pos_ok, n_neg_ok = len(pos_confs), len(neg_confs)  # 实际判定成功数
    recall = (n_pos_ok - len(pos_fail)) / n_pos_ok * 100 if n_pos_ok else 0.0
    fp_rate = len(fp_files) / n_neg_ok * 100 if n_neg_ok else 0.0
    avg_pos_conf = sum(pos_confs) / len(pos_confs) if pos_confs else 0.0
    avg_neg_conf = sum(neg_confs) / len(neg_confs) if neg_confs else 0.0

    print("\n" + "=" * 70)
    print(f"召回率 (recall)          : {n_pos_ok - len(pos_fail)}/{n_pos_ok} = {recall:.1f}%"
          f"  (判定 {n_pos_ok}/{n_pos} 张)")
    if not pos_only:
        print(f"误检率 (false positive)  : {len(fp_files)}/{n_neg_ok} = {fp_rate:.1f}%"
              f"  (判定 {n_neg_ok}/{n_neg} 张)")
        print(f"判别力 (召回 − 误检)      : {recall - fp_rate:.1f}")

    # ── 置信度可靠性（detect 落地的硬门槛：conf 必须可排序、与 present 一致）──
    print("─" * 70)
    print(f"正样本平均 conf          : {avg_pos_conf:.3f}  (n={len(pos_confs)})")
    if not pos_only:
        print(f"负样本平均 conf          : {avg_neg_conf:.3f}  (n={len(neg_confs)})")
        print(f"置信度 gap (正 − 负)     : {avg_pos_conf - avg_neg_conf:+.3f}  "
              f"（越大越能靠阈值/排序分开真假）")
    n_inc = len(inconsistent)
    n_judged = len(pos_confs) + len(neg_confs)
    inc_rate = n_inc / n_judged * 100 if n_judged else 0.0
    print(f"present↔conf 矛盾        : {n_inc}/{n_judged} = {inc_rate:.1f}% "
          f"（present 与 conf 反向，会污染 detect 排序；0 最佳）")
    if inconsistent:
        for fn, p, c in inconsistent:
            print(f"    ⚠️ {fn:<14} present={p}  conf={c:.2f}")

    # 核心：漏检 vs 真负样本 conf 能否分开 + 派生分数排序分离度
    _conf_separation_report(records)

    wall = time.perf_counter() - t_wall_start
    print(f"召回失败图号             : {pos_fail}")
    print(f"API/解析错误             : {errors}")
    print(f"总耗时 (墙钟)            : {wall:.1f}s ({wall / 60:.1f}min)")
    print(f"纯 API 耗时             : {api_secs:.1f}s  | 平均 "
          f"{api_secs / n_calls:.2f}s/张（共 {n_calls} 次调用）"
          if n_calls else "纯 API 耗时             : 无调用")
    print("=" * 70)
    print("⚠️ 提醒：① 高召回 + 高误检 = 无脑说有，不可用（参考 qwen-vl-plus）；"
          "判别力(召回−误检)才是关键，gpt-5.5 基线为 ~69。")
    print("         ② conf 必须与 present 一致、正负样本能拉开 gap——否则即便判别力够，"
          "detect 的阈值过滤 + top-3 排序仍会失效（mini/Qwen 即栽在此）。")


if __name__ == "__main__":
    main()
