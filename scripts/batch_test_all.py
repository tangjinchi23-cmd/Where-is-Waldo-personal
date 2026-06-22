"""批量跑全量原图，复刻 main.py 的三态判断，汇总结果。

用法：python scripts/batch_test_all.py
输出：逐图打印 + 末尾汇总表；结果同时落到 outputs/batch_all_results.json。
"""

import json
import os
import sys
import time

# 项目根加入 sys.path（脚本在 scripts/ 下）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from agent import run_pipeline

IMAGES_DIR = "original-images"
# 1..19 + OIP，跳过 *_annotated
NAMES = [str(i) for i in range(1, 20)] + ["OIP"]
SUMMARY_PATH = "outputs/batch_all_results.json"


def interpret(state: dict) -> dict:
    """复刻 main.py 的三态判断。"""
    result = state.get("verified_result")
    candidates = state.get("candidates") or []
    verify_ran = any("verify_confidence" in c for c in candidates)
    n_cand = len(candidates)
    if result:
        return {"found": True, "mode": "verified", "bbox": result, "candidates": n_cand}
    if candidates and not verify_ran:
        best = candidates[0]
        bbox = best.get("orig_bbox") or best.get("patch_bbox")
        return {"found": True, "mode": "detect-only", "bbox": bbox, "candidates": n_cand}
    return {"found": False, "mode": "not-found", "bbox": None, "candidates": n_cand}


def main():
    images = [(n, os.path.join(IMAGES_DIR, f"{n}.jpg")) for n in NAMES]
    images = [(n, p) for n, p in images if os.path.exists(p)]

    # 断点续跑：已写入 JSON 的图跳过，保留其结果
    results = []
    done_names = set()
    if os.path.exists(SUMMARY_PATH):
        try:
            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                results = json.load(f)
            done_names = {r["name"] for r in results}
        except (json.JSONDecodeError, KeyError):
            results = []
    todo = [(n, p) for n, p in images if n not in done_names]
    print(f"[batch] {len(images)} total, {len(done_names)} done, {len(todo)} to run: "
          f"{[n for n, _ in todo]}", flush=True)

    for idx, (name, path) in enumerate(todo, 1):
        print(f"\n===== [{idx}/{len(todo)}] {name}.jpg =====", flush=True)
        t0 = time.time()
        try:
            state = run_pipeline(path)
            info = interpret(state)
            info["error"] = None
        except Exception as exc:
            info = {"found": False, "mode": "exception", "bbox": None,
                    "candidates": None, "error": f"{type(exc).__name__}: {exc}"}
        info["name"] = name
        info["elapsed_s"] = round(time.time() - t0, 1)
        results.append(info)
        print(f"[result] {name}: found={info['found']} mode={info['mode']} "
              f"bbox={info['bbox']} candidates={info['candidates']} "
              f"elapsed={info['elapsed_s']}s err={info['error']}", flush=True)
        # 每图都落盘，中途挂掉也保住已完成的
        os.makedirs("outputs", exist_ok=True)
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # 汇总
    print("\n" + "=" * 60, flush=True)
    print("[batch] SUMMARY", flush=True)
    found = sum(1 for r in results if r["found"])
    print(f"  found {found}/{len(results)}", flush=True)
    for r in results:
        flag = "OK " if r["found"] else "MISS"
        print(f"  {flag} {r['name']:>4}  mode={r['mode']:<11} "
              f"cand={r['candidates']} bbox={r['bbox']} {r['elapsed_s']}s", flush=True)
    print(f"\n[batch] results saved → {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
