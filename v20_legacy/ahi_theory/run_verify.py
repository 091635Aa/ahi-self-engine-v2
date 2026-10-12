# -*- coding: utf-8 -*-
"""run_verify.py —— 跑六条理论路线的全部验证（多 seed 稳健性版）。

红队指出：单次 1500/1500 切分是高方差估计量（相邻超参下 split AUC 在
0.45~0.98 间跳）。因此本脚本对每条路线跑多个 seed，**全部 seed 通过才算通过**，
并报告最差 seed 的结果。
"""
import sys, os, time, json

# 合并后路径可移植：本包根目录 = 本文件的上一级（ahi_theory 的父目录）
根 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 根)
try:  # Windows 控制台/重定向默认非 UTF-8，✅❌ 等符号会 UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from ahi_theory.verify_all import ALL

SEEDS = [0, 1, 2, 3, 4, 5]

print("=" * 96)
print(f"六条理论路线 · 独立验证（{len(SEEDS)} 个 seed，全部通过才算通过）")
print("=" * 96)

summary = {}
for name, fn in ALL:
    per_seed = []
    for sd in SEEDS:
        t0 = time.time()
        try:
            r = fn(seed=sd)
        except Exception as e:
            r = dict(route=name, passed=False, criteria=[], summary=f"ERROR {type(e).__name__}: {e}")
        r["secs"] = time.time() - t0
        per_seed.append(r)
    passed = all(r["passed"] for r in per_seed)
    summary[name] = dict(all_passed=passed, per_seed=[r["passed"] for r in per_seed],
                         worst=per_seed[0]["summary"])
    print(f"\n{'='*96}\n[{name}] {per_seed[0]['route']}   ->   "
          f"{'✅ PASS' if passed else '❌ FAIL'}   (seed 结果: {[r['passed'] for r in per_seed]})")
    print("-" * 96)
    for c in per_seed[0]["criteria"]:
        allc = all(any(cc["name"] == c["name"] and cc["passed"]
                       for cc in r["criteria"]) for r in per_seed)
        v = c["value"]
        vs = f"{v:.5f}" if isinstance(v, float) else str(v)
        print(f"  {'✅' if allc else '❌'} {c['name']:<36} = {vs:>12}   要求 {c['threshold']:<24} {c['note']}")
    print(f"  小结(seed0): {per_seed[0]['summary']}")

print()
print("=" * 96)
npass = sum(1 for s in summary.values() if s["all_passed"])
print(f"总计：{npass}/6 条路线在全部 {len(SEEDS)} 个 seed 上通过")
for name, s in summary.items():
    print(f"  {'✅' if s['all_passed'] else '❌'} {name}  seeds={s['per_seed']}")
print("=" * 96)
报告路径 = os.path.join(根, "ahi_theory", "verify_report.json")
json.dump(summary, open(报告路径, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=str)
print(f"报告已落盘 {报告路径}")

