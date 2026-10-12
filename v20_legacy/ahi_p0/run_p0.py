# -*- coding: utf-8 -*-
"""run_p0.py —— 一键跑 P0 全部验收测试。"""
import sys, time
sys.path.insert(0, "/root/AHI")
from ahi_p0 import diagnostics as D

TESTS = [
    ("D1", lambda: D.d1_transfer_entropy(T=15000)),
    ("D2", D.d2_mask_test),
    ("D3", D.d3_control_twin),
    ("D4", lambda: D.d4_anesthesia_scan(steps=2500)),
    ("D5", D.d5_lambda_acceptance),
    ("D6", D.d6_threshold_quantiles),
    ("D7", D.d7_saturation_guard),
]

print("=" * 92)
print("ahi_p0 验收测试 —— P0 五项缺陷的修复是否生效")
print("=" * 92)
results = []
for name, fn in TESTS:
    t0 = time.time()
    try:
        r = fn()
        r["secs"] = time.time() - t0
    except Exception as e:
        r = dict(test=name, verdict="ERROR", detail=f"{type(e).__name__}: {e}", secs=time.time() - t0)
    results.append((name, r))
    print(f"\n[{name}] {r['test']}")
    print(f"      {r['verdict']}  ({r['secs']:.1f}s)")
    print(f"      {r['detail']}")

print()
print("=" * 92)
npass = sum(1 for _, r in results if r["verdict"] == "PASS")
print(f"结果：{npass}/{len(results)} 通过")
for name, r in results:
    mark = {"PASS": "✅", "FAIL": "❌"}.get(r["verdict"], "⚠️")
    print(f"  {mark} {name}  {r['test']}")
print("=" * 92)

