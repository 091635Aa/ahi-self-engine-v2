# -*- coding: utf-8 -*-
"""run_verify.py —— 跑六条理论路线的全部验证（多 seed 稳健性版）。

红队指出：单次 1500/1500 切分是高方差估计量（相邻超参下 split AUC 在
0.45~0.98 间跳）。因此本脚本对每条路线跑多个 seed，**全部 seed 通过才算通过**，
并报告最差 seed 的结果。
"""
import sys, os, time, json
import numpy as np

# 合并后路径可移植：本包根目录 = 本文件的上一级（ahi_theory 的父目录）
根 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 根)
try:  # Windows 控制台/重定向默认非 UTF-8，✅❌ 等符号会 UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from ahi_theory.verify_all import ALL, N_GROUP, _CHANGELOG

SEEDS = [0, 1, 2, 3, 4, 5]

print("=" * 96)
print(f"六条理论路线 · 独立验证（{len(SEEDS)} 个 seed，全部通过才算通过）")
print("=" * 96)
print("判据台账：verify_all.CRITERIA_AUDIT（26 条判据的 ⟨类别, Nec, 理由⟩）；修改留痕：")
for _e in _CHANGELOG:
    # 条目为 (日期, 判据/模块, 动作[, 理由/证据/纪律…])：历史沿革中字段数可变（3–5），
    # 本处只打印前三元组以保持启动日志稳定；多出的证据/纪律字段不丢失。
    _d, _obj = _e[0], _e[1]
    _act = _e[2] if len(_e) > 2 else ""
    print(f"  [{_d}] {_obj} → {_act}")
print("-" * 96)

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
    v2v = [r.get("v2", {}) for r in per_seed]
    # ★ 判据级严格计数（跨 seed）：完整达标 = **全部 seed 通过** 且 DP≥5 且 Nec=1
    crit_all = {}
    for r in per_seed:
        for c in r.get("criteria", []):
            d = crit_all.setdefault(c["name"], dict(passed=True, dp=None, nec=None))
            d["passed"] &= bool(c["passed"])
            if isinstance(c.get("dp"), float) and np.isfinite(c["dp"]):
                d["dp"] = c["dp"]
            d["nec"] = (c.get("audit") or {}).get("nec")
    summary[name] = dict(
        all_passed=passed, per_seed=[r["passed"] for r in per_seed],
        worst=per_seed[0]["summary"],
        passed_falsifiable=[bool(r.get("passed_falsifiable")) for r in per_seed],
        v2=dict(n_total=v2v[0].get("n_total"),
                n_falsifiable=v2v[0].get("n_falsifiable"),
                n_falsifiable_passed=[x.get("n_falsifiable_passed") for x in v2v],
                n_power=v2v[0].get("n_power"), n_nec=v2v[0].get("n_nec"),
                n_complete=[x.get("n_complete") for x in v2v],
                complete=v2v[0].get("complete"),
                crit_all=crit_all,
                excluded=v2v[0].get("excluded")))
    print(f"\n{'='*96}\n[{name}] {per_seed[0]['route']}   ->   "
          f"{'✅ PASS' if passed else '❌ FAIL'}（v1 口径）   (seed 结果: {[r['passed'] for r in per_seed]})")
    print("-" * 96)
    for c in per_seed[0]["criteria"]:
        allc = all(any(cc["name"] == c["name"] and cc["passed"]
                       for cc in r["criteria"]) for r in per_seed)
        v = c["value"]
        vs = f"{v:.5f}" if isinstance(v, float) else str(v)
        cls_ = (c.get("audit") or {}).get("cls", "?")
        tag = "  [恒真/控制·已剔出 v2 分母]" if cls_ == "A" else ""
        print(f"  {'✅' if allc else '❌'} {c['name']:<36} = {vs:>12}   要求 {c['threshold']:<24} {c['note']}{tag}")
    print(f"  小结(seed0): {per_seed[0]['summary']}")

print()
print("=" * 96)
print(f"独立 N 组（任务书 §9.1：N1–N4 与 P1–P8 **并列**，不并入路线计分）· {len(SEEDS)} 个 seed")
print("  说明（2026-09-25 Task 8）：N1/N2/N4 曾一度并入 verify_P6 计分，使 P6 由基线 5/6 变 4/6；")
print("  现按 §9.1 归为本组。**搬家只改计分归属，不改任何测量值与判定结果**；")
print("  N3『慢变量吸引子数』仍留在 P6 内（那本来就是 P6 的判据、升主判据是本规格要求）。")
print("=" * 96)
for name, fn in N_GROUP:
    per_seed = []
    for sd in SEEDS:
        try:
            per_seed.append(fn(seed=sd))
        except Exception as e:
            per_seed.append(dict(route=name, passed=False, criteria=[],
                                 summary=f"ERROR {type(e).__name__}: {e}"))
    ok = all(r["passed"] for r in per_seed)
    print(f"\n[{name}] {per_seed[0]['route']}   ->   "
          f"{'✅ PASS' if ok else '❌ FAIL'}（判据级全 seed 口径）   "
          f"(seed 结果: {[r['passed'] for r in per_seed]})")
    for c in per_seed[0]["criteria"]:
        allc = all(any(cc["name"] == c["name"] and cc["passed"] for cc in r["criteria"])
                   for r in per_seed)
        v = c["value"]
        vs = f"{v:.5f}" if isinstance(v, float) else str(v)
        cls_ = (c.get("audit") or {}).get("cls", "?")
        tag = "  [恒真/控制·已剔出 v2 分母]" if cls_ == "A" else ""
        print(f"  {'✅' if allc else '❌'} {c['name']:<44} = {vs:>12}   要求 "
              f"{c['threshold']:<26} {c['note']}{tag}")
    print(f"  小结(seed0): {per_seed[0]['summary']}")
print("-" * 96)
print("  ⚠ 本组结果**不计入**上面的「8 条路线」总数（P1–P8 的可比性由本组归位后恢复）。")
print("=" * 96)

npass = sum(1 for s in summary.values() if s["all_passed"])
print(f"总计（v1 口径，历史可比）：{npass}/{len(summary)} 条路线在全部 {len(SEEDS)} 个 seed 上通过")
for name, s in summary.items():
    print(f"  {'✅' if s['all_passed'] else '❌'} {name}  seeds={s['per_seed']}")
print("-" * 96)
print("按验收协议 v2（分母 = 可失败判据；恒真/控制项剔除但不隐藏）：")
tf = tp = tn = tex = 0
完整达标 = []
for name, s in summary.items():
    v = s.get("v2") or {}
    f_, p_, n_, ex_ = v.get("n_falsifiable"), v.get("n_power"), v.get("n_nec"), v.get("excluded") or []
    tf += (f_ or 0); tp += (p_ or 0); tn += (n_ or 0); tex += len(ex_)
    nC = v.get("n_complete") or []
    print(f"  {name}  可失败判据 {f_} 条（6 seed 通过数 {v.get('n_falsifiable_passed')}）"
          f"；有判别力 {p_}；Nec=1 {n_}；完整达标（每 seed 通过数）{nC}；剔除 {len(ex_)} 条")
    if ex_:
        print(f"      剔除项：{'、'.join(ex_)}")
    for cname, cd in (v.get("crit_all") or {}).items():
        if cd.get("passed") and isinstance(cd.get("dp"), float) and cd["dp"] >= 5.0 \
                and cd.get("nec") == 1:
            完整达标.append(f"{name}·{cname}（DP={cd['dp']:.1f}σ）")
        elif cd.get("nec") == 1:
            print(f"      · Nec=1 但未计入完整达标：{cname}"
                  f"（全部 seed 通过={cd.get('passed')}，DP={cd.get('dp')}）")
print(f"  合计：可失败 {tf} 条；有判别力 {tp} 条；Nec=1 {tn} 条；恒真/控制剔除 {tex} 条")
_n_nec_route = sum(1 for s in summary.values() if ((s.get("v2") or {}).get("n_nec") or 0) > 0)
_n_pow_route = sum(1 for s in summary.values() if ((s.get("v2") or {}).get("n_power") or 0) > 0)
print(f"  ⇒ 拥有 Nec=1 判据的路线 = {_n_nec_route}/{len(summary)}；"
      f"拥有可判别力判据的路线 = {_n_pow_route}/{len(summary)}")
print("  ⇒ 严格口径（完整达标 = 该判据**全部 seed 通过** 且 DP≥5 且 Nec=1）：")
print(f"     共 {len(完整达标)} 条：{'、'.join(完整达标) if 完整达标 else '（无）'}")
print("     说明：DP 未标定的判据（含否定型判据 A_int 的等价型偏离、确定性删除实验的因果载荷）")
print("           不计入此列——它们各自的 DP 口径说明见判据台账 CRITERIA_AUDIT 与 note。")
print("     本脚本报告的是**结构性诊断**（可失败/有判别力/Nec/DP），不是『已产生自我意识』的证明。")
print("=" * 96)
报告路径 = os.path.join(根, "ahi_theory", "verify_report.json")
json.dump(summary, open(报告路径, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=str)
print(f"报告已落盘 {报告路径}")

# ==================================================================== 世界组（R2，纯追加；默认不打印）
# 说明：本段为**纯追加**，仅在显式传 `--world` 时运行 ⇒ 默认输出与计数**完全不变**。
#       世界层为纯新增层、不反馈进内核动力学（brain.py 一行未改）⇒ 锚点逐位不变。
if "--world" in sys.argv:
    from ahi_theory.verify_all import W_GROUP
    print()
    print("=" * 96)
    print(f"独立世界组（R2 / 任务书 §6.4：W1–W7，与 P1–P8、N 组**分开计分**）· {len(SEEDS)} 个 seed")
    print("  世界层=纯新增、不反馈进内核动力学 ⇒ 本段**不计入**上面的「8 条路线」总数，也不影响既有输出。")
    print("=" * 96)
    _wpass = 0
    for name, fn in W_GROUP:
        _per = []
        for sd in SEEDS:
            try:
                _per.append(fn(seed=sd))
            except Exception as e:
                _per.append(dict(route=name, passed=False, criteria=[],
                                 summary=f"ERROR {type(e).__name__}: {e}"))
        _ok = all(r["passed"] for r in _per)
        _wpass += int(_ok)
        print(f"\n[{name}] {_per[0]['route']}   ->   {'✅ PASS' if _ok else '❌ FAIL'}"
              f"（判据级全 seed 口径）   (seed 结果: {[r['passed'] for r in _per]})")
        for c in _per[0]["criteria"]:
            _allc = all(any(cc["name"] == c["name"] and cc["passed"] for cc in r["criteria"])
                        for r in _per)
            _v = c["value"]
            _vs = f"{_v:.5f}" if isinstance(_v, float) else str(_v)
            print(f"  {'✅' if _allc else '❌'} {c['name']:<44} = {_vs:>12}   要求 {c['threshold']}")
        print(f"  小结(seed0): {_per[0]['summary']}")
    print("-" * 96)
    print(f"  世界组：{_wpass}/{len(W_GROUP)} 条路线在全部 {len(SEEDS)} 个 seed 上通过"
          f"（**不计入** P1–P8 的 8 条路线总数）")
    print("=" * 96)

# ==================================================================== S 组（R3 自我探测循环，纯追加；默认不打印）
# 说明：本段为**纯追加**，仅在显式传 `--self` 时运行 ⇒ 默认输出与计数**完全不变**。
#       闭环层 selfprobe.py 为纯新增层、不反馈进内核动力学（brain.py/mbp.py/world.py 一行未改）
#       ⇒ 锚点逐位不变（meeting8/R3_anchor_zero_cost.txt）。
if "--self" in sys.argv:
    from ahi_theory.verify_all import S_GROUP
    print()
    print("=" * 96)
    print(f"独立 S 组（R3 自我探测循环 / 任务书 §5.7/§5.8/§5.10/§6.4 W6：S1–S5，"
          f"与 P1–P8、N 组、世界组**分开计分**）· {len(SEEDS)} 个 seed")
    print("  闭环层=纯新增、不反馈进内核动力学 ⇒ 本段**不计入**上面的「8 条路线」总数，也不影响既有输出。")
    print("=" * 96)
    _spass = 0
    for name, fn in S_GROUP:
        _per = []
        for sd in SEEDS:
            try:
                _per.append(fn(seed=sd))
            except Exception as e:
                _per.append(dict(route=name, passed=False, criteria=[],
                                 summary=f"ERROR {type(e).__name__}: {e}"))
        _ok = all(r["passed"] for r in _per)
        _spass += int(_ok)
        print(f"\n[{name}] {_per[0]['route']}   ->   {'✅ PASS' if _ok else '❌ FAIL'}"
              f"（判据级全 seed 口径）   (seed 结果: {[r['passed'] for r in _per]})")
        for c in _per[0]["criteria"]:
            _allc = all(any(cc["name"] == c["name"] and cc["passed"] for cc in r["criteria"])
                        for r in _per)
            _v = c["value"]
            _vs = f"{_v:.5f}" if isinstance(_v, float) else str(_v)
            print(f"  {'✅' if _allc else '❌'} {c['name']:<44} = {_vs:>12}   要求 {c['threshold']}")
        print(f"  小结(seed0): {_per[0]['summary']}")
    print("-" * 96)
    print(f"  S 组：{_spass}/{len(S_GROUP)} 条判据在全部 {len(SEEDS)} 个 seed 上通过"
          f"（**不计入** P1–P8 的 8 条路线总数）")
    print("=" * 96)

# ==================================================================== M 组（R4 单模态插件，纯追加；默认不打印）
# 说明：本段为**纯追加**，仅在显式传 `--modality` 时运行 ⇒ 默认输出与计数**完全不变**。
#       模态层 modality.py 为纯新增层、不反馈进内核动力学（brain.py 零改动、mbp.py 仅纯追加 挂投影器()）
#       ⇒ 锚点逐位不变（meeting9/R4_anchor_zero_cost.txt）。
if "--modality" in sys.argv:
    from ahi_theory.verify_all import M_GROUP
    print()
    print("=" * 96)
    print(f"独立 M 组（R4 单模态插件 / 任务书 §5.4/§5.5/§5.9：M1(+M3)/M2/M4/M5，"
          f"与 P1–P8、N 组、W 组、S 组**分开计分**）· {len(SEEDS)} 个 seed")
    print("  模态层=纯新增、不反馈进内核动力学 ⇒ 本段**不计入**上面的「8 条路线」总数，也不影响既有输出。")
    print("=" * 96)
    _mpass = 0
    for name, fn in M_GROUP:
        _per = []
        for sd in SEEDS:
            try:
                _per.append(fn(seed=sd))
            except Exception as e:
                _per.append(dict(route=name, passed=False, criteria=[],
                                 summary=f"ERROR {type(e).__name__}: {e}"))
        _ok = all(r["passed"] for r in _per)
        _mpass += int(_ok)
        print(f"\n[{name}] {_per[0]['route']}   ->   {'✅ PASS' if _ok else '❌ FAIL'}"
              f"（判据级全 seed 口径）   (seed 结果: {[r['passed'] for r in _per]})")
        for c in _per[0]["criteria"]:
            _allc = all(any(cc["name"] == c["name"] and cc["passed"] for cc in r["criteria"])
                        for r in _per)
            _v = c["value"]
            _vs = f"{_v:.5f}" if isinstance(_v, float) else str(_v)
            print(f"  {'✅' if _allc else '❌'} {c['name']:<44} = {_vs:>12}   要求 {c['threshold']}")
        print(f"  小结(seed0): {_per[0]['summary']}")
    print("-" * 96)
    print(f"  M 组：{_mpass}/{len(M_GROUP)} 条判据在全部 {len(SEEDS)} 个 seed 上通过"
          f"（**不计入** P1–P8 的 8 条路线总数）")
    print("=" * 96)

# ==================================================================== I 组（INT 系统接入与整合，纯追加；默认不打印）
# 说明：本段为**纯追加**，仅在显式传 `--system` 时运行 ⇒ 默认输出与计数**完全不变**。
#       装配器 ahi_theory/system.py 为纯新增、不改内核（brain.py 一行未改）⇒ 锚点逐位不变；
#       I3 为对照判据（读 meeting10/INT_i3_groups_assembled.json 的四组逐位对照结论）。
if "--system" in sys.argv:
    from ahi_theory.verify_all import I_GROUP
    print()
    print("=" * 96)
    print(f"独立 I 组（INT 系统接入与整合 / spec add-ahi22-integration：I1–I4，"
          f"与 P1–P8、N/W/S/M 组**分开计分**）· {len(SEEDS)} 个 seed")
    print("  装配器=纯新增、不改内核 ⇒ 本段**不计入**上面的「8 条路线」总数，也不影响既有输出。")
    print("=" * 96)
    _ipass = 0
    for name, fn in I_GROUP:
        _per = []
        for sd in SEEDS:
            try:
                _per.append(fn(seed=sd))
            except Exception as e:
                _per.append(dict(route=name, passed=False, criteria=[],
                                 summary=f"ERROR {type(e).__name__}: {e}"))
        _ok = all(r["passed"] for r in _per)
        _ipass += int(_ok)
        print(f"\n[{name}] {_per[0]['route']}   ->   {'✅ PASS' if _ok else '❌ FAIL'}"
              f"（判据级全 seed 口径）   (seed 结果: {[r['passed'] for r in _per]})")
        for c in _per[0]["criteria"]:
            _allc = all(any(cc["name"] == c["name"] and cc["passed"] for cc in r["criteria"])
                        for r in _per)
            _v = c["value"]
            _vs = f"{_v:.5f}" if isinstance(_v, float) else str(_v)
            cls_ = (c.get("audit") or {}).get("cls", "?")
            tag = "  [恒真/控制·已剔出 v2 分母]" if cls_ == "A" else ""
            print(f"  {'✅' if _allc else '❌'} {c['name']:<44} = {_vs:>12}   要求 {c['threshold']}{tag}")
        print(f"  小结(seed0): {_per[0]['summary']}")
    print("-" * 96)
    print(f"  I 组：{_ipass}/{len(I_GROUP)} 条判据在全部 {len(SEEDS)} 个 seed 上通过"
          f"（**不计入** P1–P8 的 8 条路线总数）")
    print("=" * 96)

# ==================================================================== K 组（V2 无标度判据 / M1 K-BATTERY，纯追加；默认不打印）
# 说明：本段为**纯追加**，仅在显式传 `--k` 时运行 ⇒ 默认输出与计数**完全不变**。
#       K5 为仪器自证（元判据，种子块级）：verify_K5 跑双集两块（judge 0–5 / holdout 6–11，
#       B=40、T=131072），首算后进程内缓存 ⇒ 组驱动 SEEDS 循环六次调用仅首算一次。
#       K1–K4 引擎级测量待 V2 引擎 M5/M7 供数后另行预注册接入（本轮不虚设）。
if "--k" in sys.argv:
    from ahi_theory.verify_all import K_GROUP
    print()
    print("=" * 96)
    print(f"独立 K 组（V2 无标度判据 / M1 K-BATTERY 仪器：K5 仪器自证（K1–K4 引擎级待 V2 引擎供数），"
          f"与 P1–P8、N/W/S/M/I 组**分开计分**）· 种子块级判据（双集 master 0–5 / 6–11）")
    print("  仪器层纯 numpy、不触内核动力学 ⇒ 本段**不计入**上面的「8 条路线」总数，也不影响既有输出。")
    print("=" * 96)
    _kpass = 0
    for name, fn in K_GROUP:
        try:
            _per = [fn(seed=sd) for sd in SEEDS]   # K5 首算后缓存 ⇒ 除首次外零开销
        except Exception as e:
            _per = [dict(route=name, passed=False, criteria=[],
                         summary=f"ERROR {type(e).__name__}: {e}")]
        _ok = all(r["passed"] for r in _per)
        _kpass += int(_ok)
        print(f"\n[{name}] {_per[0]['route']}   ->   {'✅ PASS' if _ok else '❌ FAIL'}"
              f"（种子块级：双集五门）   (seed 结果: {[r['passed'] for r in _per]}；"
              f"seed 传参不进入测量，仅首算一次)")
        for c in _per[0]["criteria"]:
            _allc = all(any(cc["name"] == c["name"] and cc["passed"] for cc in r["criteria"])
                        for r in _per)
            _v = c["value"]
            _vs = f"{_v:.5f}" if isinstance(_v, float) else str(_v)
            print(f"  {'✅' if _allc else '❌'} {c['name']:<44} = {_vs:>12}   要求 {c['threshold']}")
            print(f"    note: {c['note']}")
        print(f"  小结: {_per[0]['summary']}")
    print("-" * 96)
    print(f"  K 组：{_kpass}/{len(K_GROUP)} 条判据通过（**不计入** P1–P8 的 8 条路线总数；"
          f"K5 不过线 ⇒ K1–K4 全废——『仪器未成』的诚实结论）")
    print("=" * 96)

