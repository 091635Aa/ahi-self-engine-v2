# -*- coding: utf-8 -*-
"""ahi_theory/k_engine.py —— K1/K2 引擎级测量窗仪器（预注册规则的实现；影子模式纯新增）

预登记：ahi_theory/verify_all.py `_CHANGELOG` #77（2026-10-01『K1/K2 引擎级测量窗规则预注册
  + 12-seed 双集复测计划』，先于改码，规则自登记即冻结）。本文件=该规则文本的逐条实现，
  不含任何规则之外的自由度。

规格出处：meeting11/95_架构定稿.md §3.1 K1/K2 判据式（行 377/378）+§2 L2 参数表
  （T_rec^slow=16384，行 171）+ 承接 #76 V2 引擎影子期首测结论（全窗含发育瞬态）。

【2026-10-01 规则 v2 追加（#81 预登记，E5 会签：体制条件化分段测量）】
  #77 规则 v1 在 V2 引擎当前体制下结构性不可辨识（#78 已登记：事件缓冲足迹 134× 轴长、
  稳态窗为空）；E5 会签规则 v2（#81 冻结文本）：
    (V2-1) 分段依据=g_layer 上下文 c_t 序列（机制量）；wake 连续段内测 K1（e 轴三窗+α̂ 带；
           k* 轴双窗），offline 段单独列栏；两栏分别判定分别报告，wake 栏为主判据；
    (V2-2) kill 事件保留 ±T_rec^slow=16384 R1 拍缓冲（恢复语义，一字照抄 #77）；
    (V2-3) 离线相边界不再全缓冲：边界拍剔除（维护拍括号区间）+ wake 段内首尾 T_rec^fast=2000
           小缓冲；wake 段内 S 越阈事件缓冲降为 ±2000（E5 修订 a）；
    (V2-4) 发育期（首 S 越阈前）整段剔除（照抄 #77）；
    (V2-5) 最小可测段下限（E5 修订 b）：wake 段长 < 2×T_rec^fast+64K=69,536 R1 拍
           ⇒「段不足」降级不测、禁测残段；报告段长分布；
    (V2-6) offline 降级条款：offline 不可辨识⇒wake regime_ok ∧ offline 无 SCC 崩解（SCC 未评估
           =无崩解证据，scc_assessed=False 如实标注）；
    (V2-7) K2 口径不变（C2b dwell_csn）；K1-k*/K2 在 wake 段集应用同段集；
    (V2-8) e 轴逐段测量=各段剔除后最长连续可用子段；wake 栏主判定=最长可用子段判定；
    (V2-9) R1 为逻辑拍、无墙钟耦合；
    (V2-10) 规则只引用引擎事件时间戳/体制标签；判定值不进规则；12 seed 同一规则。
  c_t 序列途径：v2_engine.导出() 未含 c_t 序列（v2_engine 零改动）；由调用方（运行器）逐 R1 拍
  读 g_layer 实例只读属性 G.c_t 采样传入（本模块不 import v2_engine/g_layer——影子纪律同律）；
  g_layer._e_c史 deque(maxlen=512) 不足以覆盖 106K 拍（≥530 观测拍）故不可用作全程源。
  本文件实现=规则 v2（`测量窗_v2`/`引擎测量汇总_v2`）+#77 规则 v1 原样保留
  （`测量窗`/`引擎测量汇总`，历史对照开关——v1 在同数据上应仍全不可辨识，用于隔离规则变更效应）。
  【2026-10-01 规则 v2.1 追加（#84 预登记，E3 会签：K1 规则 v2 澄清——离线相括号主分段
  + IM_SATURATED 污染标记；实现=`IM饱和段`/`测量窗_v21`/`引擎测量汇总_v21`；v1/v2 原样保留
  为历史对照开关）】修正 #81 V2-1 的实现与文本偏差：seed0/1 实证 c_t 全程=0（g_layer 域切换
  协议在合成课程体制下未翻转）但离线相 144/145 次 ⇒ c_t 单独作分段依据漏切（wake 段=1 段
  104999 拍未按体制结构切分）。v2.1：wake 段切割依据=离线相括号（ecae S 压力离线相事件时间戳，
  运行器维护拍括号记法，精度 ≤20 拍，交叉验证=g_layer 拍数跳变）为主；c_t 分栏保留（c_t 序列
  照常采样落盘为对照旁证量；wake/offline 两栏语义保留）；IM_SATURATED 标记（E3 裁决④）：
  逐维护拍 |IM|≥102（hdc_core.井容量 同源值）满载连续段=污染段（井集合非平稳——G0b 冷却合并
  使井向量漂移），与离线相括号同级（分段级切割）从 K1/K2/Π 测量窗剔除——只前向标记不回填改史
  （e/k* 序列原样，剔除只作用于测量窗构造）；其余缓冲规则沿用 #81 一字不改
  （kill±16384 / 段首尾与段内 S 越阈±2000 / 发育期整段剔除 / 段不足下限 69,536）。

影子模式纪律（红线）：
  · 纯新增；默认不加载（无模块级副作用——消费方显式 import 才调用）；
  · 不 import brain/world/system/three_loop/mbp/v2_engine（只读 import ahi_theory.k_battery
    ——K1/K2 仪器复用，一字不改其判据式/阈值；v2_engine 由调用方传入导出量，本模块不反向依赖）；
  · 零拟合红线：全文件无 lstsq / linalg.inv·pinv / linalg.solve / scipy.optimize / 参数搜索构造
    （C4 tokenize 级可检）；无任何数据驱动标定；
  · 规则只引用引擎事件时间戳（机制量）；K1/K2 判定值（α̂/漂移/rsqn）不进入规则、不以 K1 结果挑窗。

规则 v1 文本（#77 冻结原文的实现对应；历史对照口径）：
  (1) 事件时间戳（引擎机制量）：t_run0 / t_S越阈（上升沿）/ t_离线开始 / t_离线结束 / t_kill。
  (2) e 轴有效条件：t > t_run0 ∧ t > min(t_S越阈) ∧ ∀t_e: |t − t_e| > 16384（R1 拍）。
  (3) k*(t) 轴：T_rec^slow=16384 R1 拍=⌊16384/20⌋=819 维护拍（D-9 换算）；
      事件换算 t_maint=⌈t_R1/20⌉ 后应用同一规则。
  (4) 测量窗=最大连续稳态段集合（并集；洞不拼接不插值）。
  (5) 测量序列=e 轴取最长连续稳态段；k* 轴取有效区间内子链（双窗档位表）。
  (6) 判据：K1-e=k_battery.k1_regime_ok；K1-k*=α̂∈[0.85,1.15]∧双窗漂移≤0.05；
      K2=dwell_csn(theta_q=None) 可辨识侧 rsqn>2（C2b 稀疏自适应口径，M1 修复③'）；
      K2 默认分位口径（k2_dwell theta_q=0.9）并列报告。
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from ahi_theory.k_battery import k1_regime_ok, dfa_summary, fss_three_windows, dwell_csn, k2_dwell, \
    dfa_curve, _band_slope

__all__ = ["T_REC_SLOW_R1拍", "R1拍每维护", "维护轴T_REC_SLOW", "稳态段", "测量窗",
           "k1_引擎判定", "k1_引擎判定_k星", "k2_引擎判定", "引擎测量汇总",
           "T_REC_FAST_R1拍", "三窗最大窗", "段不足下限", "测量窗_v2", "引擎测量汇总_v2",
           "IM满载", "IM饱和段", "测量窗_v21", "引擎测量汇总_v21",
           "节律CV上限", "节律前后半差上限", "睡眠周期序列", "节律平稳守卫", "主周期估计",
           "f周期查验", "测量窗_v3", "引擎测量汇总_v3"]

#: K3 慢通道恢复常量 T_rec^slow（内核步=R1 拍；95_架构定稿 L2 参数表行 171，一字照抄）
T_REC_SLOW_R1拍 = 16384
#: 1 维护拍 = 20 R1 拍（v2_engine.R1拍每维护 同源结构常量；本模块按值预注册，不反向 import 引擎）
R1拍每维护 = 20
#: 维护拍轴的 T_rec^slow：⌊16384/20⌋=819 维护拍（#77 (3)：同一物理时长 D-9 换算）
维护轴T_REC_SLOW = T_REC_SLOW_R1拍 // R1拍每维护

#: k*(t) 轴双窗档位表（#77 (5)：≥32K⇒{16K,32K}/≥16K⇒{8K,16K}/≥8K⇒{4K,8K}/≥4K⇒{2K,4K}/
#: ≥2K⇒{1K,2K}/<2K⇒不可辨识——同 #76 缩窗档位表）
_K星窗档位 = ((32768, (16384, 32768)), (16384, (8192, 16384)), (8192, (4096, 8192)),
              (4096, (2048, 4096)), (2048, (1024, 2048)))


def _换算维护拍(t_r1: int) -> int:
    """事件时间戳 R1 拍→维护拍轴：t_maint=⌈t_R1/20⌉（#77 (3)）。"""
    return -((-int(t_r1)) // R1拍每维护)


def 稳态段(n: int, 禁区: Sequence[tuple]) -> list:
    """[0,n) 上剔除全部禁区（半开区间列表）后的最大连续有效段列表（升序、两两不交）。

    #77 (4)：测量窗=最大连续稳态段集合（并集）；洞（禁区）不拼接不插值。"""
    排除 = sorted((max(0, int(a)), min(n, int(b))) for a, b in 禁区 if int(a) < n and int(b) > 0)
    段集 = []
    游标 = 0
    for a, b in 排除:
        if a > 游标:
            段集.append((游标, a))
        游标 = max(游标, b)
    if 游标 < n:
        段集.append((游标, n))
    return 段集


def 测量窗(n拍: int, t_run0: Optional[int], t_S越阈列表: Sequence[int],
           事件R1拍列表: Sequence[int], 轴: str = "R1") -> dict:
    """#77 冻结规则：由事件时间戳计算 [0, n拍) 上的稳态测量窗。

    轴="R1"：n拍=R1 拍总数，事件时间戳=R1 拍单位，缓冲=T_REC_SLOW_R1拍=16384。
    轴="维护"：n拍=维护拍总数，事件时间戳（R1 拍单位）换算 t_maint=⌈t/20⌉，
    缓冲=维护轴T_REC_SLOW=819 维护拍（⌊16384/20⌋，D-9 换算）。

    规则只引用事件时间戳（机制量）；本函数不接收任何 K1/K2 判定值。"""
    if 轴 == "R1":
        缓冲 = T_REC_SLOW_R1拍
        事件轴 = [int(t) for t in 事件R1拍列表]
        发育界 = (min(int(t) for t in t_S越阈列表) if len(t_S越阈列表) else None)
    elif 轴 == "维护":
        缓冲 = 维护轴T_REC_SLOW
        事件轴 = [_换算维护拍(t) for t in 事件R1拍列表]
        发育界 = (_换算维护拍(min(int(t) for t in t_S越阈列表)) if len(t_S越阈列表) else None)
    else:
        raise ValueError("轴 须为 'R1' 或 '维护'")
    下界 = 0
    界源 = []
    if t_run0 is not None:
        r0 = int(t_run0) if 轴 == "R1" else _换算维护拍(int(t_run0))
        下界 = max(下界, r0)            # run0 校准期结束后
        界源.append(("t_run0", r0))
    if 发育界 is not None:
        下界 = max(下界, 发育界)        # 发育期（首次 S 越阈前）整段剔除
        界源.append(("首次S越阈", 发育界))
    禁区 = [(0, 下界)] if 下界 > 0 else []
    事件禁区 = []
    for t_e in sorted(set(事件轴)):
        a, b = t_e - 缓冲, t_e + 缓冲    # 开区间 (t_e−缓冲, t_e+缓冲) 外方有效（|t−t_e|>缓冲）
        事件禁区.append((a + 1, b))
        if a + 1 < b:
            禁区.append((max(0, a + 1), b))
    段集 = 稳态段(int(n拍), 禁区)
    最长 = max(段集, key=lambda p: p[1] - p[0]) if 段集 else None
    return dict(轴=轴, n拍=int(n拍), 缓冲=int(缓冲),
                下界=int(下界), 下界来源=界源,
                事件数=len(set(事件轴)), 段集=段集, 段数=len(段集),
                有效拍数=int(sum(b - a for a, b in 段集)),
                最长段=(None if 最长 is None else dict(起=int(最长[0]), 止=int(最长[1]),
                                                       长=int(最长[1] - 最长[0]))))


def k1_引擎判定(e序列: np.ndarray, 窗: dict, ctx: str = "wake") -> dict:
    """K1-e 判定（#77 (6)，判据一字照抄）：e 轴取最长连续稳态段，
    k_battery.k1_regime_ok 一字照抄（α̂∈[0.85,1.15] ∧ 三窗漂移≤0.05 ∧ 洞守卫）；
    <64K ⇒ k_battery.fss_three_windows 按实测缩窗诚实标注（同 #76 口径）。
    ctx 仅作 k_battery.k1_regime_ok 的标注字段透传（#81 V2-1 offline 栏：ctx="offline"；
    该参数在 k_battery 中只进返回 dict 的 ctx 键，判据逻辑逐位同 wake——非判据变更）。"""
    x = np.asarray(e序列, dtype=np.float64)
    out = dict(判据="K1-e", 窗=窗)
    if 窗["最长段"] is None or 窗["最长段"]["长"] <= 0:
        out.update(判定="不可辨识", 理由="稳态测量窗为空（事件缓冲剔除后无有效段）")
        return out
    起止 = 窗["最长段"]
    段 = x[起止["起"]:起止["止"]]
    段 = 段[np.isfinite(段)]
    out["段长"] = int(段.size)
    if 段.size < 2048:
        out.update(判定="不可辨识", 理由="最长稳态段 <2048 拍（仪器最小窗守卫）")
        return out
    if 段.size >= 65536:
        r = k1_regime_ok(段, ctx=ctx)
        out.update(判定=("PASS" if r["passed"] else "FAIL"), 窗规格="全规格三窗 {16K,32K,64K}",
                   alpha_band=float(r["alpha_band"]), 三窗漂移=float(r["three_window_drift"]),
                   fss_alphas=r["fss_alphas"], fss_windows=r["fss_windows"],
                   洞守卫=r["hole_guard"], 去趋势双口径=r["dual_report"], status=r["status"])
    else:
        fss = fss_three_windows(段)
        a = dfa_summary(段)["alpha_band"]
        drift = fss["drift"]
        ident = bool(np.isfinite(a) and np.isfinite(drift))
        过 = bool(ident and 0.85 <= a <= 1.15 and drift <= 0.05)
        out.update(判定=("PASS" if 过 else ("不可辨识" if not ident else "FAIL")),
                   窗规格="缩窗（<64K，诚实标注）", alpha_band=float(a),
                   三窗漂移=float(drift), fss_alphas=fss["alphas"], fss_windows=fss["windows"])
    return out


def k1_引擎判定_k星(k星链: np.ndarray, 窗: dict) -> dict:
    """K1-k* 判定（#77 (6)）：维护拍轴有效区间内的 k* 子链，双窗 FSS（档位表）+
    α̂∈[0.85,1.15] ∧ 双窗漂移≤0.05。<2K 点 ⇒ 不可辨识（档位表守卫）。"""
    k = np.asarray(k星链, dtype=np.float64)
    out = dict(判据="K1-k*", 窗=窗)
    if 窗["段集"]:
        子链 = np.concatenate([k[a:b] for a, b in 窗["段集"]])   # 同轴段拼接=同拍轴并集读数
    else:
        子链 = np.asarray([], dtype=np.float64)
    out["子链长"] = int(子链.size)
    窗档 = None
    for 阈, 窗对 in _K星窗档位:
        if 子链.size >= 阈:
            窗档 = 窗对
            break
    out["双窗"] = 窗档
    if 窗档 is None:
        out.update(判定="不可辨识", 理由="有效 k* 子链 <2048 点（档位表守卫，同 #76）")
        return out
    fss = fss_three_windows(子链, windows=窗档)
    a = dfa_summary(子链)["alpha_band"]
    drift = fss["drift"]
    ident = bool(np.isfinite(a) and np.isfinite(drift))
    过 = bool(ident and 0.85 <= a <= 1.15 and drift <= 0.05)
    out.update(判定=("PASS" if 过 else ("不可辨识" if not ident else "FAIL")),
               alpha_band=float(a), fss_alphas=fss["alphas"], 三窗漂移=float(drift))
    return out


def k2_引擎判定(k星链: np.ndarray, 窗: dict) -> dict:
    """K2 判定（#77 (6)）：C2b 稀疏自适应分态口径（k_battery.dwell_csn theta_q=None，
    M1 修复③'，同一 CSN 核心 R·√n>2 判据式）——可辨识侧 rsqn>2 ⇒ PASS；
    K2 默认分位口径（k2_dwell theta_q=0.9）并列报告（预期对离散占据链不可辨识，如实登记）。"""
    k = np.asarray(k星链, dtype=np.float64)
    out = dict(判据="K2", 窗=窗)
    if 窗["段集"]:
        子链 = np.concatenate([k[a:b] for a, b in 窗["段集"]])
    else:
        子链 = np.asarray([], dtype=np.float64)
    out["子链长"] = int(子链.size)
    if 子链.size < 60:
        out.update(判定="不可辨识", 理由="有效 k* 子链过短（CSN 最小样本守卫 30×双侧）")
        return out
    rc = dwell_csn(子链, theta_q=None)
    侧读 = {}
    for 侧 in ("up", "dn"):
        f = rc[侧]
        侧读[侧] = dict(status=f["status"], n=int(f["n"]),
                        R=(float(f["R"]) if np.isfinite(f.get("R", float("nan"))) else None),
                        rsqn=(float(f["rsqn"]) if np.isfinite(f.get("rsqn", float("nan"))) else None),
                        alpha=(float(f["alpha"]) if np.isfinite(f.get("alpha", float("nan"))) else None),
                        xmin=f.get("xmin"))
    可辨 = [侧 for 侧 in ("up", "dn") if rc[侧]["status"] == "ok"
            and np.isfinite(rc[侧].get("rsqn", float("nan")))]
    过 = bool(any(rc[侧]["rsqn"] > 2.0 for 侧 in 可辨))
    out.update(判定=("PASS" if (可辨 and 过) else ("不可辨识" if not 可辨 else "FAIL")),
               口径="C2b 稀疏自适应分态（dwell_csn theta_q=None，M1 修复③'）",
               阈="R·√n > 2（csn_gate 一字照抄）", theta=float(rc["theta"]),
               medw_eff=int(rc["medw_eff"]), 侧=侧读, 可辨识侧=可辨)
    # K2 默认分位口径并列报告（#77 (6)）
    r_def = k2_dwell(子链, do_sham=True)
    out["默认口径并列"] = dict(status=r_def["status"], theta_q=r_def["theta_q"], n=r_def["n"],
                               R=(float(r_def["R"]) if np.isfinite(r_def["R"]) else None),
                               rsqn=(float(r_def["rsqn"]) if np.isfinite(r_def["rsqn"]) else None),
                               sham翻负=r_def["sham"]["flipped_negative"])
    计 = {}
    for v_ in 子链.tolist():
        计[int(v_)] = 计.get(int(v_), 0) + 1
    out["主盆占比"] = (max(计.values()) / max(1, 子链.size)) if 子链.size else None
    return out


def 引擎测量汇总(e序列: np.ndarray, k星链: Sequence[int], t_run0: Optional[int],
                 t_S越阈列表: Sequence[int], 事件R1拍列表: Sequence[int]) -> dict:
    """逐 seed 引擎级测量入口：输入=V2 引擎运行导出（e 序列/k*(t) 序列/事件时间戳列表），
    输出=剔除后的窗集合 + 逐窗 K1（e 轴+ k* 轴）与 K2（C2b dwell_csn）判定 + 汇总 verdict。
    规则实现严格按 #77 预登记文本（事件时间戳进规则；判定值不进规则）。"""
    e = np.asarray(e序列, dtype=np.float64)
    k = np.asarray(k星链, dtype=np.int64)
    窗e = 测量窗(int(e.size), t_run0, t_S越阈列表, 事件R1拍列表, 轴="R1")
    窗k = 测量窗(int(k.size), t_run0, t_S越阈列表, 事件R1拍列表, 轴="维护")
    k1e = k1_引擎判定(e, 窗e)
    k1k = k1_引擎判定_k星(k, 窗k)
    k2 = k2_引擎判定(k, 窗k)
    判定列表 = [k1e.get("判定"), k1k.get("判定"), k2.get("判定")]
    汇总 = dict(
        K1_e=k1e.get("判定"), K1_k星=k1k.get("判定"), K2=k2.get("判定"),
        判定明细=dict(K1_e=dict(alpha_band=k1e.get("alpha_band"), 漂移=k1e.get("三窗漂移"),
                                窗规格=k1e.get("窗规格"), 段长=k1e.get("段长"),
                                理由=k1e.get("理由")),
                      K1_k星=dict(alpha_band=k1k.get("alpha_band"), 漂移=k1k.get("三窗漂移"),
                                  双窗=k1k.get("双窗"), 子链长=k1k.get("子链长"),
                                  理由=k1k.get("理由")),
                      K2=dict(可辨识侧=k2.get("可辨识侧"),
                              侧={s: dict(n=v["n"], rsqn=v["rsqn"], alpha=v["alpha"])
                                  for s, v in (k2.get("侧") or {}).items()},
                              默认口径=k2.get("默认口径并列"), 主盆占比=k2.get("主盆占比"),
                              理由=k2.get("理由"))),
        窗=dict(R1=窗e, 维护=窗k),
        规则身份="#77 冻结规则（事件时间戳=机制量；判定值不进规则）")
    return dict(判定=判定列表, K1_e=k1e, K1_k星=k1k, K2=k2,
                窗=dict(R1=窗e, 维护=窗k), 汇总=汇总)


# ════════════════════════════════════════════════════════════════════════
# 规则 v2（#81 预登记，E5 会签：体制条件化分段测量；判据式/阈值一字不改，只改测量窗规则实现）
# ════════════════════════════════════════════════════════════════════════

#: K3 快通道恢复常量 T_rec^fast（k_battery DUAL_AXIS_TABLE "T_rec^fast 2000 步"/g_layer.T_rec快
#: 同源；影子装配步轴=通电 R1 拍 ⇒ 1 步=1 R1 拍；#81 V2-3）
T_REC_FAST_R1拍 = 2000
#: 三窗最大窗（k_battery.PRE["fss_windows"]={16384,32768,65536} 最大窗；#81 V2-5 机制语义）
三窗最大窗 = 65536
#: 最小可测段下限（#81 V2-5，E5 修订 b）：2×T_rec^fast+64K = 69,536 R1 拍。
#: 【文本差异如实登记】任务文本括号注「=104K 拍」与公式字面不一致；按公式字面执行
#: （机制语义=段须容纳首尾小缓冲+三窗最大窗；104K=106,496>主臂总拍 106,000 将令任何 run
#: 全段不足，与「显式降级而非全灭」语义不符；按 104K 重判只会更保守、不产生新 PASS——见 #81）。
段不足下限 = 2 * T_REC_FAST_R1拍 + 三窗最大窗


def _连续真段(掩码: np.ndarray) -> list:
    """布尔掩码的极大连续 True 段列表（半开区间，升序）。"""
    m = np.asarray(掩码, dtype=bool)
    if m.size == 0:
        return []
    d = np.diff(m.astype(np.int8))
    起 = [int(i) + 1 for i in np.flatnonzero(d == 1)]
    止 = [int(i) + 1 for i in np.flatnonzero(d == -1)]
    if m[0]:
        起 = [0] + 起
    if m[-1]:
        止 = 止 + [int(m.size)]
    return [(a, b) for a, b in zip(起, 止)]


def _段内剔除(段: tuple, 禁区: Sequence[tuple]) -> list:
    """段（半开区间）内剔除禁区列表后的极大连续子段列表（洞不拼接不插值，#77 (4) 同构）。"""
    a0, b0 = int(段[0]), int(段[1])
    排除 = sorted((max(a0, int(x)), min(b0, int(y))) for x, y in 禁区
                  if int(x) < b0 and int(y) > a0)
    子段 = []
    游标 = a0
    for x, y in 排除:
        if x > 游标:
            子段.append((游标, x))
        游标 = max(游标, y)
    if 游标 < b0:
        子段.append((游标, b0))
    return 子段


def _事件禁区(t_e: int, 缓冲: int) -> tuple:
    """事件 t_e 的禁区（半开区间）：剔除 [t_e−缓冲, t_e+缓冲] ⇒ 有效条件 |t−t_e|>缓冲
    （#77 (2) 文本字面；v1 实现保留端点拍属历史对照口径，v2 按文本字面）。"""
    return (int(t_e) - int(缓冲), int(t_e) + int(缓冲) + 1)


def 测量窗_v2(n拍: int, t_run0: Optional[int], t_S越阈列表: Sequence[int],
              kill事件R1拍列表: Sequence[int], 离线相区间列表: Sequence[tuple],
              c_t序列: Sequence, 栏: str = "wake") -> dict:
    """#81 规则 v2：由 c_t 序列（机制量）+事件时间戳构造 R1 轴指定栏（wake/offline）的分段测量窗。

    输入（全部引擎机制量，由调用方传入；本模块不 import v2_engine/g_layer）：
      t_run0/t_S越阈列表/kill事件R1拍列表 —— #77 (1) 同口径事件时间戳（R1 拍）；
      离线相区间列表 —— [(起,止)] 维护拍括号记法区间（#77 (1)，R1 拍）；
      c_t序列 —— 逐 R1 拍的 g_layer 上下文（'wake'/'offline' 字符串或 0/1 码），
                 由运行器逐拍读 g_layer 实例只读属性 G.c_t 采样（途径登记于 #81）。
    规则步骤（#81 V2-1…V2-5）：分栏掩码 → 发育期整段剔除（V2-4）→ 原始连续段
    → 每段：段不足判定（V2-5）→ 段内禁区（kill±16384〔V2-2〕+S 越阈±2000〔V2-3 修订 a〕
    +离线相边界拍〔V2-3〕+段首尾 2000 小缓冲〔V2-3〕）→ 可用子段集。
    规则只引用机制量；不接收任何 K1/K2 判定值。"""
    if 栏 not in ("wake", "offline"):
        raise ValueError("栏 须为 'wake' 或 'offline'")
    n = int(n拍)
    carr = np.asarray(c_t序列)
    if carr.size != n:
        raise ValueError("c_t序列 长度 %d ≠ n拍 %d" % (carr.size, n))
    if carr.dtype.kind in ("U", "S", "O"):
        码 = np.array([1 if str(v) == "offline" else 0 for v in carr], dtype=np.int8)
    else:
        码 = (np.asarray(carr, dtype=np.int8) != 0).astype(np.int8)
    # (V2-4) 发育期下界（照抄 #77：run0 校准期 + 首次 S 越阈前整段剔除）
    下界 = 0
    界源 = []
    if t_run0 is not None:
        下界 = max(下界, int(t_run0))
        界源.append(("t_run0", int(t_run0)))
    if len(t_S越阈列表):
        发育界 = min(int(t) for t in t_S越阈列表)
        下界 = max(下界, 发育界)
        界源.append(("首次S越阈", 发育界))
    # (V2-1) 分栏掩码（c_t 机制量）
    掩码 = (码 == (1 if 栏 == "offline" else 0))
    if 下界 > 0:
        掩码[:min(下界, n)] = False
    原始段 = _连续真段(掩码)
    S事件 = sorted(set(int(t) for t in t_S越阈列表))
    kill事件 = sorted(set(int(t) for t in kill事件R1拍列表))
    离线区间 = [(max(0, int(a)), min(n, int(b))) for a, b in 离线相区间列表
                if int(a) < n and int(b) > 0]
    段表 = []
    全部可用子段 = []
    for (a, b) in 原始段:
        长 = int(b - a)
        不足 = bool(长 < 段不足下限)                       # (V2-5) 最小可测段下限（禁区剔除前）
        禁区 = []
        # (V2-3) 段首尾 T_rec^fast 小缓冲（切换瞬态；覆盖边界拍与发育界邻域）
        禁区.append((a, min(b, a + T_REC_FAST_R1拍)))
        禁区.append((max(a, b - T_REC_FAST_R1拍), b))
        # (V2-3/E5 修订 a) 段内 S 越阈事件 ±T_rec^fast（不保 16384）
        for t in S事件:
            if a <= t < b:
                禁区.append(_事件禁区(t, T_REC_FAST_R1拍))
        # (V2-2) kill 事件 ±T_rec^slow（恢复语义，一字照抄 #77）
        for t in kill事件:
            禁区.append(_事件禁区(t, T_REC_SLOW_R1拍))
        # (V2-3) 离线相边界拍剔除（维护拍括号区间）
        for (s, e) in 离线区间:
            禁区.append((s, e))
        子段 = _段内剔除((a, b), 禁区)
        最长子段 = (max(子段, key=lambda p: p[1] - p[0]) if 子段 else None)
        段表.append(dict(起=int(a), 止=int(b), 长=长, 段不足=不足,
                         可用子段=[] if 不足 else [(int(x), int(y)) for x, y in 子段],
                         可用拍数=(0 if 不足 else int(sum(y - x for x, y in 子段))),
                         最长可用子段=(None if (不足 or 最长子段 is None) else
                                       dict(起=int(最长子段[0]), 止=int(最长子段[1]),
                                            长=int(最长子段[1] - 最长子段[0])))))
        if not 不足:
            全部可用子段.extend((int(x), int(y)) for x, y in 子段)
    长度表 = [p["长"] for p in 段表]
    分布 = dict(段数=len(段表),
                中位段长=(float(np.median(长度表)) if 长度表 else 0.0),
                最长段=(int(max(长度表)) if 长度表 else 0),
                段不足数=int(sum(1 for p in 段表 if p["段不足"])),
                可测段数=int(sum(1 for p in 段表 if not p["段不足"])),
                总可用拍数=int(sum(p["可用拍数"] for p in 段表)))
    return dict(规则身份="#81 规则 v2（体制条件化分段测量；E5 会签）", 栏=栏, 轴="R1", n拍=n,
                缓冲=dict(kill=T_REC_SLOW_R1拍, S越阈段内=T_REC_FAST_R1拍,
                          段首尾=T_REC_FAST_R1拍, 边界拍="维护拍括号区间剔除"),
                下界=int(下界), 下界来源=界源, 段不足下限=int(段不足下限),
                原始段表=段表, 可用段集=全部可用子段, 段长分布=分布)


def _维护轴窗_v2(可用段集: Sequence[tuple], k链长: int) -> dict:
    """#81 V2-7：wake 可用段集 R1 轴→维护拍轴（t_maint=⌈t/20⌉，#77 (3) D-9 换算照抄），
    clip 到 [0, k链长) 并合并相邻，得 k* 子链段集（同段集口径）。"""
    段集 = []
    for a, b in 可用段集:
        ma, mb = max(0, _换算维护拍(a)), min(int(k链长), _换算维护拍(b))
        if ma < mb:
            段集.append((ma, mb))
    段集.sort()
    合并 = []
    for s in 段集:
        if 合并 and s[0] <= 合并[-1][1]:
            合并[-1] = (合并[-1][0], max(合并[-1][1], s[1]))
        else:
            合并.append(list(s))
    最长 = (max(合并, key=lambda p: p[1] - p[0]) if 合并 else None)
    return dict(轴="维护", n拍=int(k链长), 段集=[(int(a), int(b)) for a, b in 合并],
                段数=len(合并), 有效拍数=int(sum(b - a for a, b in 合并)),
                最长段=(None if 最长 is None else dict(起=int(最长[0]), 止=int(最长[1]),
                                                       长=int(最长[1] - 最长[0]))))


def _k1_逐段判定(e序列: np.ndarray, 窗: dict, ctx: str) -> list:
    """#81 V2-1/V2-8：指定栏逐段判定（每段剔除后最长可用子段；k_battery.k1_regime_ok
    判据一字照抄、ctx 作标注透传；段不足段显式记「段不足」禁测残段〔V2-5〕）。"""
    x = np.asarray(e序列, dtype=np.float64)
    out = []
    for p in 窗["原始段表"]:
        rec = dict(起=p["起"], 止=p["止"], 长=p["长"])
        if p["段不足"]:
            rec.update(判定="段不足", 子段长=0,
                       理由="段长 <%d 拍（#81 V2-5 最小可测段下限），降级不测、禁测残段" % 段不足下限)
            out.append(rec)
            continue
        最长 = p["最长可用子段"]
        if 最长 is None or 最长["长"] <= 0:
            rec.update(判定="不可辨识", 子段长=0, 理由="禁区剔除后无可用子段")
            out.append(rec)
            continue
        纯 = x[最长["起"]:最长["止"]]
        纯 = 纯[np.isfinite(纯)]
        rec["子段长"] = int(纯.size)
        if 纯.size < 2048:
            rec.update(判定="不可辨识", 理由="最长可用子段 <2048 拍（仪器最小窗守卫）")
            out.append(rec)
            continue
        r = k1_引擎判定(x, dict(轴="R1", 段集=[(最长["起"], 最长["止"])], 最长段=最长), ctx=ctx)
        rec.update(判定=r.get("判定"), alpha_band=r.get("alpha_band"),
                   漂移=r.get("三窗漂移"), 窗规格=r.get("窗规格"), 理由=r.get("理由"))
        out.append(rec)
    return out


def 引擎测量汇总_v2(e序列, k星链, t_run0: Optional[int], t_S越阈列表: Sequence[int],
                    kill事件R1拍列表: Sequence[int], 离线相区间列表: Sequence[tuple],
                    c_t序列: Sequence) -> dict:
    """#81 规则 v2 逐 seed 引擎级测量入口：输入=V2 引擎运行导出（e 序列/k*(t) 序列/事件时间戳/
    离线相区间/c_t 序列〔调用方逐拍读 g_layer 只读属性 G.c_t 采样，途径登记于 #81〕），
    输出=wake/offline 两栏分段测量窗+逐段判定+两栏汇总 verdict+段长分布表。
    wake 栏主判据=最长可用子段的 K1-e 判定（V2-8，延续 #77「取最长连续稳态段」口径）；
    K1-k*/K2 在 wake 段集应用同段集（V2-7）；offline 栏不可辨识⇒降级条款（V2-6）。
    规则实现严格按 #81 冻结文本（机制量进规则；判定值不进规则）。"""
    e = np.asarray(e序列, dtype=np.float64)
    k = np.asarray(k星链, dtype=np.int64)
    窗w = 测量窗_v2(int(e.size), t_run0, t_S越阈列表, kill事件R1拍列表,
                    离线相区间列表, c_t序列, 栏="wake")
    窗o = 测量窗_v2(int(e.size), t_run0, t_S越阈列表, kill事件R1拍列表,
                    离线相区间列表, c_t序列, 栏="offline")
    # —— wake 栏：逐段判定（V2-8）+ 主判据=最长可用子段（V2-8，延续 #77 口径）——
    wake逐段 = _k1_逐段判定(e, 窗w, ctx="wake")
    最长w = (max(窗w["可用段集"], key=lambda p: p[1] - p[0]) if 窗w["可用段集"] else None)
    窗w主 = dict(轴="R1", 段集=窗w["可用段集"],
                 最长段=(None if 最长w is None else dict(起=int(最长w[0]), 止=int(最长w[1]),
                                                         长=int(最长w[1] - 最长w[0]))))
    k1e_w = k1_引擎判定(e, 窗w主, ctx="wake")
    # —— K1-k*/K2：wake 可用段集同段集（V2-7）——
    窗k = _维护轴窗_v2(窗w["可用段集"], int(k.size))
    k1k = k1_引擎判定_k星(k, 窗k)
    k2 = k2_引擎判定(k, 窗k)
    # —— offline 栏：逐段判定（V2-1）+ 主判定=最长可用子段同构口径 + 降级条款（V2-6）——
    offline逐段 = _k1_逐段判定(e, 窗o, ctx="offline")
    可测o = [p for p in 窗o["原始段表"] if not p["段不足"] and p["最长可用子段"] is not None]
    最长o = (max(可测o, key=lambda p: p["最长可用子段"]["长"]) if 可测o else None)
    if 最长o is not None:
        m = 最长o["最长可用子段"]
        r_o = k1_引擎判定(e, dict(轴="R1", 段集=[(m["起"], m["止"])], 最长段=m), ctx="offline")
        k1o主 = r_o.get("判定")
    else:
        k1o主 = "不可辨识"
    降级触发 = bool(k1o主 == "不可辨识")
    wake过 = bool(k1e_w.get("判定") == "PASS")
    降级 = dict(触发=降级触发,
                条款="offline 不可辨识 ⇒ 降级判据 = wake regime_ok ∧ offline 无 SCC 崩解"
                     "（g_layer 起飞门降级条款预注册原文照抄；#81 V2-6）",
                wake_regime_ok=wake过,
                scc_assessed=False, scc崩解证据=None,
                scc语义="本测量无补偿流邻接注入（G6/E7 夹具同律）⇒ SCC 未评估=无崩解证据"
                        "（与 g_layer 起飞门 scc_ok=(_scc_closed is not False) 同语义）",
                降级判定=(None if not 降级触发 else ("成立" if wake过 else "不成立")))
    汇总 = dict(
        K1_e_wake=k1e_w.get("判定"), K1_k星=k1k.get("判定"), K2=k2.get("判定"),
        K1_e_offline=k1o主, 降级条款=降级,
        段长分布=dict(wake=窗w["段长分布"], offline=窗o["段长分布"]),
        规则身份="#81 规则 v2（体制条件化分段测量；E5 会签；判定值不进规则）")
    return dict(窗=dict(wake=窗w, offline=窗o, 维护=窗k),
                K1_e_wake=k1e_w, K1_k星=k1k, K2=k2,
                K1_offline=dict(主判定=k1o主, 逐段=offline逐段, 降级条款=降级),
                wake逐段=wake逐段, 汇总=汇总)


# ════════════════════════════════════════════════════════════════════════
# 规则 v2.1（#84 预登记，E3 会签：K1 规则 v2 澄清——离线相括号主分段
#           + IM_SATURATED 污染标记；判据式/阈值一字不改，只改测量窗规则实现）
# ════════════════════════════════════════════════════════════════════════

#: IM 满载阈（hdc_core.井容量 同源值 102=0.1·D；按值预注册不反向 import——同 R1拍每维护 先例；
#: G0b 冷却/合并机制下 |IM| 稳态=102 且井向量漂移 ⇒ 满载段=井集合非平稳段）
IM满载 = 102


def IM饱和段(IM维护序列: Sequence[int], 满载: int = IM满载,
             每维护: int = R1拍每维护) -> list:
    """IM_SATURATED 前向标记（#84 E3 裁决④）：逐维护拍 |IM| 序列 → 满载连续段的 R1 轴区间列表。

    输入 IM维护序列[i] = 维护拍 i+1（1 基）末（W-HOME 后）的 |IM| 读数（hdc 导出，由运行器记录；
    该状态承载至下一维护拍）⇔ R1 块 [每维护·i, 每维护·(i+1))。|IM|≥满载 的连续维护拍块合并为
    半开区间列表（升序、两两不交）。只前向标记不回填改史：序列原样落盘，剔除只作用于测量窗构造。"""
    a = np.asarray(IM维护序列, dtype=np.int64)
    块 = [(int(i) * int(每维护), int(i) * int(每维护) + int(每维护))
          for i in np.flatnonzero(a >= int(满载))]
    合并: list = []
    for s, e in 块:
        if 合并 and s <= 合并[-1][1]:
            合并[-1] = (合并[-1][0], max(合并[-1][1], e))
        else:
            合并.append((s, e))
    return [(int(s), int(e)) for s, e in 合并]


def 测量窗_v21(n拍: int, t_run0: Optional[int], t_S越阈列表: Sequence[int],
              kill事件R1拍列表: Sequence[int], 离线相区间列表: Sequence[tuple],
              IM饱和区间列表: Sequence[tuple], c_t序列: Sequence, 栏: str = "wake") -> dict:
    """#84 规则 v2.1：离线相括号主分段 + IM_SATURATED 分段级剔除（#81 其余缓冲规则沿用一字不改）。

    与 #81 `测量窗_v2` 的差异（逐条）：
      (i) wake 栏原始段=〔发育期下界, n) 减 离线相括号区间〔闭括号 [起,止] → 半开 [起,止+1)，
          运行器维护拍括号记法，精度 ≤20 拍〕减 IM 饱和段 的补集——非 c_t 掩码
          （#81 V2-1 假设 c_t 随体制翻转；seed0/1 实证全程=0 漏切——#84(b)(i)）；
      (ii) offline 栏原始段=离线相括号区间本体（减 IM 饱和段与发育期下界）；
      (iii) c_t 分栏保留：c_t 序列照常校验长度并统计离线指称（对照旁证量并列报告；
            分段不再依赖 c_t）；
      (iv) 段内禁区沿用 #81：段首尾 T_rec^fast=2000 + 段内 S 越阈 ±2000（E5 修订 a）
           + kill ±T_rec^slow=16384（V2-2）；段不足下限 2×2000+64K=69,536（V2-5，禁区剔除前）。
    规则只引用机制量（事件时间戳/离线相括号/|IM| 序列/c_t 旁证）；不接收任何 K1/K2 判定值。"""
    if 栏 not in ("wake", "offline"):
        raise ValueError("栏 须为 'wake' 或 'offline'")
    n = int(n拍)
    # (iii) c_t 分栏对照量（机制量照常校验/统计；#84：分段不再依赖 c_t）
    carr = np.asarray(c_t序列)
    if carr.size != n:
        raise ValueError("c_t序列 长度 %d ≠ n拍 %d" % (carr.size, n))
    if carr.dtype.kind in ("U", "S", "O"):
        c离线 = np.array([1 if str(v) == "offline" else 0 for v in carr], dtype=np.int8)
    else:
        c离线 = (np.asarray(carr, dtype=np.int8) != 0).astype(np.int8)
    # (V2-4 照抄) 发育期下界：run0 校准期 + 首次 S 越阈前整段剔除
    下界 = 0
    界源 = []
    if t_run0 is not None:
        下界 = max(下界, int(t_run0))
        界源.append(("t_run0", int(t_run0)))
    if len(t_S越阈列表):
        发育界 = min(int(t) for t in t_S越阈列表)
        下界 = max(下界, 发育界)
        界源.append(("首次S越阈", 发育界))
    # (i)/(ii) 分段级切割集合：离线相括号（闭→半开）∪ IM 饱和段（已为半开）——同级
    离线切 = sorted((max(0, int(a)), min(n, int(b) + 1)) for a, b in 离线相区间列表
                    if int(a) < n and int(b) >= 0)
    饱和切 = sorted((max(0, int(a)), min(n, int(b))) for a, b in IM饱和区间列表
                    if int(a) < n and int(b) > 0)
    if 栏 == "wake":
        原始段 = []
        if 下界 < n:
            原始段 = _段内剔除((下界, n), 离线切 + 饱和切)
    else:
        原始段 = []
        for s, e in 离线切:
            for p in _段内剔除((s, e), 饱和切):
                a_, b_ = max(int(p[0]), 下界), int(p[1])
                if b_ > a_:
                    原始段.append((a_, b_))
    S事件 = sorted(set(int(t) for t in t_S越阈列表))
    kill事件 = sorted(set(int(t) for t in kill事件R1拍列表))
    段表 = []
    全部可用子段 = []
    for (a, b) in 原始段:
        长 = int(b - a)
        不足 = bool(长 < 段不足下限)                       # (V2-5 照抄) 禁区剔除前判定
        禁区 = []
        if not 不足:
            # (V2-3 照抄) 段首尾 T_rec^fast 小缓冲（切换瞬态）
            禁区.append((a, min(b, a + T_REC_FAST_R1拍)))
            禁区.append((max(a, b - T_REC_FAST_R1拍), b))
            # (V2-3/E5 修订 a 照抄) 段内 S 越阈事件 ±T_rec^fast
            for t in S事件:
                if a <= t < b:
                    禁区.append(_事件禁区(t, T_REC_FAST_R1拍))
            # (V2-2 照抄) kill 事件 ±T_rec^slow（恢复语义）
            for t in kill事件:
                禁区.append(_事件禁区(t, T_REC_SLOW_R1拍))
            子段 = _段内剔除((a, b), 禁区)
        else:
            子段 = []
        最长子段 = (max(子段, key=lambda p: p[1] - p[0]) if 子段 else None)
        段表.append(dict(起=int(a), 止=int(b), 长=长, 段不足=不足,
                         可用子段=[] if 不足 else [(int(x), int(y)) for x, y in 子段],
                         可用拍数=(0 if 不足 else int(sum(y - x for x, y in 子段))),
                         最长可用子段=(None if (不足 or 最长子段 is None) else
                                       dict(起=int(最长子段[0]), 止=int(最长子段[1]),
                                            长=int(最长子段[1] - 最长子段[0])))))
        if not 不足:
            全部可用子段.extend((int(x), int(y)) for x, y in 子段)
    长度表 = [p["长"] for p in 段表]
    分布 = dict(段数=len(段表),
                中位段长=(float(np.median(长度表)) if 长度表 else 0.0),
                最长段=(int(max(长度表)) if 长度表 else 0),
                段不足数=int(sum(1 for p in 段表 if p["段不足"])),
                可测段数=int(sum(1 for p in 段表 if not p["段不足"])),
                总可用拍数=int(sum(p["可用拍数"] for p in 段表)))
    # (iv) 污染段统计（IM_SATURATED；#84(b)(ii)/(iv)）+ c_t 旁证（#84(b)(i) c_t 分栏保留）
    污染拍数 = int(sum(e - s for s, e in 饱和切))
    离线括号拍数 = int(sum(e - s for s, e in 离线切))
    c旁证 = dict(c_t离线拍数=int(c离线.sum()),
                 c_t离线拍数在离线括号内=int(sum(int(c离线[s:min(n, e)].sum())
                                                 for s, e in 离线切)),
                 离线括号拍数=离线括号拍数,
                 括号数=len(离线切),
                 语义="c_t 在当前体制未翻转（seed0/1 实证全程=0）⇒ 分段不依赖 c_t；"
                      "本栏仅旁证对照（#84(b)(i) c_t 分栏保留）")
    return dict(规则身份="#84 规则 v2.1（离线相括号主分段+IM_SATURATED 前向标记；E3 会签）",
                栏=栏, 轴="R1", n拍=n,
                缓冲=dict(kill=T_REC_SLOW_R1拍, S越阈段内=T_REC_FAST_R1拍,
                          段首尾=T_REC_FAST_R1拍, 边界拍="离线相括号+IM饱和段（分段级切割，同级）"),
                下界=int(下界), 下界来源=界源, 段不足下限=int(段不足下限),
                离线括号切分数=len(离线切), IM饱和切分数=len(饱和切),
                污染段统计=dict(段数=len(饱和切), 总拍数=污染拍数,
                                总拍占比=(float(污染拍数) / n if n else 0.0)),
                c_t分栏旁证=c旁证,
                原始段表=段表, 可用段集=全部可用子段, 段长分布=分布)


def 引擎测量汇总_v21(e序列, k星链, t_run0: Optional[int], t_S越阈列表: Sequence[int],
                     kill事件R1拍列表: Sequence[int], 离线相区间列表: Sequence[tuple],
                     IM饱和区间列表: Sequence[tuple], c_t序列: Sequence) -> dict:
    """#84 规则 v2.1 逐 seed 引擎级测量入口：输入=V2 引擎运行导出（e 序列/k*(t) 序列/事件时间戳/
    离线相区间〔运行器维护拍括号记法，精度 ≤20 拍——途径登记于 #84(b)(i)〕/IM 饱和段
    〔`IM饱和段` 由逐维护拍 |IM| 序列导出，E3 裁决④ 前向标记〕/c_t 序列〔对照旁证，#81 途径不变〕），
    输出=wake/offline 两栏分段测量窗+逐段判定+两栏汇总 verdict+段长分布+污染段统计。
    wake 栏主判据=最长可用子段的 K1-e 判定（V2-8 口径延续）；K1-k*/K2 在 wake 段集应用同段集
    （V2-7）；offline 栏不可辨识⇒降级条款（V2-6 照抄）。判据式/阈值一字不改
    （k_battery.k1_regime_ok/dwell_csn 原样复用）；规则只引用机制量；判定值不进规则。"""
    e = np.asarray(e序列, dtype=np.float64)
    k = np.asarray(k星链, dtype=np.int64)
    窗w = 测量窗_v21(int(e.size), t_run0, t_S越阈列表, kill事件R1拍列表,
                    离线相区间列表, IM饱和区间列表, c_t序列, 栏="wake")
    窗o = 测量窗_v21(int(e.size), t_run0, t_S越阈列表, kill事件R1拍列表,
                    离线相区间列表, IM饱和区间列表, c_t序列, 栏="offline")
    # —— wake 栏：逐段判定（V2-8 延续）+ 主判据=最长可用子段 ——
    wake逐段 = _k1_逐段判定(e, 窗w, ctx="wake")
    最长w = (max(窗w["可用段集"], key=lambda p: p[1] - p[0]) if 窗w["可用段集"] else None)
    窗w主 = dict(轴="R1", 段集=窗w["可用段集"],
                 最长段=(None if 最长w is None else dict(起=int(最长w[0]), 止=int(最长w[1]),
                                                         长=int(最长w[1] - 最长w[0]))))
    k1e_w = k1_引擎判定(e, 窗w主, ctx="wake")
    # —— K1-k*/K2：wake 可用段集同段集（V2-7）——
    窗k = _维护轴窗_v2(窗w["可用段集"], int(k.size))
    k1k = k1_引擎判定_k星(k, 窗k)
    k2 = k2_引擎判定(k, 窗k)
    # —— offline 栏：逐段判定 + 主判定=最长可用子段同构口径 + 降级条款（V2-6 照抄）——
    offline逐段 = _k1_逐段判定(e, 窗o, ctx="offline")
    可测o = [p for p in 窗o["原始段表"] if not p["段不足"] and p["最长可用子段"] is not None]
    最长o = (max(可测o, key=lambda p: p["最长可用子段"]["长"]) if 可测o else None)
    if 最长o is not None:
        m = 最长o["最长可用子段"]
        r_o = k1_引擎判定(e, dict(轴="R1", 段集=[(m["起"], m["止"])], 最长段=m), ctx="offline")
        k1o主 = r_o.get("判定")
    else:
        k1o主 = "不可辨识"
    降级触发 = bool(k1o主 == "不可辨识")
    wake过 = bool(k1e_w.get("判定") == "PASS")
    降级 = dict(触发=降级触发,
                条款="offline 不可辨识 ⇒ 降级判据 = wake regime_ok ∧ offline 无 SCC 崩解"
                     "（g_layer 起飞门降级条款预注册原文照抄；#81 V2-6 沿用）",
                wake_regime_ok=wake过,
                scc_assessed=False, scc崩解证据=None,
                scc语义="本测量无补偿流邻接注入（G6/E7 夹具同律）⇒ SCC 未评估=无崩解证据"
                        "（与 g_layer 起飞门 scc_ok=(_scc_closed is not False) 同语义）",
                降级判定=(None if not 降级触发 else ("成立" if wake过 else "不成立")))
    汇总 = dict(
        K1_e_wake=k1e_w.get("判定"), K1_k星=k1k.get("判定"), K2=k2.get("判定"),
        K1_e_offline=k1o主, 降级条款=降级,
        段长分布=dict(wake=窗w["段长分布"], offline=窗o["段长分布"]),
        污染段统计=dict(wake=窗w["污染段统计"], offline=窗o["污染段统计"]),
        规则身份="#84 规则 v2.1（离线相括号主分段+IM_SATURATED 前向标记；E3 会签；判定值不进规则）")
    return dict(窗=dict(wake=窗w, offline=窗o, 维护=窗k),
                K1_e_wake=k1e_w, K1_k星=k1k, K2=k2,
                K1_offline=dict(主判定=k1o主, 逐段=offline逐段, 降级条款=降级),
                wake逐段=wake逐段, 汇总=汇总)


# ════════════════════════════════════════════════════════════════════════
# 规则 v3（#86 预登记，E5 会签全文：全 record 测量窗+节律平稳守卫
#          +F(n) 周期查验+K2 伴生口径；判据式/阈值一字不改，只改测量窗规则实现；
#          v2.1/v2/v1 原样保留为历史对照开关）
# ════════════════════════════════════════════════════════════════════════
#
# #85 的 12-seed v2.1 实录归因：wake 段按离线相括号切割后 145 段/中位 700 拍、段不足下限 69,536
# ⇒ 全部段不足=结构性不可辨识。根因=分段依据把被调节体制的**内禀节律**（睡眠-觉醒周期，离线相
# ~144–145 次/106K 拍）当污染剔除——与 #76 末 64K 子窗跨 50+ 周期 PASS 的事实矛盾（反证）。
# 规则 v3（#86 冻结文本的实现对应）：
#   (i)   测量窗=全 record：剔除集=发育期下界（run0+首 S 越阈前，V2-4 照抄）∪ kill±T_rec^slow=16384
#         （V2-2 照抄）∪ IM_SATURATED 污染段（E3 裁决④ 照抄）；睡眠-觉醒节律包含在内（不按离线相
#         括号/c_t 切割）。段集=稳态段并集（洞不拼接不插值，#77(4) 同构）；段不足下限 69,536 不作
#         分段门（全 record 口径下段=剔除集补集；K1 最小窗守卫 2048+<64K 缩窗诚实标注照抄）。
#   (ii)  节律平稳守卫：睡眠周期序列=相邻离线相括号起点间隔（R1 拍），CV ≤ 0.5 ∧ 前后半 record
#         周期均值相对差 ≤ 0.2；越界 ⇒ K1 记「不可辨识—归因节律漂移」；周期样本 <2 ⇒ 守卫不可判
#         （不触发，如实标注——不借样本不足放宽 K1）。
#   (iii) F(n) 在 n≈主周期处无平台/隆起查验（防节律伪 α，附报告）：主周期=相邻 S 越阈间隔中位数；
#         DFA F(n) 网格在 n∈[0.5,2]×主周期 的局部对数斜率与隆起比如实并列——本轮不设判别阈
#         （零拟合：不造区间估计/不搜索参数；判读留 E5 后续会签），仅污染可见化。
#   (iv)  K2=全 record 主判据+剔除离线常数段伴生口径同报：伴生=全 record 段集再剔除离线相括号区间
#         （离线相内 S 冻结⇒k* 常数段污染驻留分布）；两口径判定同侧（「两指数 CI 一致」的实现口径=
#         两 rsqn>2 判定同值——区间宽度由 csn_gate R·√n 判据式自带，不另造估计）⇒通过；
#         分歧如实登记（主 PASS 伴非 PASS⇒FAIL；主 FAIL⇒FAIL；主不可辨识⇒不可辨识）。
#   (v)   wake/offline 分栏降辅助报告：v2.1 分栏口径（离线相括号主分段）的 K1-e 栏主判定并列输出
#         为辅助量，主判定不依赖；c_t 旁证量保留。

#: 节律平稳守卫阈（#86 冻结，E5 会签；定值不凑数）：睡眠周期序列 CV 上限
节律CV上限 = 0.5
#: 节律平稳守卫阈（#86 冻结）：前后半 record 周期均值相对差上限
节律前后半差上限 = 0.2


def 睡眠周期序列(离线相区间列表: Sequence[tuple]) -> np.ndarray:
    """睡眠-觉醒周期序列（#86 (ii)）：相邻离线相括号起点间隔（R1 拍，升序排序后一阶差分）。

    离线相括号=被调节体制的内禀节律事件（机制量）；间隔序列=节律本体。样本 <2 ⇒ 空数组。"""
    起点列表 = sorted(int(a) for a, _b in 离线相区间列表)
    if len(起点列表) < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(np.asarray(起点列表, dtype=np.float64))


def 节律平稳守卫(离线相区间列表: Sequence[tuple], n拍: int) -> dict:
    """#86 (ii) 节律平稳守卫：CV ≤ 0.5 ∧ 前后半 record 周期均值相对差 ≤ 0.2；越界 ⇒ K1 记
    「不可辨识—归因节律漂移」。前后半按 record 中点分（周期归入其起点所在半场）；
    分母=两半均值绝对值的均值（对称口径）。周期样本 <2 ⇒ 不可判（不触发守卫，如实标注）。"""
    周期 = 睡眠周期序列(离线相区间列表)
    n = int(n拍)
    out = dict(判据="节律平稳守卫", 阈=dict(CV上限=节律CV上限, 前后半相对差上限=节律前后半差上限),
               周期样本数=int(周期.size))
    if 周期.size < 2:
        out.update(判定="不可判", 越界=False,
                   理由="睡眠周期序列样本 <2（守卫不触发，如实标注——不借样本不足放宽 K1；#86 (ii)）")
        return out
    起点 = np.asarray(sorted(int(a) for a, _b in 离线相区间列表), dtype=np.float64)
    中点 = n / 2.0
    前半 = 周期[起点[:-1] < 中点]
    后半 = 周期[起点[:-1] >= 中点]
    均值 = float(周期.mean())
    cv = (float(周期.std(ddof=0) / 均值) if 均值 > 0 else float("inf"))
    if 前半.size and 后半.size:
        m1, m2 = float(前半.mean()), float(后半.mean())
        分母 = 0.5 * (abs(m1) + abs(m2))
        差 = (float(abs(m2 - m1) / 分母) if 分母 > 0 else float("inf"))
    else:
        m1 = m2 = float("nan")
        差 = float("inf")
    越界 = bool(cv > 节律CV上限 or 差 > 节律前后半差上限)
    越界明细 = "; ".join(filter(None, [
        ("CV=%.3f > %.1f" % (cv, 节律CV上限)) if cv > 节律CV上限 else "",
        ("前后半相对差=%.3f > %.1f" % (差, 节律前后半差上限)) if 差 > 节律前后半差上限 else ""]))
    out.update(判定=("越界" if 越界 else "平稳"), 越界=越界, CV=cv, 周期中位数=float(np.median(周期)),
               前半均值=m1, 后半均值=m2, 前后半相对差=差, 前半样本=int(前半.size), 后半样本=int(后半.size),
               理由=("" if not 越界 else
                     "节律平稳守卫越界（%s）——归因节律漂移（#86 (ii) E5 会签）；底层 α 判定原样并列（不删除）"
                     % 越界明细))
    return out


def 主周期估计(t_S越阈列表: Sequence[int]) -> dict:
    """#86 (iii)：主周期=相邻 S 越阈间隔中位数（R1 拍）；样本 <2 ⇒ nan（如实）。"""
    事件 = sorted(set(int(t) for t in t_S越阈列表))
    out = dict(判据="主周期估计", 口径="相邻 S 越阈间隔中位数（R1 拍；#86 (iii)）", 事件数=len(事件))
    if len(事件) < 2:
        out.update(主周期=float("nan"), 间隔样本=0, 判定="不可判")
        return out
    间隔 = np.diff(np.asarray(事件, dtype=np.float64))
    out.update(主周期=float(np.median(间隔)), 间隔样本=int(间隔.size),
               间隔中位=float(np.median(间隔)), 间隔均值=float(间隔.mean()),
               判定="可判")
    return out


def f周期查验(e段: np.ndarray, 主周期: float) -> dict:
    """#86 (iii) F(n) 在 n≈主周期处无平台/隆起查验（防节律伪 α；**附报告——本轮不设判别阈**）。

    量测（零拟合：只读 k_battery.dfa_curve 的 F(n) 网格值，不造估计器/不搜索参数）：
      窗 = n∈[0.5,2]×主周期 内的 F(n) 网格点（窗内点 <3 ⇒ 不可判，如实）；
      局部斜率 = 窗内 log F–log n 的 OLS 斜率（k_battery._band_slope 同源同法；斜率→0 即平台）；
      隆起比 = 窗内相邻三点 F_mid/((F_left+F_right)/2) − 1 的最大值（局部非单调凸起幅度）。
    判读留 E5 后续会签；本函数只如实并列量测值。"""
    x = np.asarray(e段, dtype=np.float64)
    out = dict(判据="F(n) 周期查验", 口径="DFA F(n) 网格在 n∈[0.5,2]×主周期 的局部斜率与隆起比"
                                            "（附报告，不设判别阈——#86 (iii)）")
    if not (np.isfinite(主周期) and 主周期 > 0) or x.size < 2048:
        out.update(判定="不可判", 理由="主周期不可判或主段 <2048 拍（仪器最小窗守卫）")
        return out
    n_grid, F = dfa_curve(x)
    m = (n_grid >= 0.5 * 主周期) & (n_grid <= 2.0 * 主周期) & np.isfinite(F)
    out["窗点数"] = int(m.sum())
    out["窗网格"] = [int(v) for v in n_grid[m].tolist()]
    if int(m.sum()) < 3:
        out.update(判定="不可判", 理由="n∈[0.5,2]×主周期 窗内 F(n) 网格点 <3（如实）")
        return out
    ng, Fg = n_grid[m], F[m]
    斜率 = _band_slope(ng, Fg, float(ng.min()), float(ng.max()))
    隆起 = 0.0
    for i in range(1, ng.size - 1):
        邻均 = 0.5 * (Fg[i - 1] + Fg[i + 1])
        if 邻均 > 0:
            隆起 = max(隆起, float(Fg[i] / 邻均 - 1.0))
    out.update(判定="量测（附报告）", 局部斜率=float(斜率), 隆起比=隆起,
               主周期=float(主周期), 窗左=float(0.5 * 主周期), 窗右=float(2.0 * 主周期))
    return out


def 测量窗_v3(n拍: int, t_run0: Optional[int], t_S越阈列表: Sequence[int],
              kill事件R1拍列表: Sequence[int], IM饱和区间列表: Sequence[tuple]) -> dict:
    """#86 规则 v3：全 record 测量窗——剔除集=发育期下界 ∪ kill±T_rec^slow ∪ IM_SATURATED 段。

    与 #84 规则 v2.1 的差异：**睡眠-觉醒节律包含在内**——离线相括号与 c_t 都不进分段
    （被调节体制的内禀节律，非污染：#76 末 64K 子窗跨 50+ 周期 PASS 即反证；v2.1 按离线相括号
    切割后 145 段全段不足=结构性不可辨识的根因，#85 实录）。
    剔除集（全部机制量）：
      下界 = max(t_run0, 首次 S 越阈)（V2-4 照抄）；
      kill 事件 ±T_rec^slow=16384（V2-2 照抄，|t−t_e|>缓冲 字面口径同 _事件禁区）；
      IM_SATURATED 满载段（E3 裁决④ 照抄，ke.IM饱和段 导出的半开 R1 区间）。
    段集=剔除集补集的极大连续段（洞不拼接不插值，#77(4) 同构）；段不足下限 69,536 不作分段门
    （全 record 口径；K1 侧最小窗守卫 2048+<64K 缩窗诚实标注照抄 k1_引擎判定）。
    规则只引用机制量；不接收任何 K1/K2 判定值。"""
    n = int(n拍)
    # 发育期下界（V2-4 照抄）
    下界 = 0
    界源 = []
    if t_run0 is not None:
        下界 = max(下界, int(t_run0))
        界源.append(("t_run0", int(t_run0)))
    if len(t_S越阈列表):
        发育界 = min(int(t) for t in t_S越阈列表)
        下界 = max(下界, 发育界)
        界源.append(("首次S越阈", 发育界))
    # 禁区：kill ±16384 + IM 饱和段（半开区间）
    禁区 = []
    kill事件 = sorted(set(int(t) for t in kill事件R1拍列表))
    for t in kill事件:
        禁区.append(_事件禁区(t, T_REC_SLOW_R1拍))
    饱和切 = sorted((max(0, int(a)), min(n, int(b))) for a, b in IM饱和区间列表
                    if int(a) < n and int(b) > 0)
    禁区 += 饱和切
    段集 = []
    if 下界 < n:
        段集 = _段内剔除((下界, n), 禁区)
    最长 = (max(段集, key=lambda p: p[1] - p[0]) if 段集 else None)
    污染拍数 = int(sum(e_ - s_ for s_, e_ in 饱和切))
    return dict(规则身份="#86 规则 v3（全 record 测量窗+节律平稳守卫；E5 会签）",
                轴="R1", n拍=n, 缓冲=dict(kill=T_REC_SLOW_R1拍,
                                          IM饱和="分段级剔除（E3 裁决④ 照抄）",
                                          节律="睡眠-觉醒节律包含在内（离线相括号不切割）"),
                下界=int(下界), 下界来源=界源,
                kill事件数=len(kill事件), IM饱和切分数=len(饱和切),
                污染段统计=dict(段数=len(饱和切), 总拍数=污染拍数,
                                总拍占比=(float(污染拍数) / n if n else 0.0)),
                段集=[(int(a), int(b)) for a, b in 段集], 段数=len(段集),
                有效拍数=int(sum(b - a for a, b in 段集)),
                最长段=(None if 最长 is None else dict(起=int(最长[0]), 止=int(最长[1]),
                                                       长=int(最长[1] - 最长[0]))))


def _v21栏主判定(e序列: np.ndarray, 窗: dict, ctx: str) -> tuple:
    """#86 (v) 辅助报告用：v2.1 分栏口径（测量窗_v21 输出）的 K1-e 栏主判定（最长可用子段）。"""
    x = np.asarray(e序列, dtype=np.float64)
    可测 = [p for p in 窗["原始段表"] if not p["段不足"] and p["最长可用子段"] is not None]
    if not 可测:
        return "不可辨识", None
    m = max(可测, key=lambda p: p["最长可用子段"]["长"])["最长可用子段"]
    r = k1_引擎判定(x, dict(轴="R1", 段集=[(m["起"], m["止"])], 最长段=m), ctx=ctx)
    return r.get("判定"), r


def 引擎测量汇总_v3(e序列, k星链, t_run0: Optional[int], t_S越阈列表: Sequence[int],
                    kill事件R1拍列表: Sequence[int], 离线相区间列表: Sequence[tuple],
                    IM饱和区间列表: Sequence[tuple], c_t序列: Sequence) -> dict:
    """#86 规则 v3 逐 seed 引擎级测量入口（输入=与 v2.1 同一机制量集合）。

    主判据（wake/offline 分栏降辅助——(v)）：
      K1-e = 全 record 最长段（k1_引擎判定 一字照抄：α̂∈[0.85,1.15]∧三窗漂移≤0.05；<64K 缩窗
             诚实标注）∧ 节律平稳守卫（(ii)：越界⇒「不可辨识—归因节律漂移」，底层 α 判定并列）；
      K1-k* = 全 record 段集维护轴换算（双窗档位表，#77(5) 照抄）；
      F(n) 周期查验 = K1-e 主段上的 (iii) 附报告量；
      K2 = 全 record 主判据 + 剔除离线常数段伴生口径（(iv)：两口径 rsqn>2 判定同值⇒PASS；
            主 PASS 伴非 PASS⇒FAIL；主 FAIL⇒FAIL；主不可辨识⇒不可辨识；一致性如实）。
    规则实现严格按 #86 冻结文本（机制量进规则；判定值不进规则）。"""
    e = np.asarray(e序列, dtype=np.float64)
    k = np.asarray(k星链, dtype=np.int64)
    窗v3 = 测量窗_v3(int(e.size), t_run0, t_S越阈列表, kill事件R1拍列表, IM饱和区间列表)
    # —— 节律平稳守卫（(ii)；越界⇒K1 记不可辨识—归因节律漂移）——
    守卫 = 节律平稳守卫(离线相区间列表, n拍=int(e.size))
    # —— K1-e 主判据：全 record 最长段（<64K 缩窗诚实标注照抄 k1_引擎判定）——
    最长 = 窗v3["最长段"]
    窗主 = dict(轴="R1", 段集=窗v3["段集"], 最长段=最长)
    k1e = k1_引擎判定(e, 窗主, ctx="wake")
    k1e判定 = k1e.get("判定")
    if 守卫["越界"]:
        底层判定 = k1e判定
        k1e = dict(k1e)
        k1e["判定"] = "不可辨识"
        k1e["守卫前判定"] = 底层判定
        k1e["理由"] = ((守卫.get("理由") or "节律平稳守卫越界——归因节律漂移（#86 (ii)）")
                       + "；底层 α 判定=%s（α=%s 漂移=%s）原样并列"
                       % (底层判定, k1e.get("alpha_band"), k1e.get("三窗漂移")))
        k1e判定 = "不可辨识"
    # —— F(n) 周期查验（(iii) 附报告；主段=全 record 最长段）——
    周期 = 主周期估计(t_S越阈列表)
    f查验 = dict(判定="不可判", 理由="全 record 最长段为空")
    if 最长 is not None:
        f查验 = f周期查验(e[最长["起"]:最长["止"]], 周期.get("主周期", float("nan")))
    # —— K1-k*：全 record 段集同段集（维护轴换算照抄 _维护轴窗_v2）——
    窗k = _维护轴窗_v2(窗v3["段集"], int(k.size))
    k1k = k1_引擎判定_k星(k, 窗k)
    # —— K2：全 record 主判据 + 剔除离线常数段伴生口径（(iv)）——
    k2 = k2_引擎判定(k, 窗k)
    n_e = int(e.size)
    离线切 = sorted((max(0, int(a)), min(n_e, int(b) + 1)) for a, b in 离线相区间列表
                    if int(a) < n_e and int(b) >= 0)
    段集伴生 = []
    for (a, b) in 窗v3["段集"]:
        段集伴生 += _段内剔除((a, b), 离线切)
    窗k伴 = _维护轴窗_v2(段集伴生, int(k.size))
    k2伴 = k2_引擎判定(k, 窗k伴)
    主判 = k2.get("判定")
    伴判 = k2伴.get("判定")
    if 主判 == "不可辨识":
        k2判定 = "不可辨识"
    elif 主判 == "FAIL":
        k2判定 = "FAIL"
    else:                                       # 主 PASS：两指数 CI 一致（判定同值）⇒通过；分歧如实⇒FAIL
        k2判定 = "PASS" if 伴判 == "PASS" else "FAIL"
    # —— wake/offline 分栏辅助报告（(v)：v2.1 分栏口径并列，主判定不依赖）——
    窗w_aux = 测量窗_v21(int(e.size), t_run0, t_S越阈列表, kill事件R1拍列表,
                         离线相区间列表, IM饱和区间列表, c_t序列, 栏="wake")
    窗o_aux = 测量窗_v21(int(e.size), t_run0, t_S越阈列表, kill事件R1拍列表,
                         离线相区间列表, IM饱和区间列表, c_t序列, 栏="offline")
    auxw判定, auxw明细 = _v21栏主判定(e, 窗w_aux, ctx="wake")
    auxo判定, auxo明细 = _v21栏主判定(e, 窗o_aux, ctx="offline")
    carr = np.asarray(c_t序列)
    if carr.dtype.kind in ("U", "S", "O"):
        c离线拍数 = int(sum(1 for v in carr if str(v) == "offline"))
    else:
        c离线拍数 = int((carr != 0).sum())
    汇总 = dict(
        K1_e=k1e判定, K1_k星=k1k.get("判定"), K2=k2判定,
        节律守卫=守卫["判定"], K2伴生=伴判, K2两口径一致=bool(主判 == 伴判),
        F周期查验=f查验.get("判定"),
        K1_e_wake辅助=auxw判定, K1_e_offline辅助=auxo判定,
        判定明细=dict(
            K1_e=dict(alpha_band=k1e.get("alpha_band"), 漂移=k1e.get("三窗漂移"),
                      窗规格=k1e.get("窗规格"), 段长=k1e.get("段长"), 理由=k1e.get("理由"),
                      守卫=dict(CV=守卫.get("CV"), 前后半相对差=守卫.get("前后半相对差"),
                                周期样本=守卫.get("周期样本数"), 判定=守卫.get("判定"))),
            K1_k星=dict(alpha_band=k1k.get("alpha_band"), 漂移=k1k.get("三窗漂移"),
                        双窗=k1k.get("双窗"), 子链长=k1k.get("子链长"), 理由=k1k.get("理由")),
            K2=dict(可辨识侧=k2.get("可辨识侧"), 伴生可辨识侧=k2伴.get("可辨识侧"),
                    侧={s_: dict(n=v_["n"], rsqn=v_["rsqn"], alpha=v_["alpha"])
                        for s_, v_ in (k2.get("侧") or {}).items()},
                    伴生侧={s_: dict(n=v_["n"], rsqn=v_["rsqn"], alpha=v_["alpha"])
                            for s_, v_ in (k2伴.get("侧") or {}).items()},
                    主盆占比=k2.get("主盆占比"), 伴生主盆占比=k2伴.get("主盆占比"),
                    理由=k2.get("理由")),
            节律守卫=dict(CV=守卫.get("CV"), 周期中位数=守卫.get("周期中位数"),
                          前半均值=守卫.get("前半均值"), 后半均值=守卫.get("后半均值"),
                          前后半相对差=守卫.get("前后半相对差"), 理由=守卫.get("理由")),
            F周期查验=dict(局部斜率=f查验.get("局部斜率"), 隆起比=f查验.get("隆起比"),
                           主周期=周期.get("主周期"), 窗点数=f查验.get("窗点数"), 理由=f查验.get("理由")),
            辅助分栏=dict(wake=dict(判定=auxw判定, 窗=dict(wake=窗w_aux["段长分布"])),
                          offline=dict(判定=auxo判定, 窗=dict(offline=窗o_aux["段长分布"])),
                          c_t_offline拍数=c离线拍数,
                          语义="v2.1 分栏口径（离线相括号主分段）降辅助报告——主判定不依赖（#86 (v)）")),
        窗=dict(全record=窗v3, 维护=窗k, 维护伴生=窗k伴),
        规则身份="#86 规则 v3（全 record 测量窗+节律平稳守卫+K2 伴生口径；E5 会签；判定值不进规则）")
    return dict(窗=dict(全record=窗v3, 维护=窗k, 维护伴生=窗k伴),
                K1_e=k1e, K1_k星=k1k, K2=k2, K2伴生=k2伴,
                节律守卫=守卫, 主周期=周期, F周期查验=f查验,
                辅助分栏=dict(wake=dict(判定=auxw判定, 明细=auxw明细),
                              offline=dict(判定=auxo判定, 明细=auxo明细)),
                汇总=汇总)
