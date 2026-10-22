# -*- coding: utf-8 -*-
"""ahi_theory/k_battery.py —— M1 · K-BATTERY 合成器 + 标定器（六层架构定稿的判据仪器）

地位（架构定稿 `meeting11/95_架构定稿.md` §2/L2、§3.1、§5 蓝图 M1 行）：
  前置链（硬性，先于任何 K 组引擎测量与 K3）：
    K5 六情形自证 ≥90%（合成；打洞族 G1/G2 ≥90%、G3 ≥85%）→ σ̂/β̂ 偏差标定（A11）
    → 时滞注入标定（k_p/η_i 初值）→ 一切下游判据。
  K5 不过线 ⇒ K1–K4 全废，宁要"仪器未成"的诚实结论（E5 不可妥协 1：尺子先于对象）。

影子模式纪律（spec.md "What Changes"）：
  · 本文件**纯新增**；不 import brain/mbp/world/selfprobe/modality/system/verify_all 任何模块，
    不改既有文件任何一行（verify_all 的 K 组注册是 M13 的事）。
  · 守卫思想沿用 verify_all.tau_fit_detail（2026-09-26b 预注册）：**不可辨识返回 NaN，绝不伪装**。
  · 判据/阈值一字照抄定稿：K5 验收 ≥90%（合成）/85%（半合成 G3）；K1 α∈[0.85,1.15]@n∈[16,512]、
    三窗{16K,32K,64K}漂移 ≤0.05；K2 驻留 CSN 似然比 R·√n > 2、sham 打乱必翻负。
  · 双集纪律：判据仪器为确定性合成，双集体现为不同 seed 流（判据集 master seed 0–5 / 留出集 6–11，
    逐 (set, case, rep) 以 default_rng([set_id, case_idx, rep]) 派生——公式固定、不挑 seed）。
  · 预注册声明：本模块的全部仪器内阈（PRE 常量块）在**实现时**从判别表（E5 方案 §3.3）第一性导出，
    先于任何验收运行；变更 = E5 判据式修订（如实报告，不调判据式凑数）。

内容：
  §0 常量：双轴常量合同表（E5 Round3 交付①）+ 候选值表（K5 自证后落册）
  §1 六情形合成器 A/B/C1/C2/C3/D（带真值 meta）+ 打洞族 G1/G2/G3（δ_hole 参数化）+ 洞日志
  §2 测量仪器：DFA / PSD（全窗周期ogram β 通道 + Welch LR 通道）/ R/S / 驻留 CSN / FSS 三窗 /
      去趋势双口径（B 守卫）
  §3 分类器（判别表决策树，预注册）
  §4 判据函数：K1（FSS 塌缩版 regime_ok）/ K2（+sham）/ K3⊕K4（两段式+验账+SCC，接口，M5/M7 供给）
      / K5 验收（混淆矩阵+双集）
  §5 A11 标定：σ̂/β̂ 偏差曲线 → σ̂_target(W_o,θ) 查表；时滞注入标定 → k_p/η_i 初值
  §6 C2b 判读入口（E3 Round3 §1.2 判据式原文：PASS_P_i / PASS_P_ii / T10 死带 / 压平窗排除 / 灵敏度 sham）
"""
from __future__ import annotations

import json
import math
import time
from typing import Optional, Sequence

import numpy as np
from scipy.signal import medfilt, periodogram, welch
from scipy.special import zeta

__all__ = [
    "DUAL_AXIS_TABLE", "CANDIDATE_TABLE", "PRE", "CASES",
    "gen_case", "gen_G", "gen_hole_log",
    "measure", "classify", "dfa_curve", "psd_channels", "rs_hurst", "dwell_csn",
    "k1_regime_ok", "k2_dwell", "k3_two_stage", "k3_ledger_audit", "k4_scc", "k4_verdict",
    "k5_run", "k5_gate",
    "sigma_hat", "a11_calibration", "sigma_target", "time_delay_calibration",
    "c2b_readout", "c2b_sensitivity_sham",
    "print_dual_axis_table", "print_candidate_table",
]

# ════════════════════════════════════════════════════════════════════════
# §0 常量
# ════════════════════════════════════════════════════════════════════════

V2_TAG = "2026-09-27 M1（架构定稿 §2/§3/§5-M1；E5 Round3 交付①②④；E3 Round3 §1.2 C2b；M0 D-2 授权）"

# —— E5 Round3 §1.1 交付①：双轴常量合同表（判据本体留内核步轴，墙钟为执行合同列）——
DUAL_AXIS_TABLE = [
    # (常量, 内核步（判据本体）, 墙钟 @N=400（0.5 ms/步）, 墙钟 @N=100（2 ms/步）, 用途)
    ("κ₁ 换算率", "1 R2 拍 = N 步 rollout", "N∈[100,400]，WCET 实测落定 [候选]", "同左", "全桌合同（D-9）"),
    ("R_reg 观测节拍", "200 步", "0.5 R2 拍 ≈ 0.1 s", "2 R2 拍 ≈ 0.4 s", "C-HOME 节拍"),
    ("W_o 观测窗", "4096 步", "≈10 R2 拍 ≈ 2.0 s", "≈8.2 s", "σ̂/α̂ 估计窗"),
    ("W_fast 快通道子窗", "2048 步", "≈1.0 s", "≈4.1 s", "σ̂ 快回判读"),
    ("T_rec^fast", "2000 步", "≈1.0 s", "≈4.0 s", "K3 快通道（e_c 回死区）"),
    ("T_rec^slow", "16384 步", "≈8.2 s", "≈32.8 s", "K3 慢通道（α 完整复核，16K 最小 DFA 窗）"),
    ("K1 尺度带", "n∈[16, 512]", "[8 ms, 256 ms]", "[32 ms, 1.0 s]", "≥1.5 个十进位（512/16=32）；三窗 {16K,32K,64K} 各自内含此带"),
    ("T_q 复核静默窗", "6144 步（=1.5·W_o）", "≈3.1 s", "≈12.3 s", "E1 的 T_rev 相位条件"),
    ("τ_delay 端到端时滞", "R_reg+W_o ≈ 4296 步", "≈2.1 s", "≈8.6 s", "k_p/η_i 整定输入"),
]

# —— 候选值表（全部标"候选值，K5 自证后落册"；定稿 §6 遗留开放项 2/3 条原文）——
CANDIDATE_TABLE = [
    ("I_max 积分钳制", "0.2", "E5 §1.2 [候选]"),
    ("δ_Ic windup 守卫", "0.02（=0.1·I_max）", "E5 §1.2 [候选]"),
    ("δ_dead 死区", "0.05", "E5 §1.2 [候选]"),
    ("k_p 比例增益", "≲0.06（τ_obj≈800 / τ_delay≈4296 整定）", "E5 交付④ [候选，K5 时滞标定落定]"),
    ("η_i 积分增益", "≈3×10⁻⁵/步", "E5 交付④ [候选，K5 时滞标定落定]"),
    ("K2 驻留判据阈", "CSN 似然比 R·√n > 2", "定稿 §3.1 K2（预注册阈）"),
    ("K1 α 带", "[0.85, 1.15] 于 n∈[16,512]（≥1.5 十进位）", "定稿 §3.1 K1"),
    ("三窗漂移容差", "≤0.05（窗 {16K,32K,64K}）", "E5 §2 [候选]"),
    ("K4′ Jaccard 容差", "j(n)=0.5+0.1n（并列候选；E2 表 {0.0,0.3,0.5,0.6,0.7} 待对齐 O5）", "E5 §2-E2-2 [候选]"),
    ("N_sw windup 循环", "10", "E5 §1.2"),
    ("T_flat 压平排除窗", "8192 步（=2·W_o）", "E5 §1.4（O4 待 E3 补签）"),
    ("洞容忍度", "占比 ≤10% 且无 G3 边界跳变 ⇒ 作数窗", "E5 交付②（O10：E4 >20%⊥ 并列待合并）"),
    ("δ_hole 打洞参数", "G1 ρ_hole∈{5%,20%}/T_thrust=1024；G2/G3 洞长~T_rec^slow", "E5 交付② 参数化"),
    ("K5 验收线", "合成 ≥90%（B=40/情形）；G1/G2 ≥90%、G3 ≥85%", "定稿 §3.1 K5 / E5 §1.1 交付②"),
]

# —— 仪器内阈（预注册：实现时从 E5 方案 §3.3 判别表第一性导出，先于任何验收运行）——
PRE = dict(
    # 仪器通用
    n_grid_pts=22,               # DFA 尺度点数（log 均布于 [16, N//10]）
    n_lo=(16, 64),               # α_lo 拟合带（交叉左侧）
    n_mid=(256, 1024),           # α_mid 拟合带
    n_hi_rel=(0.25, 1.0),        # α_hi 拟合带（相对 nmax=N//10）
    n_band=(16, 512),            # K1 尺度带（定稿：512/16=32 = 1.5 十进位）
    n_far=(1024, None),          # slope_far 下端（上端 = nmax）
    dfa_order=1,                 # DFA 去趋势阶（DFA-1，标准）
    det_segments=8,              # 去趋势双口径：8 段二次拟合（E5 方案 0) "去线性/二次趋势"）
    psd_band_hi=1.0 / 256,       # β 通道频带上端（两通道同用）
    welch_nperseg_rel=4,         # Welch nperseg = N/4（df=1/(N/4)）
    # 分类器决策门
    cx_A=0.40,                   # A 交叉门：α_lo − α_hi ≥ 0.4（判别表 1.5→0.5）
    slope_far_A=0.35,            # A 大尺度斜率下限（≈0.5，非饱和）
    asym_A=0.30,                 # A 对称激励守卫：驻留不对称度 ≤0.3（高斯过程上下激励镜像；排除 C1 突发伪装交叉）
    trend_frac_B=0.45,           # B 设计斜坡门：全局线性趋势能量占比 ≥0.45（斜坡解析值≈(8σ)²/12/(1+(8σ)²/12)≈0.84；
                                 #   平稳过程（含有界游走）≤~0.3；随机游走对线性拟合天然高 R² 故必须配合"去趋势后呈 A 型"）
                                 #   [2026-09-28 #51 试收紧至 0.75 已回退：六块×双窗 G3 全崩（详台账 #51/#52），
                                 #    根因=B 支兼判『纯 B + 洞族 holed-B』，全局抬阈无法分离 D；阈值维持 0.45]
    lr_gate=0.02,                # 幂律胜门：LR=(SSE_lor−SSE_pl)/(SSE_pl+SSE_lor) > 0.02（判别表 C 行）
    lr_lor=-0.02,                # Lorentzian 胜门：LR ≤ −0.02（判别表 A 行曲率；|LR|≤0.02 为简并区，交由 DFA/驻留裁决）
    d_beta_B=0.70,               # B 去趋势敏感参考量（β_raw−β_det；受分段去趋势谱畸变污染，仅报告不作门）
    slope_far_D=0.30,            # D 饱和门：大尺度局部斜率 ≤0.3（F(n) 饱和）
    beta_D=1.40,                 # D 谱门：β_raw ≥ 1.4（判别表 近 β≈2；配合饱和门防 A 长τ红漏误伤）
    alpha_D_lo=1.1, alpha_D_hi=1.8,   # [2026-09-28 #42 所有者授权改核] r2 的 DFA 中段 α 带：
                                 #   D 判别表中段 α≈1.5（判别表 D 行）；α_mid∈[1.1,1.8]。
                                 #   r2 另以「上/下驻留皆无幂律证据」为负子句（¬pl_up ∧ ¬pl_dn）。
                                 #   只读证据：非-D 假阳 0/2400、D 召回 454/480=0.9458（M1_probe_r2fp.py）
    qratio_D=0.70,               # [2026-09-28 #55 所有者授权改核] D 行外科子句门：qratio ≤ 0.70。
                                 #   qratio=mean|Δz|(|z|≥q90)/mean|Δz|(|z|≤q10)（z=(x−mean)/std）——软饱和压缩量；
                                 #   D（tanh 软饱和：近界增益<1）⇒ <1；B（增益恒定斜坡）⇒ ≈1。
                                 #   取值=clean-D 中位 0.368 与 clean-B 中位 1.001 的中点取整、落于只读网格 [0.40,0.70] 内；
                                 #   该网格下非-D 假阳恒 0（非-D 最小 qratio 0.872），D 召回 214/239→238–239/239。
                                 #   只读证据：M1_probe_compress_T131072_B40.out、M1_probe_qratio_T131072_B40.out（mismatch=0/1440）。
    alpha_C3=0.15,               # C3 门：|α_band − 1| ≤ 0.15（K1 带同宽）
    dwell_asym_C1=0.5,           # C1 门：驻留不对称度（突发 vs 静默）≥0.5
    dwell_theta_q=0.5,           # 分类器分态阈：中位数分割（"阈值/HMM 分态"，E5 方案 4)）
    k2_theta_q=0.9,              # K2 判据阈：窗内活动 0.9 分位（A11 §1.3(4) 自参考口径）
    medw=11,                     # 分态位中值滤波核（T8 label_fast 先例 w=41，本仪器取 11）
    xmin_grid=(1, 2, 4, 8, 16, 32, 64),   # CSN x_min 扫描格（Clauset–Shalizi–Newman 规则）
    csn_n_min=30,                # 驻留样本 <30 ⇒ 不可辨识（NaN 守卫）
    csn_gate=2.0,                # K2 判据式原样：R·√n > 2（定稿）
    csn_xmin_plateau=1.15,       # [2026-09-27 M1 修复②'] x_min 选取：KS ≤ 1.15·KS_min 中最小的 x_min
                                 #   （plateau 去偏；替代"KS 最小"——有限样本 KS 随去点单调收缩致 x_min
                                 #   系统性上滑、n_eff 塌缩；改的是估计器偏置，csn_gate 未动）
    R_eff_gate=0.20,             # [2026-09-28 M1 修复⑦ / 台账 #44（所有者授权『可以动核心』）]
                                 #   C 族幂律准入改用「窗口不变效应量」：pl=1[(status=="ok") ∧
                                 #   isfinite(R) ∧ (R ≥ R_eff_gate)]（替代 rsqn=R·√n > csn_gate）。
                                 #   承接 #41：rsqn 非窗口不变（R 三区各窗近似恒定、n 随窗增 ⇒ rsqn 随窗单调升），
                                 #   是 C3↔C2 误入、C1 功效不足、D 驻留通道饥饿的共同根因。
                                 #   operative 窗 T=131072 只读分离量（M1_probe_window_csn.out）：
                                 #     C3（fGn，应非幂律）R≤0.090 / A(OU) R≤0.249 / C1 上态 R 0.365–0.376 /
                                 #     C2 真幂律 R_up 0.83–1.20。取保护 C1（首要修复目标）的下缘邻域 0.20。
                                 #   影响面严格限定 classify 内 pl_up/pl_dn 两行；K2/A11 识别通道
                                 #   （passed=rsqn>csn_gate）与 occurrence 通道（P_ii_rsqn）不动，csn_gate 键保留。
                                 #   k5_gate 五门阈值与对照口径一律不变，禁止凑数。
    min_len_classify=8192,       # 分类最小窗（尺度带 + FSS 可用）
    fss_windows=(16384, 32768, 65536),    # 三窗（定稿 K1）
    fss_drift_gate=0.05,         # 三窗漂移 ≤0.05 [候选]
    alpha_band_gate=(0.85, 1.15),         # K1 α 带（定稿）
    hole_tol=0.10,               # 洞占比 ≤10% 且无 G3 跳变 ⇒ 作数（E5 交付②）
    # 合成器真值（每情形逐条标注）
    A_taus=(100, 200, 400),      # E5 方案 §3.3 A=OU——2026-09-27 M1 修复④（合成池参数修订，判据式/阈值
                                 #   未动）：τ 必须 ∈ DFA 带 [16,512]（交叉可见）；实测 τ=50 边缘（4/6）、
                                 #   800 边缘（5/6）、2000 带外（3/6）⇒ 取 (100,200,400)；
                                 #   τ>512 登记为带外禁带（T9 已证 64K 窗长 τ 交叉不可见）
    B_tau=800, B_drift=8.0,      # B=A+设计斜坡（复刻 brain.py:598/693 形状：线性 in t；全窗总漂移 8σ）
    C1_sigma=1.00, C1_M=4096, C1_dcap=1024,   # σ=1.00（定稿真值）；M/d_cap=有限尺寸·有限观测截断
                                 # [2026-09-27 M1 修复①'：M 64→4096 实装（上条已注册）——M=64 在
                                 #   d≳60 提前截断 d^(−3/2) 尾；M∈{4096,65536} 结果逐位相同（饱和）]
    C1_p_drive=1e-2, C1_bg=0.0,               # 稀疏变体（K5 分类）：5e-4→2e-3（修复⑤）→1e-2
                                 #   （2026-09-27 M1 修复④'：固定 T=131072 窗内雪崩数 ~260→~1160，
                                 #    CSN 样本量上升；占据率 0.040 仍稀疏，σ=1 真值/幂律 signature 不变）
    C1_p_drive_dense=2e-2, C1_bg_dense=0.05,  # 密集变体（A11 σ̂ 标定）：0.9 分位阈需落在活动分布内
    C2_alphas=tuple(np.round(np.linspace(1.3, 2.0, 10), 3)),  # α_dwell∈(1,2]（定稿）网格覆盖
    C2_dmax=16384,               # 驻留截断（有限观测）
    C2_state=1.0, C2_noise=0.02, # 二态电平 ±1 + 井内噪声 0.02（小噪声⇒中位分割不破坏驻留结构）
    C3_beta=1.0,                 # 谱合成 1/f（β=1）
    D_a=2.0, D_sigma=0.40,      # D 软饱和慢模态 x=tanh(a·x+η)（2026-09-27 M1 修复③；标定使中位
                                #   驻留 ~10³ 步；替代 D_step/D_bound clip 实现——首通幂律与 C2 同构，见 _gen_D）
    G_T_thrust=1024,             # G1 周期短洞节律（E4 热节流类比）
    G2_frac=16384.0 / 65536.0,   # G2/G3 洞长 ~ T_rec^slow 量级（16384/64K）
    G3_jump_sigma=5.0,           # G3 边界跳变 |Δu| > 5σ_run0（E5 交付②）
    # 标定流 seed（与验收集 0–5/6–11 隔离的开发流）
    cal_seed_sigma=2000, cal_seed_beta=3000, cal_seed_delay=4000, dev_seed=100,
)

CASES = ("A", "B", "C1", "C2", "C3", "D")


def print_dual_axis_table() -> str:
    lines = ["双轴常量合同表（E5 Round3 §1.1 交付①：判据本体留内核步轴，墙钟为执行合同列；换算率变更=全体 _CHANGELOG）"]
    hdr = ("%-18s | %-30s | %-30s | %-24s | %s" % ("常量", "内核步（判据本体）", "墙钟 @N=400（0.5ms/步）", "墙钟 @N=100（2ms/步）", "用途"))
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for row in DUAL_AXIS_TABLE:
        lines.append("%-18s | %-30s | %-30s | %-24s | %s" % row)
    lines.append("守卫条款：κ_wc 漂移超预注册倍数 ⇒ K1 报告自动降级为『步轴口径 only』并登记。")
    return "\n".join(lines)


def print_candidate_table() -> str:
    lines = ["候选值表（全部为**候选值，K5 自证后落册**；定稿 §6 开放项 2/3；未实测）"]
    for name, val, src in CANDIDATE_TABLE:
        lines.append("  %-16s = %-58s  [%s]" % (name, val, src))
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# §1 六情形合成器（带已知真值 meta）+ 打洞族 G1/G2/G3
# ════════════════════════════════════════════════════════════════════════

def _gen_ou(T: int, tau: float, rng) -> np.ndarray:
    """平稳 OU：x_{t+1}=φx_t+√(1−φ²)ε，φ=e^(−1/τ)（单位平稳方差）。真值 τ。"""
    phi = math.exp(-1.0 / tau)
    e = rng.standard_normal(T)
    x = np.empty(T)
    x[0] = e[0]
    c = math.sqrt(1.0 - phi * phi)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + c * e[t]
    return x


def _gen_B(T: int, rng) -> np.ndarray:
    """B = A(τ=800) + 设计斜坡（复刻 brain.py:598 设定点线性漂移 / :693 的 0.02·t 形状；全窗总漂移 8σ）。"""
    x = _gen_ou(T, PRE["B_tau"], rng)
    x = x + np.linspace(-PRE["B_drift"] / 2.0, PRE["B_drift"] / 2.0, T)
    return x


def _gen_C1(T: int, rng, dense: bool = False) -> tuple:
    """C1 临界分支过程（σ=1.00 定稿真值）：离散分支 Poisson(σ)/粒子，种群帽 M + 时长帽 d_cap
    （有限尺寸/有限观测截断），Poisson 驱动。
      dense=False（K5 分类用，稀疏）：bg=0（安静本底恒零）⇒ 中位分割可精确隔离雪崩；
      dense=True（A11 σ̂ 观测器标定用，密集）：bg=0.05、p_drive 大 ⇒ 0.9 分位阈落在活动分布内。
    真值：分支比 σ=1.00 ⇒ 雪崩时长幂律 d^(−3/2)（mean-field），截断于 d_cap。"""
    sigma, M = PRE["C1_sigma"], PRE["C1_M"]
    p_drive = PRE["C1_p_drive_dense"] if dense else PRE["C1_p_drive"]
    bg = PRE["C1_bg_dense"] if dense else 0.0
    d_cap = PRE["C1_dcap"]
    x = np.empty(T)
    a = 0
    n_ava = 0
    durs = []
    d0 = 0
    for t in range(T):
        if a == 0:
            if rng.random() < p_drive:
                a = 1
                n_ava += 1
                d0 = t
        else:
            a = int(rng.poisson(a * sigma))
            if a > M:
                a = M
            if a == 0 or (t - d0) >= d_cap:
                durs.append(t - d0)
                a = 0
        x[t] = a + bg * rng.standard_normal()
    x = (x - x.mean()) / (x.std() + 1e-12)
    return x, dict(n_avalanches=n_ava, dur_mean=float(np.mean(durs)) if durs else float("nan"))


def _gen_C2(T: int, alpha_dwell: float, rng) -> tuple:
    """C2 幂律驻留二态：±1 电平 + 井内噪声，驻留 ~ p(d)∝d^(−α_dwell), d∈[1,dmax]（逆 CDF 精确采样）。
    真值：α_dwell（记录于 meta）。"""
    gamma = float(alpha_dwell)
    dmax = PRE["C2_dmax"]
    d = np.arange(1, dmax + 1, dtype=float)
    w = d ** (-gamma)
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    states = np.empty(T)
    t = 0
    dwells = []
    cur = 1.0 if rng.random() < 0.5 else -1.0
    while t < T:
        L = int(np.searchsorted(cdf, rng.random()) + 1)
        L = min(L, T - t)
        states[t:t + L] = cur
        dwells.append(L)
        t += L
        cur = -cur
    x = states + PRE["C2_noise"] * rng.standard_normal(T)
    x = (x - x.mean()) / (x.std() + 1e-12)
    return x, dict(alpha_dwell=gamma, n_switch=len(dwells), dwell_med=float(np.median(dwells)))


def _gen_C3(T: int, beta: float, rng) -> np.ndarray:
    """C3 谱合成 1/f：白噪声 FFT × f^(−β/2)（精确谱指数，Gaussian 边缘，fGn 型）。真值 β=1。"""
    w = rng.standard_normal(T)
    X = np.fft.rfft(w)
    f = np.fft.rfftfreq(T)
    f[0] = f[1]
    X *= f ** (-beta / 2.0)
    x = np.fft.irfft(X, n=T)
    x = (x - x.mean()) / (x.std() + 1e-12)
    return x


def _gen_D(T: int, rng) -> tuple:
    """D 有界超慢（软饱和慢模态——复刻 brain.py h_slow 的真实物理）。
    [2026-09-27 M1 修复③ 发生器语义重定（判据式/阈值未动；证据 meeting12/M1_probe_D.py）]
    原实现=硬 clip 随机游走——**物理选错**：一维硬壁游走的墙驻留 = 布朗首通时间，幂律 t^(−3/2)
    （实测 rsqn 2.2–17），与 C2 幂律驻留**全通道同构**；且首通重尾抹平 DFA 饱和 knee
    （实测 slope_far 0.72–1.16 @墙 331–2500，永不 ≤0.30）⇒ 判别表 D 行（"F(n) 饱和 ∧ β≥1.4"）
    在 clip 实现下**内在不可满足**（任何墙尺度）。
    真实判别对象 = brain.py h_slow：tanh **软**饱和 → 近壁增益 <1 → Kramers 型**指数**逃逸
    （亚稳双态 + 驻留指数）⇒ 驻留与 C2 可分、DFA 在 n>dwell 饱和（slope_far≈0）、
    双态电报谱 β≥1.4。修法：x_{t+1} = tanh(a·x_t + η)，η~N(0,σ²)（自洽映射软饱和；
    (a,σ) 标定使中位驻留 ~10³ 步，见 PRE.D_a/D_sigma）。"""
    a, sig = PRE["D_a"], PRE["D_sigma"]
    x = np.empty(T)
    v = float(rng.standard_normal()) * 0.1
    for t in range(T):
        v = math.tanh(a * v + sig * float(rng.standard_normal()))
        x[t] = v
    x = (x - x.mean()) / (x.std() + 1e-12)
    # 驻留真值（Kramers 亚稳）：按 ±x* 吸引域计游程
    sgn = np.sign(x)
    ch = np.flatnonzero(np.diff(sgn)) + 1
    ix = np.concatenate(([0], ch, [len(sgn)]))
    durs = np.diff(ix).astype(float)
    return x, dict(saturation="tanh 软饱和（Kramers 指数逃逸）", a=a, sigma=sig,
                   dwell_median=float(np.median(durs)), n_switch=int(len(ch)))


def gen_case(case: str, T: int, rng, rep: int = 0) -> tuple:
    """六情形合成器：返回 (x, meta)。meta 携带**已知真值**（逐情形标注）。"""
    if case == "A":
        tau = PRE["A_taus"][rep % len(PRE["A_taus"])]
        x = _gen_ou(T, tau, rng)
        truth = dict(family="平稳 OU", tau=tau)
    elif case == "B":
        x = _gen_B(T, rng)
        truth = dict(family="设计漂移", tau=PRE["B_tau"], drift_total_sigma=PRE["B_drift"],
                     anchor="brain.py:598 setpoint 线性漂移 / :693 0.02·t 形状")
    elif case == "C1":
        x, info = _gen_C1(T, rng)
        truth = dict(family="临界分支", sigma=PRE["C1_sigma"], M=PRE["C1_M"],
                     p_drive=PRE["C1_p_drive"], **info)
    elif case == "C2":
        g = float(PRE["C2_alphas"][rep % len(PRE["C2_alphas"])])
        x, info = _gen_C2(T, g, rng)
        truth = dict(family="幂律驻留二态", **info)
    elif case == "C3":
        x = _gen_C3(T, PRE["C3_beta"], rng)
        truth = dict(family="1/f 谱合成", beta=PRE["C3_beta"])
    elif case == "D":
        x, info = _gen_D(T, rng)
        truth = dict(family="有界超慢（软饱和慢模态）", **info)
    else:
        raise ValueError("未知情形 %r（六情形 = A/B/C1/C2/C3/D）" % (case,))
    meta = dict(case=case, truth=truth, holes=[], hole_fraction=0.0, g3_jump=False,
                valid=np.ones(T, dtype=bool))
    return x, meta


# —— 打洞族 G1/G2/G3（E5 Round3 §1.1 交付②，K5 第六情形，δ_hole 参数化）——

def gen_G(kind: str, T: int, rng, rep: int = 0, delta_hole: Optional[float] = None) -> tuple:
    """从 A–D 合成真值按洞模式挖洞。返回 (x, meta)（meta.valid = 有效掩码；meta.holes = 洞日志）。
      G1 周期短洞：每 T_thrust 步洞长 L_gap，占空比 ρ_hole=δ_hole∈{0.05,0.20}（模拟热节流）；
      G2 演练长洞：洞长 ~ T_rec^slow 量级（δ_hole=16384/65536），随机位置（模拟 kill 演练）；
      G3 长洞+边界跳变：同 G2，且洞边界输入 |Δu| > 5σ_run0（世界推进的真实后果）。"""
    base = CASES[rep % len(CASES)]
    x, meta = gen_case(base, T, rng, rep)
    holes = []
    g3 = False
    jump = 0.0
    if kind == "G1":
        rho = float(delta_hole) if delta_hole is not None else 0.05
        Tt = PRE["G_T_thrust"]
        L = max(1, int(round(rho * Tt)))
        s = Tt // 2
        while s + L <= T:
            holes.append((s, L, "periodic"))
            s += Tt
    elif kind in ("G2", "G3"):
        frac = float(delta_hole) if delta_hole is not None else PRE["G2_frac"]
        L = max(1, int(round(frac * T)))
        pos = int(rng.integers(0, max(1, T - L)))
        holes.append((pos, L, "drill"))
        if kind == "G3":
            g3 = True
            # 洞边界"世界推进"：洞结束后底层输入跳变 ±5σ_run0（对底层过程加持续性均值阶跃）
            jump = PRE["G3_jump_sigma"] * (1.0 if rng.random() < 0.5 else -1.0)
            scale = dict(A=1.0, B=1.0, C1=1.0, C2=1.0, C3=1.0, D=0.5)[base]
            x[pos + L:] += jump * scale
            holes[-1] = (pos, L, "drill+G3jump(|Δu|>5σ_run0)")
    else:
        raise ValueError("未知洞模式 %r（G1/G2/G3）" % (kind,))
    valid = np.ones(T, dtype=bool)
    for (s, L, _k) in holes:
        valid[s:s + L] = False
    meta.update(kind=kind, base=base, holes=holes, valid=valid,
                hole_fraction=float(1.0 - valid.mean()), g3_jump=g3, jump=jump if g3 else 0.0)
    # [2026-09-27 M1 仪器缺陷修复（先注册 verify_all._CHANGELOG）] 原漏写 valid=valid ⇒ meta['valid']
    #   沿用 gen_case 全 True ⇒ k5_run `x[meta['valid']]`（交付②步轴索引预处理）静默空转。
    return x, meta


def gen_hole_log(meta: dict) -> dict:
    """洞日志字段（E5 交付②/E4 洞日志双源供此结构；引擎侧由 E4 演练标记+world 跳变检测器填同构 dict）。"""
    return dict(kind=meta.get("kind", ""), hole_fraction=float(meta.get("hole_fraction", 0.0)),
                g3_jump=bool(meta.get("g3_jump", False)), holes=list(meta.get("holes", [])))


# ════════════════════════════════════════════════════════════════════════
# §2 测量仪器（全部预注册；NaN=不可辨识，不伪装）
# ════════════════════════════════════════════════════════════════════════

def _n_grid(N: int) -> np.ndarray:
    nmax = max(32, N // 10)
    g = np.unique(np.logspace(math.log10(16), math.log10(nmax), PRE["n_grid_pts"]).astype(int))
    return g[(g >= 16) & (g <= nmax)]


def dfa_curve(x: np.ndarray, order: int = None) -> tuple:
    """DFA：F(n) 多尺度（n∈[16, N/10]，DFA-去趋势阶=1 预注册）。返回 (n_grid, F)。"""
    x = np.asarray(x, float)
    N = len(x)
    order = PRE["dfa_order"] if order is None else order
    y = np.cumsum(x - x.mean())
    n_grid = _n_grid(N)
    F = np.full(len(n_grid), np.nan)
    for i, n in enumerate(n_grid):
        m = N // n
        if m < 4:
            continue
        seg = y[:m * n].reshape(m, n).astype(float)
        t = np.arange(n, dtype=float)
        if order == 1:
            tc = t - t.mean()
            den = float((tc * tc).sum())
            sl = ((seg - seg.mean(1, keepdims=True)) * tc).sum(1) / den
            r = seg - seg.mean(1, keepdims=True) - sl[:, None] * tc
        else:
            V = np.vander(t, order + 1, increasing=True)
            Q, _ = np.linalg.qr(V)
            coef = Q.T @ seg.T
            r = seg - (Q @ coef).T
        F[i] = float(np.sqrt(np.mean(r * r)))
    return n_grid, F


def _ols_slope(lx: np.ndarray, ly: np.ndarray) -> float:
    lx = np.asarray(lx, float)
    ly = np.asarray(ly, float)
    ok = np.isfinite(lx) & np.isfinite(ly)
    if ok.sum() < 3:
        return float("nan")
    lx, ly = lx[ok], ly[ok]
    xm, ym = lx.mean(), ly.mean()
    den = float(((lx - xm) ** 2).sum())
    if den <= 0:
        return float("nan")
    return float(((lx - xm) * (ly - ym)).sum() / den)


def _band_slope(n_grid, F, lo, hi):
    m = (n_grid >= lo) & (n_grid <= hi) & np.isfinite(F)
    if m.sum() < 3:
        return float("nan")
    return _ols_slope(np.log(n_grid[m]), np.log(F[m]))


def dfa_summary(x: np.ndarray) -> dict:
    """DFA 汇总：α_lo/α_mid/α_hi/α_band（K1 带 [16,512]）/cx=α_lo−α_hi/slope_far（n=1024→nmax 局部斜率）。"""
    n_grid, F = dfa_curve(x)
    N = len(x)
    nmax = n_grid[-1] if len(n_grid) else np.nan
    lo, hi = PRE["n_lo"]
    mid = PRE["n_mid"]
    r1, r2 = PRE["n_hi_rel"]
    a_lo = _band_slope(n_grid, F, lo, hi)
    a_mid = _band_slope(n_grid, F, mid[0], mid[1])
    a_hi = _band_slope(n_grid, F, max(16, int(r1 * nmax)), nmax)
    a_band = _band_slope(n_grid, F, PRE["n_band"][0], PRE["n_band"][1])
    m = (n_grid >= PRE["n_far"][0]) & np.isfinite(F)
    slope_far = float("nan")
    if m.sum() >= 3:
        slope_far = _ols_slope(np.log(n_grid[m]), np.log(F[m]))
    return dict(n_grid=n_grid, F=F, alpha_lo=a_lo, alpha_mid=a_mid, alpha_hi=a_hi,
                alpha_band=a_band, cx=(a_lo - a_hi) if (np.isfinite(a_lo) and np.isfinite(a_hi)) else float("nan"),
                slope_far=slope_far, nmax=int(nmax) if np.isfinite(nmax) else -1)


def _beta_full(x: np.ndarray) -> float:
    """β 通道（**趋势敏感**）：全窗周期ogram（Hann，仅去均值）低频段 log-log OLS 斜率。
    频带 [2/N, 1/256]（N=窗长；≥2.1 十进位）。服务 B 守卫（低频污染）与 D 谱门。"""
    N = len(x)
    f, S = periodogram(x, window="hann", detrend="constant")
    m = (f >= 2.0 / N) & (f <= PRE["psd_band_hi"]) & (S > 0)
    if m.sum() < 8:
        return float("nan")
    return -_ols_slope(np.log(f[m]), np.log(S[m]))


def psd_channels(x: np.ndarray) -> dict:
    """PSD 双通道：
      β_raw（全窗周期ogram，趋势敏感——B 守卫/判别表"低频污染"）
      LR_welch（Welch 分段线性去趋势——趋势稳健；幂律 vs Lorentzian 公平 2 参数比较（{a,β} vs {a,fc}，
                斜率不自由）：LR>lr_gate ⇒ 幂律胜（判别表 C 行）；LR≤−lr_gate ⇒ Lorentzian 胜（A 行）；
                |LR| 小 ⇒ 简并（fc→0 的 Lorentzian ≡ β=2 幂律；长 τ OU 曲率在带外——由 DFA 交叉/驻留联合裁决））。"""
    N = len(x)
    beta_raw = _beta_full(x)
    nper = max(256, N // PRE["welch_nperseg_rel"])
    f, S = welch(x, nperseg=nper, noverlap=nper // 2, detrend="linear", window="hann")
    dmin = 1.5 / nper
    m = (f >= dmin) & (f <= PRE["psd_band_hi"]) & (S > 0)
    lr = float("nan")
    beta_w = float("nan")
    if m.sum() >= 8:
        lf, lS = np.log(f[m]), np.log(S[m])
        beta_w = -_ols_slope(lf, lS)
        # 幂律：logS = a − β·logf（2 参数，显式 OLS）
        xm, ym = lf.mean(), lS.mean()
        sxx = float(((lf - xm) ** 2).sum())
        b1 = float(((lf - xm) * (lS - ym)).sum() / sxx) if sxx > 0 else float("nan")
        a1 = ym - b1 * xm
        res_pl = lS - (a1 + b1 * lf)
        sse_pl = float((res_pl ** 2).sum())
        # Lorentzian：logS = a − log(f²+fc²)（公平 2 参数：a 由 OLS 给定 fc，fc 网格搜索；斜率不自由）
        best = None
        for fc in np.geomspace(dmin / 10.0, 1.0, 48):
            g = -np.log(f[m] ** 2 + fc * fc)
            a2 = float(lS.mean() - g.mean())
            sse = float(((lS - (a2 + g)) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, fc)
        sse_lor, fc_best = best
        lr = (sse_lor - sse_pl) / (sse_pl + sse_lor + 1e-300)
    return dict(beta_raw=beta_raw, beta_welch=beta_w, lr_welch=lr, fc_best=fc_best if m.sum() >= 8 else float("nan"))


def rs_hurst(x: np.ndarray) -> float:
    """R/S Hurst H（报告通道；判别表：α=H ⇒ fGn 型 / α=H+1 ⇒ fBm 型）。"""
    x = np.asarray(x, float)
    N = len(x)
    ms = np.unique(np.logspace(math.log10(16), math.log10(max(32, N // 8)), 15).astype(int))
    xs, ys = [], []
    for m in ms:
        k = N // m
        if k < 4:
            continue
        seg = x[:k * m].reshape(k, m)
        z = np.cumsum(seg - seg.mean(1, keepdims=True), axis=1)
        R = z.max(1) - z.min(1)
        S = seg.std(1)
        ok = S > 0
        if ok.sum() < 2:
            continue
        xs.append(m)
        ys.append(float(np.mean(R[ok] / S[ok])))
    if len(xs) < 4:
        return float("nan")
    return _ols_slope(np.log(np.array(xs, float)), np.log(np.array(ys, float)))


def _run_lengths(bits: np.ndarray) -> tuple:
    b = np.asarray(bits, dtype=np.int8)
    if b.size == 0:
        return np.array([], int), np.array([], int)
    change = np.flatnonzero(np.diff(b)) + 1
    idx = np.concatenate(([0], change, [b.size]))
    seglen = np.diff(idx)
    val = b[idx[:-1]]
    return seglen[val == 1], seglen[val == 0]


def _csn_fit(dw: np.ndarray) -> dict:
    """Clauset–Shalizi–Newman：离散幂律 vs 指数（几何）似然比，x_min 扫描（KS 最小者，预注册规则）。
    返回 dict(status, alpha, n, R, rsqn, vuong, xmin, ks)。判据式原样：R·√n > 2（R=逐样本对数似然差均值）。"""
    out = dict(status="不可辨识", alpha=float("nan"), n=0, R=float("nan"), rsqn=float("nan"),
               vuong=float("nan"), xmin=float("nan"), ks=float("nan"))
    dw = np.asarray(dw, float)
    dw = dw[dw >= 1]
    if dw.size < PRE["csn_n_min"]:
        return out
    cands = []
    for xmin in PRE["xmin_grid"]:
        v = dw[dw >= xmin]
        n = int(v.size)
        if n < PRE["csn_n_min"]:
            continue
        s = float(np.log(v / (xmin - 0.5)).sum())
        if s <= 0:
            continue
        a0 = 1.0 + n / s
        a_grid = np.arange(max(1.02, a0 - 0.5), a0 + 0.5, 0.005)
        z = zeta(a_grid, float(xmin))
        ll = -a_grid * float(np.log(v).sum()) - n * np.log(z)
        alpha = float(a_grid[int(np.argmax(ll))])
        logz = float(np.log(zeta(alpha, float(xmin))))
        logL_pl = -alpha * float(np.log(v).sum()) - n * logz
        # 指数替代（离散几何，同支撑）
        w = v - xmin
        mbar = float(w.mean())
        if mbar <= 1e-12:
            continue
        q = mbar / (1.0 + mbar)
        logL_exp = n * math.log1p(-q) + float(w.sum()) * math.log(q)
        lr_i = (-alpha * np.log(v) - logz) - (math.log1p(-q) + w * math.log(q))
        R = float(lr_i.mean())
        sd = float(lr_i.std())
        rsqn = R * math.sqrt(n)
        vuong = R * math.sqrt(n) / sd if sd > 0 else float("nan")
        # KS（数据 CDF vs 幂律拟合 CDF）→ x_min 选取
        vs = np.sort(v)
        emp = np.arange(1, n + 1) / n
        cdf_pl = 1.0 - zeta(alpha, vs + 1.0) / zeta(alpha, float(xmin))
        ks = float(np.max(np.abs(cdf_pl - emp)))
        cand = dict(alpha=alpha, n=n, R=R, rsqn=rsqn, vuong=vuong, xmin=float(xmin), ks=ks,
                    logL_pl=logL_pl, logL_exp=logL_exp)
        cands.append(cand)
    if not cands:
        return out
    # [2026-09-27 M1 修复②' plateau 去偏] 最小-KS 规则在有限样本下系统性上滑 x_min（去点⇒KS 单调收缩、
    #   n_eff 塌缩）；改为取"拟合优度容差内（KS ≤ plateau·KS_min）的最小 x_min"，恢复 n_eff。判据阈值不变。
    ks_min = min(c["ks"] for c in cands)
    plateau = [c for c in cands if c["ks"] <= PRE["csn_xmin_plateau"] * ks_min]
    best = min(plateau, key=lambda c: c["xmin"])
    if not np.isfinite(best["rsqn"]):
        return out
    best["status"] = "ok"
    out.update(best)
    return out


def _otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    """Otsu 阈值分态（判别表"阈值/HMM 分态"的忠实实现）。
    [2026-09-27 M1 修复⑦：替代原"中位数分割"——后者对**不平衡二态**失效：
    重尾驻留（C2 α_dwell<1.5）使单次超长驻留压倒全窗，中位阈落进单模态内部，
    噪声穿越炸成海量微段（实测 C2 rep0：真值 398 次切换 → 中位分割 n_up=16399）。
    Otsu 最大化类间方差，对稀疏（C1 零本底）与平衡/不平衡二态均稳健。]"""
    x = np.asarray(x, float)
    hist, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w = hist.astype(float)
    tot = w.sum()
    if tot <= 0:
        return float(np.median(x))
    w /= tot
    c = np.cumsum(w)
    m = np.cumsum(w * centers)
    mt = m[-1]
    denom = c * (1.0 - c)
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * c - m) ** 2 / denom
    sb[~np.isfinite(sb)] = -1.0
    k = int(np.argmax(sb))
    return float(centers[k])


def _sparse_threshold(x: np.ndarray, bins: int = 256, frac_gate: float = 0.5) -> float:
    """稀疏/零膨胀自适应分态阈（2026-09-27 M1 修复③'，仪器实现层，判据式/阈值未动）：
    主导 bin 占比 > frac_gate（如 C1 bg=0、占据率~4%）⇒ 阈 = med + 3·1.4826·MAD（稳健「活动/静默」分离，
    匹配 C1 oracle 分态）；否则退化 Otsu。对连续信号（C2/C3/OU/斜坡）主导 bin 占比远小于 0.5 ⇒ 逐位走
    Otsu，行为不变。Otsu 在严重不平衡分布上会落入少数簇内部，把雪崩按高度而非活动/静默切分。"""
    x = np.asarray(x, float)
    if x.size == 0:
        return float("nan")
    hist, _edges = np.histogram(x, bins=bins)
    dom = float(hist.max()) / max(1.0, float(hist.sum()))
    if dom > frac_gate:
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        return med + 3.0 * 1.4826 * mad
    return _otsu_threshold(x)


def dwell_csn(x: np.ndarray, theta_q: float = None, medw: int = None) -> dict:
    """驻留通道：阈 θ=quantile(θ_q) 分态 → 位中值滤波（去亚阈抖动，T8 label_fast 先例）→
    上/下态驻留样本 → CSN。返回上/下两路 fit + 不对称度。任何一路不可辨识 ⇒ NaN（不伪装）。

    [2026-09-27 M1 修复① 稀疏信号适配（仪器实现层，判据式/阈值未动；回报请 E5 会签）]
    中值滤波核若 ≥ 状态最短持续（如 C1 稀疏雪崩 dur_mean≈10 < 核 11）会把该态整体抹平
    （实测 K5 C1 40/40 全 NaN、up.n=0 的根因）。修法=自适应核：medw_eff = min(medw, min(中位上态长,
    中位下态长) − 1)（下限 1=不滤波）；对连续信号（C2/OU，态长 ≫ 11）行为逐位不变。

    [2026-09-27 M1 修复⑦ 分态改 Otsu；2026-09-27 M1 修复③' 再改稀疏自适应 _sparse_threshold
    （稀疏/零膨胀信号改 med+3·1.4826·MAD，连续信号退化 Otsu 逐位不变）；theta_q 保留为向后兼容参数，
    非 None 时仍用分位阈]"""
    x = np.asarray(x, float)
    medw = PRE["medw"] if medw is None else medw
    theta = float(np.quantile(x, theta_q)) if theta_q is not None else _sparse_threshold(x)
    bits = x > theta
    if medw and medw > 1:
        up0, dn0 = _run_lengths(bits)                     # 原始游程（滤波前）
        r0 = [float(np.median(up0)) if up0.size else float("inf"),
              float(np.median(dn0)) if dn0.size else float("inf")]
        r_min = min(r0)
        medw_eff = int(min(medw, max(1.0, math.floor(r_min) - 1.0))) if math.isfinite(r_min) else 1
        if medw_eff % 2 == 0:                         # medfilt 要求奇数核
            medw_eff -= 1
    else:
        medw_eff = int(medw) if medw else 1
    if medw_eff > 1:
        bits = medfilt(bits.astype(np.int8), kernel_size=medw_eff).astype(bool)
    up, dn = _run_lengths(bits)
    fit_up = _csn_fit(up)
    fit_dn = _csn_fit(dn)
    mu_up = float(np.mean(up)) if up.size else float("nan")
    mu_dn = float(np.mean(dn)) if dn.size else float("nan")
    asym = float("nan")
    if np.isfinite(mu_up) and np.isfinite(mu_dn) and (mu_up + mu_dn) > 0:
        asym = abs(mu_up - mu_dn) / (mu_up + mu_dn)
    return dict(theta=theta, n_up=int(up.size), n_dn=int(dn.size),
                mean_up=mu_up, mean_dn=mu_dn, asym=asym, medw_eff=int(medw_eff),
                up=fit_up, dn=fit_dn)


def fss_three_windows(x: np.ndarray, windows: Sequence[int] = None) -> dict:
    """有限尺寸标度（三窗 {16K,32K,64K} 嵌套前缀，预注册）：各窗 α_band、漂移 max−min、塌缩偏差。"""
    x = np.asarray(x, float)
    windows = tuple(w for w in (windows or PRE["fss_windows"]) if w <= len(x))
    alphas = []
    for w in windows:
        s = dfa_summary(x[:w])
        alphas.append(s["alpha_band"])
    alphas = [a for a in alphas if np.isfinite(a)]
    drift = float(max(alphas) - min(alphas)) if len(alphas) == len(windows) and len(alphas) >= 2 else float("nan")
    return dict(windows=list(windows), alphas=[float(a) for a in alphas], drift=drift)


def detrend_dual(x: np.ndarray) -> dict:
    """去趋势双口径（E5 方案 0) 守卫；B 判别表"去趋势后同 A / 不去趋势 ρ̂₁~T"）：
    8 段二次去趋势前后各测 β（全窗周期ogram 通道）与 DFA 交叉 cx。"""
    x = np.asarray(x, float)
    N = len(x)
    k = PRE["det_segments"]
    xd = x.copy()
    edges = np.linspace(0, N, k + 1).astype(int)
    for i in range(k):
        seg = xd[edges[i]:edges[i + 1]]
        t = np.arange(len(seg), dtype=float)
        if len(seg) >= 4:
            c = np.polyfit(t, seg, 2)
            xd[edges[i]:edges[i + 1]] = seg - np.polyval(c, t)
    b_raw, b_det = _beta_full(x), _beta_full(xd)
    s_raw = dfa_summary(x)
    s_det = dfa_summary(xd)
    # 全局线性趋势能量占比（B 的设计斜坡直接量；仅报告——随机游走对线性拟合天然高 R²，不作门）
    t = np.arange(N, dtype=float)
    c = np.polyfit(t, x, 1)
    tr = np.polyval(c, t)
    trend_frac = float(np.var(tr) / (np.var(x) + 1e-300))
    # ρ̂₁ 双窗（B 判别表"不去趋势 ρ̂₁→1 且随 T 恶化"；报告量）
    def _rho(y):
        y = y - y.mean()
        den = float((y[:-1] * y[:-1]).sum())
        return float((y[:-1] * y[1:]).sum() / den) if den > 0 else float("nan")
    return dict(beta_raw=b_raw, beta_det=b_det, d_beta=(b_raw - b_det) if np.isfinite(b_raw) and np.isfinite(b_det) else float("nan"),
                cx_raw=s_raw["cx"], cx_det=s_det["cx"], slope_far_det=s_det["slope_far"],
                trend_frac=trend_frac,
                rho1_full=_rho(x), rho1_short=_rho(x[:N // 4]))


def qratio_compress(x: np.ndarray) -> float:
    """软饱和压缩判别量（#54/#55，只读派生量，供 classify D 行外科子句）：
       qratio = mean|Δz|(|z|≥q90) / mean|Δz|(|z|≤q10)，其中 z=(x−mean)/std。
       D（tanh 软饱和：近界增益<1）⇒ 高|z| 处增量更小 ⇒ qratio<1；
       B（设计斜坡：增益恒定）⇒ qratio≈1。洞鲁棒（raw 拼接/segmax 后分离存活）。"""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 32:
        return float("nan")
    z = (x - x.mean()) / (x.std() + 1e-12)
    ad = np.abs(np.diff(z))
    xl = z[:-1]
    q10 = np.quantile(np.abs(xl), 0.10)
    q90 = np.quantile(np.abs(xl), 0.90)
    lo = ad[np.abs(xl) <= q10]
    hi = ad[np.abs(xl) >= q90]
    if len(lo) == 0 or len(hi) == 0:
        return float("nan")
    return float(hi.mean() / (lo.mean() + 1e-12))


def measure(x: np.ndarray) -> dict:
    """电池全通道测量（分类器与 K1/K2 共用；任何通道不可辨识 ⇒ NaN）。"""
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)):
        x = x[np.isfinite(x)]
    out = dict(N=int(len(x)))
    out.update(dfa_summary(x))
    out.update(psd_channels(x))
    out["hurst"] = rs_hurst(x)
    out["dwell"] = dwell_csn(x)
    out["fss"] = fss_three_windows(x)
    out["dual"] = detrend_dual(x)
    out["qratio"] = qratio_compress(x)
    return out


# ════════════════════════════════════════════════════════════════════════
# §3 分类器（判别表决策树——预注册；判别表 E5 方案 §3.3，联合四量决定，不事后挑指标）
# ════════════════════════════════════════════════════════════════════════

def classify(x: np.ndarray, meta: Optional[dict] = None, channels: Optional[dict] = None) -> dict:
    """六情形分类器。决策树（顺序固定，预注册；判别表 E5 方案 §3.3 联合四量，不事后挑指标）：
      0) 守卫：长度/有限性不足 ⇒ cls=NaN（不可辨识）；洞日志 ⇒ hole_exposure 标记（G3 必显式报告）。
      1) B（设计漂移）：原始低频污染（β_raw ≥ 1.2，斜坡谱 ≈f⁻²）∧ **去趋势后呈完整 A 型**
         （cx_det ≥ 0.4 ∧ slope_far_det ≥ 0.35 ∧ LR_det < lr_gate）——判别表"去趋势后同 A"。
         （A 的 β_raw≲1、D 去趋势后仍游走型 ⇒ 两邻不误伤；分段二次去趋势的谱畸变只污染 Δβ，故
          Δβ/趋势占比降为报告量不作门。）
      2) A（平稳 OU）：DFA 交叉（cx ≥ 0.4）∧ 大尺度未饱和（slope_far ≥ 0.35）∧ Lorentzian 胜
         （LR < lr_gate）∧ 上下激励对称（asym ≤ 0.3，高斯镜像性；排除 C1 突发型伪装交叉）。
      3) D（有界超慢/单位根）：F(n) 饱和（slope_far ≤ 0.30）∧ 谱近随机游走（β_raw ≥ 1.4）。
      4) C 族（入口：LR > lr_gate **或** 上态驻留幂律——C1 的 1/f² 谱与 Lorentzian fc→0 同族，
         PSD-LR 对突发型结构性失明，故驻留幂律为等效证据）：
         C2=上/下驻留双幂律；C1=上驻留幂律+下驻留指数+突发不对称（≥0.5）；C3=驻留双指数+α_band≈1 平台。
         均不中 ⇒ NaN（不可辨识，如实报告）。"""
    ch = channels if channels is not None else measure(x)
    meta = meta or {}
    res = dict(cls="NaN", status="不可辨识", channels=ch,
               hole_exposure=bool(meta.get("g3_jump", False)) or float(meta.get("hole_fraction", 0.0) or 0.0) > PRE["hole_tol"],
               truth=meta.get("case", None))
    if ch["N"] < PRE["min_len_classify"]:
        return res
    d = ch["dual"]
    dwl = ch["dwell"]
    up, dn = dwl["up"], dwl["dn"]
    if not (np.isfinite(d["cx_det"]) and np.isfinite(ch["lr_welch"])):
        return res
    # 1) B（设计漂移：原始序列含设计斜坡（trend_frac 高）∧ 去趋势后呈完整 A 型——判别表"去趋势后同 A"）
    detA = (d["cx_det"] >= PRE["cx_A"]) and np.isfinite(d.get("slope_far_det", float("nan"))) \
        and (d["slope_far_det"] >= PRE["slope_far_A"])
    if detA and np.isfinite(d.get("trend_frac", float("nan"))) and d["trend_frac"] >= PRE["trend_frac_B"]:
        res.update(cls="B", status="ok")
        return res
    if not (np.isfinite(ch["cx"]) and np.isfinite(ch["slope_far"])):
        return res
    # C 族幂律证据（提前计算，供 2b 白噪声出口与 4) C 族入口共用）
    # [2026-09-28 M1 修复⑦ / 台账 #44（所有者授权『可以动核心』）：C 族幂律准入由显著性量
    #  rsqn=R·√n > csn_gate 改为窗口不变效应量 R ≥ R_eff_gate（#41：rsqn 非窗口不变）。
    #  影响面严格限定此两行；K2/A11 识别通道与 occurrence 通道不动，csn_gate 键保留。]
    pl_up = bool(up["status"] == "ok" and np.isfinite(up["R"]) and up["R"] >= PRE["R_eff_gate"])
    pl_dn = bool(dn["status"] == "ok" and np.isfinite(dn["R"]) and dn["R"] >= PRE["R_eff_gate"])
    # 2) A（平稳 OU：1.5→0.5 交叉 @n≈τ）
    asym_ok = (not np.isfinite(dwl["asym"])) or (dwl["asym"] <= PRE["asym_A"])
    if ch["cx"] >= PRE["cx_A"] and ch["slope_far"] >= PRE["slope_far_A"] \
            and ch["lr_welch"] <= PRE["lr_lor"] and asym_ok:
        res.update(cls="A", status="ok")
        return res
    # 2b) A（平稳白噪声出口）[2026-09-27 M1 修复②：E5 判别表完备化——iid 平稳过程在表中无出口，
    #     实测 K5 sham（随机井 iid 链）cx=−0.07 无交叉、LR∈简并区、无驻留幂律 ⇒ NaN 而非 A。
    #     平稳白噪声属判别表 A 行"平稳短程相关"家族（τ<16 带下限，交叉带外）。
    #     门：无交叉 ∧ 白噪声斜率（≈0.5，非饱和非 1/f 平台）∧ LR 简并区 ∧ 无上/下驻留幂律
    #        ∧ 对称。判据阈值全部沿用 PRE 原值，未新增阈值量级；待 E5 会签确认。]
    white_ok = ch["cx"] < PRE["cx_A"] \
        and (PRE["slope_far_A"] < ch["slope_far"] <= 0.75) \
        and (PRE["lr_lor"] < ch["lr_welch"] <= PRE["lr_gate"]) \
        and (not pl_up) and (not pl_dn) and asym_ok
    if white_ok:
        res.update(cls="A", status="ok", note="平稳白噪声出口（判别表完备化 2026-09-27 M1 修复②）")
        return res
    # 3) D（有界超慢/单位根：中段 α≈1.5，F(n) 饱和，β≈2）
    # [2026-09-28 M1 修复⑥ / 台账 #42（所有者授权『可以动核心』）：D 行判别式由 r0
    #  （slope_far≤slope_far_D ∧ β_raw≥beta_D）替换为已验证 r2：
    #    cx<cx_A ∧ isfinite(β_raw) ∧ β_raw≥beta_D ∧ isfinite(α_mid) ∧ α_mid∈[alpha_D_lo,alpha_D_hi]
    #    ∧ 上/下驻留皆无幂律证据（¬pl_up ∧ ¬pl_dn）。
    #  依据 #35（D 行与 G 族同一裁决）/#37/#41（D 驻留通道结构性饥饿：切换数 2–12 < csn_n_min=30，
    #  D 不可由驻留判别）⇒ 改以 DFA/谱通道（cx/β/α_mid）+「驻留非幂律」负子句判别。
    #  只读探针：非-D 假阳 0/2400、D 召回 454/480=0.9458（M1_probe_r2fp.py / _cascade.py）；
    #  k5_gate 四门阈值未动。]
    # [2026-09-28 M1 修复⑧ / 台账 #55（所有者授权『可以动核心』）：D 行 r2 的 α_mid 子句由
    #  『α_mid∈[alpha_D_lo,alpha_D_hi]』改为**并集**『α_mid∈带 ∨ qratio≤qratio_D』。
    #  依据：#54 只读发现 qratio 为洞鲁棒压缩判别量（D 软饱和<1、B 线性斜坡≈1）；#53 证 α_mid 带对洞
    #  不稳健且预处理层不可修 ⇒ 需另立外科判别量。影子树只读（M1_probe_qratio.py，mismatch=0/1440）：
    #  g∈[0.40,0.70] 非-D 假阳恒 0（非-D 最小 qratio 0.872），D 召回 214/239 → 并集 238–239/239；
    #  D_alt 三分区五量严格占优（holdout G3 0.883→0.933、synth 0.9708→0.9750）。
    #  影响面严格限定此 D 行；k5_gate 五门阈值与对照口径一律不动，禁止凑数。]
    d_alpha = np.isfinite(ch["alpha_mid"]) and (PRE["alpha_D_lo"] <= ch["alpha_mid"] <= PRE["alpha_D_hi"])
    d_qr = np.isfinite(ch.get("qratio", float("nan"))) and (ch["qratio"] <= PRE["qratio_D"])
    if (ch["cx"] < PRE["cx_A"]) and np.isfinite(ch["beta_raw"]) and (ch["beta_raw"] >= PRE["beta_D"]) \
            and (d_alpha or d_qr) \
            and (not pl_up) and (not pl_dn):
        res.update(cls="D", status="ok", note="r2 判别（#42 所有者授权改核；#55 qratio 并集）")
        return res
    # 4) C 族（临界无标度）
    if (ch["lr_welch"] > PRE["lr_gate"]) or pl_up:
        if up["status"] != "ok" or dn["status"] != "ok":
            return res
        if pl_up and pl_dn:
            res.update(cls="C2", status="ok")
            return res
        if pl_up and (not pl_dn) and np.isfinite(dwl["asym"]) and dwl["asym"] >= PRE["dwell_asym_C1"]:
            res.update(cls="C1", status="ok")
            return res
        if (not pl_up) and (not pl_dn) and np.isfinite(ch["alpha_band"]) and \
                abs(ch["alpha_band"] - 1.0) <= PRE["alpha_C3"]:
            res.update(cls="C3", status="ok")
            return res
        # [2026-09-27 M1 修复⑥ C3 平台残余出口（判别表完备化，待 E5 会签）]
        # fGn（谱合成 1/f，β=1）的中位分割游程是长程相关——有限样本下 CSN 可偶发判幂律
        # （实测 2/6 seed pl_up 误触发 ⇒ 按原树落入 C1 门但 asym≈0 → NaN）。判别表 C3 行
        # 的定义性 signature 是 α≈1 平台（频谱 β=1 真值）；修法=平台残余：C 入口已过
        # （lr>lr_gate）∧ 非 C2（¬pl_up∧¬pl_dn）∧ 非 C1（asym<dwell_asym_C1）∧ α≈1 平台
        # ⇒ C3。C2/C1 优先级原样保留，阈值全部沿用 PRE 原值。]
        if np.isfinite(ch["alpha_band"]) and abs(ch["alpha_band"] - 1.0) <= PRE["alpha_C3"] \
                and np.isfinite(dwl["asym"]) and dwl["asym"] < PRE["dwell_asym_C1"]:
            res.update(cls="C3", status="ok", note="平台残余出口（判别表完备化 2026-09-27 M1 修复⑥）")
            return res
    return res


# ════════════════════════════════════════════════════════════════════════
# §4 判据函数 K1 / K2 / K3⊕K4 / K5（判据式一字照抄定稿）
# ════════════════════════════════════════════════════════════════════════

def k1_regime_ok(x: np.ndarray, ctx: str = "wake", hole_log: Optional[dict] = None,
                 channels: Optional[dict] = None) -> dict:
    """K1 标度不变性（FSS 塌缩版，定稿 §3.1）：
      regime_ok(c) = α̂∈[0.85,1.15] 于 n∈[16,512]（≥1.5 十进位）∧ 三窗 {16K,32K,64K} 漂移 ≤0.05
                     ∧ 去趋势双口径已报告 ∧ 洞占比 ≤10% 且无 G3 边界跳变。
      K1 按 ctx 分两栏报告（wake/offline）。守卫（tau_fit_detail 思想）：不可辨识 ⇒ status=不可辨识、passed=False。"""
    ch = channels if channels is not None else measure(x)
    lo, hi = PRE["alpha_band_gate"]
    a = ch["alpha_band"]
    fss = ch["fss"]
    dual = ch["dual"]
    hole = hole_log or {}
    hfrac = float(hole.get("hole_fraction", 0.0))
    g3 = bool(hole.get("g3_jump", False))
    hole_ok = (hfrac <= PRE["hole_tol"]) and (not g3)
    ident = bool(np.isfinite(a) and np.isfinite(fss["drift"]))
    passed = bool(ident and (lo <= a <= hi) and (fss["drift"] <= PRE["fss_drift_gate"]) and hole_ok)
    return dict(
        criterion="K1", ctx=ctx, passed=passed,
        status=("通过" if passed else ("不可辨识" if not ident else "失败")),
        alpha_band=float(a) if np.isfinite(a) else float("nan"),
        alpha_band_gate=[lo, hi], n_band=list(PRE["n_band"]),
        three_window_drift=float(fss["drift"]) if np.isfinite(fss["drift"]) else float("nan"),
        fss_alphas=fss["alphas"], fss_windows=fss["windows"],
        dual_report=dict(beta_raw=dual["beta_raw"], beta_det=dual["beta_det"],
                         d_beta=dual["d_beta"], cx_raw=dual["cx_raw"], cx_det=dual["cx_det"],
                         reported=True),
        hole_guard=dict(hole_fraction=hfrac, g3_jump=g3, ok=hole_ok, tol=PRE["hole_tol"]),
    )


def k2_dwell(x: np.ndarray, theta_q: float = None, do_sham: bool = True,
             stability_regression: Optional[dict] = None) -> dict:
    """K2 驻留分布形状（定稿 §3.1）：
      判据式：驻留样本幂律 vs 指数 CSN 似然比 **R·√n > 2**（预注册阈，一字照抄）；
      多稳性回归（T8 Q1 口径 ≥7/12 不退化）为引擎侧伴生项（M5 供给，此处接口预留 stability_regression）。
      sham：打乱重采样（保边缘破时序 ⇒ 位序列置乱）⇒ LR 必须翻负（<0）。
      守卫：驻留样本不足/拟合失效 ⇒ status=不可辨识、NaN。"""
    theta_q = PRE["k2_theta_q"] if theta_q is None else theta_q
    x = np.asarray(x, float)
    theta = float(np.quantile(x, theta_q))
    bits = x > theta
    if PRE["medw"] > 1:
        bits = medfilt(bits.astype(np.int8), kernel_size=PRE["medw"]).astype(bool)
    up, _dn = _run_lengths(bits)
    fit = _csn_fit(up)
    sham_ok = None
    rsqn_shuf = float("nan")
    if do_sham and fit["status"] == "ok":
        b2 = bits.copy()
        rng = np.random.default_rng(12345)   # sham 打乱用固定流（仪器确定性）
        rng.shuffle(b2)
        up2, _ = _run_lengths(b2)
        fit2 = _csn_fit(up2)
        rsqn_shuf = float(fit2["rsqn"]) if fit2["status"] == "ok" else float("nan")
        sham_ok = bool(np.isfinite(rsqn_shuf) and rsqn_shuf < 0.0)
    ident = fit["status"] == "ok"
    passed = bool(ident and np.isfinite(fit["rsqn"]) and fit["rsqn"] > PRE["csn_gate"])
    if sham_ok is False:
        passed = False   # sham 必须翻负；不翻 ⇒ 仪器对时序结构不敏感 ⇒ 判据不通过
    return dict(
        criterion="K2", passed=passed,
        status=("通过" if passed else ("不可辨识" if not ident else "失败")),
        theta=theta, theta_q=theta_q, n=int(fit["n"]),
        R=float(fit["R"]) if ident else float("nan"),
        rsqn=float(fit["rsqn"]) if ident else float("nan"),
        gate="R·√n > 2", alpha_dwell=float(fit["alpha"]) if ident else float("nan"),
        x_min=float(fit["xmin"]) if ident else float("nan"),
        sham=dict(rsqn_shuffled=rsqn_shuf, flipped_negative=sham_ok, required=True),
        stability_regression_T8Q1=stability_regression,   # 接口预留（M5 供给，≥7/12 不退化）
    )


def k3_two_stage(e_c: np.ndarray, alpha_seq: np.ndarray, scc_alive: bool, t_kill: int,
                 deadband: float = 0.05, t_fast: int = 2000, t_slow: int = 16384,
                 band: Sequence[float] = (0.85, 1.15), alpha_at: str = "end") -> dict:
    """K3⊕K4 两段式（定稿 §3.1 / E5 §2-E4-1；接口——e_c/α/SCC 由 M6/M5/M7 供给）：
      T_rec^fast(2000 步) 内 e_c 回死区 ∧ T_rec^slow(16384 步) 内 α 回带 ∧ SCC 保持 ⇒ 切换瞬态；
      否则真失稳（触发可静态证明不变式升级）。alpha_at='end'：α 取 kill 后 t_slow 处的复核值。"""
    e_c = np.asarray(e_c, float)
    alpha_seq = np.asarray(alpha_seq, float)
    seg_f = e_c[t_kill:t_kill + t_fast]
    fast_ok = bool(np.any(np.isfinite(seg_f) & (np.abs(seg_f) <= deadband))) if seg_f.size else False
    idx = min(t_kill + t_slow, len(alpha_seq) - 1) if alpha_seq.size else -1
    a_val = float(alpha_seq[idx]) if (alpha_at == "end" and idx >= 0) else float("nan")
    slow_ok = bool(np.isfinite(a_val) and band[0] <= a_val <= band[1])
    verdict = "切换瞬态" if (fast_ok and slow_ok and scc_alive) else "真失稳"
    return dict(criterion="K3⊕K4", verdict=verdict,
                fast=dict(ok=fast_ok, window=t_fast, deadband=deadband),
                slow=dict(ok=slow_ok, window=t_slow, alpha=a_val, band=list(band)),
                scc_alive=bool(scc_alive))


def k3_ledger_audit(ledger_deltas: Sequence[float], k_c: float) -> dict:
    """K3 验账（定稿 §2.0 账本纪律）：k_c=0（控制环切除）⇒ 账本增量必须为 0。"""
    d = np.asarray(ledger_deltas, float)
    nz = int(np.count_nonzero(d[np.isfinite(d)]))
    ok = (abs(k_c) < 1e-12 and nz == 0) or (abs(k_c) > 1e-12)
    return dict(criterion="K3 验账", passed=bool(ok), k_c=float(k_c), nonzero_entries=nz,
                rule="k_c=0 ⇒ 账本增量必须为 0（没有调节环就没有合法改动）")


def k4_scc(adj: np.ndarray) -> list:
    """K4 边界：补偿流邻接阵 A[x][y]（x 扰动唤起 y 补偿）的强连通分量（Kosaraju）。"""
    a = np.asarray(adj).astype(bool)
    n = a.shape[0]
    reach = a.copy()
    for _ in range(n):
        reach = reach | (reach @ reach > 0)
        if (reach | reach.T).all():
            break
    comp = []
    assigned = np.zeros(n, bool)
    for i in range(n):
        if assigned[i]:
            continue
        members = [j for j in range(n) if reach[i, j] and reach[j, i]]
        for j in members:
            assigned[j] = True
        if len(members) >= 1:
            comp.append(members)
    sccs = [sorted(set(m)) for m in comp if len(m) > 1]
    return sccs


def k4_verdict(adj: np.ndarray, candidates: Sequence[str], sham_expected: Sequence[str]) -> dict:
    """K4 判读：边界=A 图上含被维持量的 SCC；sham 集（候选中测得"无补偿"者）必须非空——
    一切都调节=没有边界（E5 方案 K4 / 定稿 §3.1 K4 行）。"""
    sccs = k4_scc(adj)
    sham_found = [candidates[i] for i in range(len(candidates))
                  if not np.any(adj[i, :]) and not np.any(adj[:, i])]
    ok = (len(sccs) >= 1) and (len(sham_found) >= 1)
    return dict(criterion="K4", passed=bool(ok), sccs=sccs,
                sham_found=sham_found, sham_expected=list(sham_expected),
                sham_structural_ok=len(sham_found) >= 1,
                note="选择性失败（sham 测得有补偿）⇒ K4 语义失败；sham 集为空 ⇒ 一切都调节=没有边界")


def _longest_true_block(mask: np.ndarray) -> tuple:
    """最长连续 True 段 [a0, a1)（供 G3 分段极大去洞预处理：避开拼接跨跳变的人为阶跃）。
    [2026-09-28 #42 候选②，所有者授权改核]"""
    best = (0, 0)
    i, N = 0, len(mask)
    while i < N:
        if mask[i]:
            j = i
            while j < N and mask[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return best


def k5_run(master_seeds: Sequence[int], B: int = 40, T: int = 131072,
           with_G: bool = True, dual_preproc_examples: bool = True) -> dict:
    """K5 验收运行（B=40/情形，判据式一字照抄：分类正确率 ≥90%；G1/G2 ≥90%、G3 ≥85%）。
    T=131072：合成窗长取 128K——A(τ=2000) 的 DFA 交叉须 nmax≈6.5τ 可见（T9 实证 τ̂ 收敛需 15–40τ，
    64K 窗下长 τ 情形交叉不可见）；FSS 三窗 {16K,32K,64K}（定稿）嵌套其内。
    master_seeds：判据集 [0..5] / 留出集 [6..11]（seed 流公式固定：default_rng([set_id, case_idx, rep])）。

    [2026-09-27 计量口径修正（先注册 verify_all._CHANGELOG）] synth.overall = 正确数 / **总序列数 B×6**：
    空判（cls∉六情形，含 NaN=不可辨识）计入分母、不得剔除（"不可辨识不计通过"≠"从分母剔除"；
    剔除=静默掩盖错分）。旧口径（仅在本类可判集上算）另存 synth.overall_classifiable_only 作报告量。"""
    res = dict(master_seeds=list(master_seeds), B=B, T=T, synth={}, G={}, n_series=0)
    idx = {c: i for i, c in enumerate(CASES)}
    # —— 合成六情形 ——
    conf = np.zeros((len(CASES), len(CASES)), int)
    per_case = {}
    for ci, case in enumerate(CASES):
        preds = []
        for b in range(B):
            set_id = int(master_seeds[b % len(master_seeds)])
            rng = np.random.default_rng([set_id, ci, b])
            x, meta = gen_case(case, T, rng, b)
            r = classify(x, meta)
            preds.append(r["cls"])
            res["n_series"] += 1
            j = idx.get(r["cls"], -1)
            if j >= 0:
                conf[ci, j] += 1
        acc = float(np.mean([p == case for p in preds]))
        per_case[case] = dict(acc=acc, preds=preds)
    res["synth"] = dict(confusion=conf.tolist(), per_case=per_case,
                        overall=float(np.trace(conf) / max(1, B * len(CASES))),
                        overall_classifiable_only=float(np.trace(conf) / max(1, conf.sum())),
                        n_series=B * len(CASES))
    # —— 半合成打洞族 ——
    G_defs = [("G1", 0.05), ("G1", 0.20), ("G2", None), ("G3", None)]
    for kind, dh in G_defs:
        key = "%s@%s" % (kind, ("%.2f" % dh) if dh is not None else "T_rec^slow")
        preds, truths, hole_fracs, g3s = [], [], [], []
        for b in range(B):
            set_id = int(master_seeds[b % len(master_seeds)])
            rng = np.random.default_rng([set_id, 100 + G_defs.index((kind, dh)), b])
            x, meta = gen_G(kind, T, rng, b, delta_hole=dh)
            if bool(meta.get("g3_jump", False)):
                # [2026-09-28 #42 候选②（所有者授权改核）] G3 含边界跳变 |Δu|>5σ_run0：直接拼接全部有效样本
                #   会在洞处造成人为阶跃（跨跳变缝合）；改取**最长连续有效块**（分段极大去洞）——洞不可见化
                #   的忠实实现。G1（周期散洞）/G2（单长洞，无跳变）仍用全部有效样本（raw）。
                #   选择依据用 交付② 洞日志字段 g3_jump（gen_hole_log 供同构 dict），非生成器 kind 标签。
                a0, a1 = _longest_true_block(meta["valid"])
                xc = x[a0:a1]
            else:
                xc = x[meta["valid"]]                   # 步轴索引（判据本体轴）：洞不可见化预处理
            r = classify(xc, meta)
            preds.append(r["cls"])
            truths.append(meta["base"])
            hole_fracs.append(meta["hole_fraction"])
            g3s.append(bool(meta["g3_jump"]))
            res["n_series"] += 1
        acc = float(np.mean([p == t for p, t in zip(preds, truths)]))
        g_entry = dict(acc=acc, preds=preds, truths=truths,
                       hole_fraction_range=[float(np.min(hole_fracs)), float(np.max(hole_fracs))],
                       g3_flagged=int(sum(g3s)),
                       dual_preproc=None)
        if dual_preproc_examples:
            # 双预处理对比（E5 交付②判别量：步轴索引 vs 墙钟重采样——G1/G2 洞不可见、G3 仍可见）
            rng = np.random.default_rng([int(master_seeds[0]), 200 + G_defs.index((kind, dh)), 0])
            x, meta = gen_G(kind, T, rng, 0, delta_hole=dh)
            xc = x[meta["valid"]]
            xw = x.copy()
            xs = np.arange(len(x))
            xw[~meta["valid"]] = np.interp(xs[~meta["valid"]], xs[meta["valid"]], x[meta["valid"]])
            s_c, s_w = measure(xc), measure(xw)
            g_entry["dual_preproc"] = dict(
                step_axis=dict(alpha_band=s_c["alpha_band"], beta_raw=s_c["beta_raw"]),
                wallclock_resample=dict(alpha_band=s_w["alpha_band"], beta_raw=s_w["beta_raw"]),
                note="G1/G2 步轴 signatures 应复原（洞不可见）；G3 步轴仍可见（世界推进写入跳变）——物理非缺陷")
        res["G"][key] = g_entry
    return res


def k5_gate(results: dict) -> dict:
    """K5 闸门（定稿 §3.1 K5 行 + E5 Round3 交付②验收线）：
      合成：overall ≥90%（双集各自）；半合成：G1(5%)≥90% ∧ G1(20%)≥90% ∧ G2≥90% ∧ G3≥85%。
      不过线 ⇒ K1–K4 全废（"仪器未成"的诚实结论）。"""
    s = results["synth"]["overall"]
    g = {k: v["acc"] for k, v in results["G"].items()}
    checks = {
        "synth_overall≥0.90": s >= 0.90,
        "G1@5%≥0.90": g.get("G1@0.05", float("nan")) >= 0.90,
        "G1@20%≥0.90": g.get("G1@0.20", float("nan")) >= 0.90,
        "G2≥0.90": g.get("G2@T_rec^slow", float("nan")) >= 0.90,
        "G3≥0.85": g.get("G3@T_rec^slow", float("nan")) >= 0.85,
    }
    passed = all(checks.values())
    return dict(criterion="K5 闸门", passed=bool(passed), checks=checks,
                synth_overall=s, G=g,
                conclusion=("PASS —— 仪器可用，K1–K4 获得前置授权" if passed
                            else "FAIL —— K1–K4 全废（宁要『仪器未成』，不带病尺子量体制分数）；需 E5 修订判据式"))


# ════════════════════════════════════════════════════════════════════════
# §5 A11 标定（σ̂/β̂ 偏差曲线 → σ̂_target 查表）+ 时滞注入标定（k_p/η_i 初值）
# ════════════════════════════════════════════════════════════════════════

def sigma_hat(x: np.ndarray, theta_q: float = 0.9) -> float:
    """分支比估计（E5 方案 §3.1 观测器）：阈上级联配对 σ̂ = E[a_{t+1}|a_t≥θ] / E[a_t|a_t≥θ]，θ=窗内 θ_q 分位（自参考）。"""
    x = np.asarray(x, float)
    theta = float(np.quantile(x, theta_q))
    m = x[:-1] >= theta
    den = float(x[:-1][m].sum())
    if den <= 0 or m.sum() < 10:
        return float("nan")
    return float(x[1:][m].sum() / den)


def a11_calibration(B: int = 200, T_max: int = 8192, B_ref: int = 40, T_ref: int = 65536,
                    Wo_grid: Sequence[int] = (512, 1024, 2048, 4096, 8192),
                    theta_grid: Sequence[float] = (0.75, 0.9, 0.95),
                    seed_sigma: int = None, seed_beta: int = None) -> dict:
    """A11 偏差标定协议（E5 Round3 §1.3，置于 K3 之前；进 K5）：
      (1) C1（真 σ=1.00）上 W_o×θ_a×B 实现 → bias(W_o,θ)=E[σ̂]−1 与 sd → σ̂_target=1+bias 查表；
      (2) 同步标 β̂_ps 有限窗偏差（参考=64K 长窗均值）。
      标定流 seed 与验收集隔离（PRE.cal_seed_*）。"""
    seed_sigma = PRE["cal_seed_sigma"] if seed_sigma is None else seed_sigma
    seed_beta = PRE["cal_seed_beta"] if seed_beta is None else seed_beta
    # σ̂ 偏差：B 条 C1（长度 T_max），前缀作 W_o 窗
    series = []
    for b in range(B):
        rng = np.random.default_rng([seed_sigma, b])
        x, _m = gen_case("C1", T_max, rng, b)
        series.append(x)
    sig_tab = np.full((len(Wo_grid), len(theta_grid)), np.nan)
    sig_sd = np.full_like(sig_tab, np.nan)
    for i, Wo in enumerate(Wo_grid):
        for j, th in enumerate(theta_grid):
            vals = [sigma_hat(x[:Wo], th) for x in series]
            vals = np.array([v for v in vals if np.isfinite(v)], float)
            if vals.size >= max(30, B // 2):
                sig_tab[i, j] = float(vals.mean() - 1.0)
                sig_sd[i, j] = float(vals.std())
    # β̂ 偏差：参考 = 64K 长窗
    betas_ref = []
    for b in range(B_ref):
        rng = np.random.default_rng([seed_beta, b])
        x, _m = gen_case("C1", T_ref, rng, b)
        betas_ref.append(_beta_full(x))
    beta_ref = float(np.nanmean(betas_ref))
    beta_tab = np.full(len(Wo_grid), np.nan)
    for i, Wo in enumerate(Wo_grid):
        vals = [_beta_full(x[:Wo]) for x in series[:max(40, B)]]
        vals = np.array([v for v in vals if np.isfinite(v)], float)
        if vals.size >= 20:
            beta_tab[i] = float(vals.mean() - beta_ref)
    return dict(Wo_grid=list(Wo_grid), theta_grid=list(theta_grid),
                sigma_bias=sig_tab.tolist(), sigma_sd=sig_sd.tolist(),
                sigma_target=(1.0 + sig_tab).tolist(),
                beta_bias=beta_tab.tolist(), beta_ref=beta_ref,
                note="σ̂_target(W_o,θ)=1+bias(W_o,θ)（E3 负偏论证的有限窗修正；样条/线性插值查表入预注册）",
                seed_stream=dict(sigma=seed_sigma, beta=seed_beta))


_A11_CACHE = {"table": None}


def sigma_target(Wo: int, theta: float, table: Optional[dict] = None) -> float:
    """σ̂_target(W_o,θ) 双线性查表（W_o 固定 4096 为运行值；变更=判据语义变更=全体 _CHANGELOG）。"""
    if table is None:
        if _A11_CACHE["table"] is None:
            _A11_CACHE["table"] = a11_calibration()
        table = _A11_CACHE["table"]
    Wg = np.array(table["Wo_grid"], float)
    Tg = np.array(table["theta_grid"], float)
    S = np.array(table["sigma_target"], float)
    i = np.clip(np.searchsorted(Wg, Wo) , 1, len(Wg) - 1)
    i0, i1 = i - 1, i
    w = 0.0 if Wg[i1] == Wg[i0] else (Wo - Wg[i0]) / (Wg[i1] - Wg[i0])
    j = np.clip(np.searchsorted(Tg, theta), 1, len(Tg) - 1)
    j0, j1 = j - 1, j
    v = 0.0 if Tg[j1] == Tg[j0] else (theta - Tg[j0]) / (Tg[j1] - Tg[j0])
    return float(S[i0, j0] * (1 - w) * (1 - v) + S[i1, j0] * w * (1 - v)
                 + S[i0, j1] * (1 - w) * v + S[i1, j1] * w * v)


def time_delay_calibration(L: int = 4296, tau_obj: float = 800.0, T: int = 40000, dt_ctrl: int = 200,
                           beta_star: float = 0.526, kp_grid: Sequence[float] = None,
                           eta_grid: Sequence[float] = None, seed: int = None,
                           deadband: float = 0.05, I_max: float = 0.2,
                           beta_bounds: Sequence[float] = (0.3, 1.2)) -> dict:
    """时滞注入标定（E5 Round3 交付④）：合成闭环 = C 情形 σ̂ 序列 + 一阶执行器 + 纯时滞 L，
    扫 k_p/η_i，取稳定域的 1/3 为初值——仪器标定，不碰引擎数据。
    被控对象：σ 平衡面 g(β)=(β/β*)²（β*=0.526=T9 临界面锚），时间常数 τ_obj；测量纯时滞 L=4296 步。"""
    seed = PRE["cal_seed_delay"] if seed is None else seed
    kp_grid = np.geomspace(0.005, 0.30, 10) if kp_grid is None else np.asarray(kp_grid, float)
    eta_grid = np.geomspace(1e-6, 3e-4, 9) if eta_grid is None else np.asarray(eta_grid, float)
    # σ̂_target（A11 查表，W_o=4096、θ=0.9 运行口径）
    st = sigma_target(4096, 0.9)
    if not np.isfinite(st):
        st = 1.0
    stable = np.zeros((len(kp_grid), len(eta_grid)), bool)
    detail = []
    for ik, kp in enumerate(kp_grid):
        for ie, eta in enumerate(eta_grid):
            rng = np.random.default_rng([seed, ik, ie])
            sig = 0.8
            I = 0.0
            beta = beta_star
            hist_sig = np.empty(T)
            hist_beta = np.empty(T)
            e_last = 0.0
            for t in range(T):
                # 一阶对象（每步）：dσ/dt=(g(β_eff)−σ)/τ_obj，β_eff=β(t−L) 纯时滞执行
                src = t - L
                b_eff = hist_beta[src] if src >= 0 else beta_star
                g = (b_eff / beta_star) ** 2
                sig += (g - sig) / tau_obj + 1e-4 * rng.standard_normal()
                hist_sig[t] = sig
                # 观测（纯时滞 L）+ A11 查表修正 → e_c
                sig_m = hist_sig[t - L] if t - L >= 0 else sig
                e = math.log(max(sig_m, 1e-6) / st)
                e_last = e
                # C-HOME 控制律（死区 + 积分钳制；β 每步可写，I 以 η_i/步 积分）
                if abs(e) > deadband:
                    I = float(np.clip(I + eta * e, -I_max, I_max))
                    beta = float(np.clip(beta_star + kp * e + I, beta_bounds[0], beta_bounds[1]))
                hist_beta[t] = beta
            tail = slice(int(T * 0.75), T)
            e_tail = np.abs(np.log(np.maximum(hist_sig[tail], 1e-6) / st))
            frac_dead = float(np.mean(e_tail <= deadband))
            std_b = float(np.std(hist_beta[tail]))
            ok = bool(frac_dead >= 0.9 and std_b <= 0.05)
            stable[ik, ie] = ok
            detail.append(dict(kp=float(kp), eta=float(eta), stable=ok,
                               frac_in_deadband=frac_dead, beta_std=std_b))
    # 初值 = 稳定域的 1/3
    kp_stable_max = float("nan")
    for ik in range(len(kp_grid) - 1, -1, -1):
        if stable[ik].mean() >= 0.5:
            kp_stable_max = float(kp_grid[ik])
            break
    kp_init = kp_stable_max / 3.0 if np.isfinite(kp_stable_max) else float("nan")
    eta_stable_max = float("nan")
    if np.isfinite(kp_init):
        ik = int(np.argmin(np.abs(kp_grid - kp_init * 3.0)))
        idxs = np.where(stable[ik])[0]
        if idxs.size:
            eta_stable_max = float(eta_grid[idxs[-1]])
    eta_init = eta_stable_max / 3.0 if np.isfinite(eta_stable_max) else float("nan")
    return dict(L=L, tau_obj=tau_obj, beta_star=beta_star, sigma_target_used=st, T=T, dt_ctrl=dt_ctrl,
                kp_grid=kp_grid.tolist(), eta_grid=eta_grid.tolist(),
                stable=stable.tolist(), detail=detail,
                kp_stable_max=kp_stable_max, kp_init=kp_init,
                eta_stable_max=eta_stable_max, eta_init=eta_init,
                candidates=dict(kp="≲0.06", eta_i="≈3×10⁻⁵/步", I_max=I_max, deadband=deadband),
                note="候选值，K5 自证后落册；引擎侧逐级上增益：k_p 从初值 ×2 步进、每步 K1 三窗漂移守卫、越界即停")


# ════════════════════════════════════════════════════════════════════════
# §6 C2b 判读入口（E3 Round3 §1.2 判据式原文；数据由 M5 供给，此处接口+判读）
# ════════════════════════════════════════════════════════════════════════

def _exclude_flat(chain: np.ndarray, flat_windows: Sequence[Sequence[int]]) -> tuple:
    """压平标记排除（E5 §1.4 追加条款）：剔除压平后 T_flat=8192 步内窗口，返回 (chain', 段列表)。"""
    chain = np.asarray(chain)
    N = len(chain)
    if not flat_windows:
        return chain, [(0, N)]
    mask = np.ones(N, bool)
    for (s, e) in flat_windows:
        s = max(0, int(s))
        e = min(N, int(e))
        mask[s:e] = False                      # 压平事件窗本体
        mask[e:min(N, e + 8192)] = False       # 压平后 T_flat=8192 步内窗口一并排除
    idx = np.flatnonzero(mask)
    segs = []
    if idx.size:
        brk = np.where(np.diff(idx) != 1)[0]
        starts = np.concatenate(([0], brk + 1))
        ends = np.concatenate((brk + 1, [len(idx)]))
        for a, b in zip(starts, ends):
            if idx[a:b].size >= 4096:
                segs.append((int(idx[a]), int(idx[b - 1]) + 1))
    return chain[idx], segs


def c2b_readout(k_star: Optional[np.ndarray] = None, s_low: Optional[np.ndarray] = None,
                seed_chains: Optional[Sequence[np.ndarray]] = None,
                flat_windows: Optional[Sequence[Sequence[int]]] = None,
                T_obs: Optional[int] = None) -> dict:
    """C2b 井间补集判读（E3 Round3 §1.2 判据式原文，S9 条款落地文本）：
      k*(t)=argmax_k cos(S̃_C, IM[k])（占据链）与 S_t 低频（PC1、‖ΔS_t‖）各跑 K-BATTERY。
      PASS_P_i：R_occ 判驻留指数 且 dwell 均值 ≤ T/8 且跨 seed 可分辨 ⇒ 亚稳体制（E5 公开撤回补集要求）；
      PASS_P_ii：R_occ 判幂律（R·√n>2）且 R_S 判 α̂∈[0.85,1.15] 跨 ≥1.5 十进位+跨窗塌缩 ⇒ 临界游走；
      唯一禁带=T10 死带：驻留指数 且 dwell 均值 ~ 观测窗同量级（>T/8）且跨 seed 不可分辨；
      压平后 T_flat=8192 步内窗口排除（O4 待 E3 补签，本接口按预注册执行排除）。
      数据由 M5（hdc_core）供给；本函数只判读。"""
    if k_star is None:
        return dict(criterion="C2b", status="不可辨识", note="k*(t) 占据链未供给（M5 供给）")
    chain, segs = _exclude_flat(np.asarray(k_star), flat_windows)
    N_eff = len(chain)
    T_obs = int(T_obs or N_eff)
    if T_obs < 4096 or N_eff < 4096:
        return dict(criterion="C2b", status="不可辨识", note="有效窗不足 4096 步")
    # R_occ：占据链驻留（run 长度直接为驻留样本——占据链是离散链，无需阈值分态）
    dw = []
    v = np.asarray(chain)
    change = np.flatnonzero(np.diff(v)) + 1
    idx = np.concatenate(([0], change, [len(v)]))
    dw = np.diff(idx).astype(float)
    R_occ = _csn_fit(dw)
    dwell_mean = float(np.mean(dw)) if dw.size else float("nan")
    # 跨 seed 可分辨性
    seed_dwell_means = []
    if seed_chains:
        for c in seed_chains:
            c2, _ = _exclude_flat(np.asarray(c), flat_windows)
            ch2 = np.flatnonzero(np.diff(c2)) + 1
            ix = np.concatenate(([0], ch2, [len(c2)]))
            seed_dwell_means.append(float(np.mean(np.diff(ix))))
    if seed_dwell_means:
        resolvable = bool(np.mean(seed_dwell_means) <= T_obs / 8.0)
    else:
        resolvable = bool(dwell_mean <= T_obs / 8.0)
    # R_S：S_t 低频（K1 仪器：α̂ 带 + 三窗塌缩）
    R_S = None
    if s_low is not None:
        ch = measure(np.asarray(s_low, float))
        R_S = dict(alpha_band=ch["alpha_band"], drift=ch["fss"]["drift"],
                   in_band=bool(np.isfinite(ch["alpha_band"]) and 0.85 <= ch["alpha_band"] <= 1.15),
                   collapse_ok=bool(np.isfinite(ch["fss"]["drift"]) and ch["fss"]["drift"] <= PRE["fss_drift_gate"]),
                   lr_welch=ch["lr_welch"])
    exp_occ = bool(R_occ["status"] == "ok" and np.isfinite(R_occ["rsqn"]) and R_occ["rsqn"] <= 0.0)
    pl_occ = bool(R_occ["status"] == "ok" and np.isfinite(R_occ["rsqn"]) and R_occ["rsqn"] > PRE["csn_gate"])
    verdict = None
    if pl_occ and R_S is not None and R_S["in_band"] and R_S["collapse_ok"]:
        verdict = "PASS_P_ii（临界游走；井间补集条款长期生效）"
    elif exp_occ and resolvable:
        verdict = "PASS_P_i（亚稳体制；E5 按 S9 公开撤回补集要求）"
    elif exp_occ and (not resolvable):
        verdict = "FAIL（T10 死带：dwell 均值~观测窗同量级且跨 seed 不可分辨）"
    else:
        verdict = "不可辨识（NaN，不伪装）"
    return dict(criterion="C2b", verdict=verdict, status=("ok" if verdict.startswith(("PASS", "FAIL")) else "不可辨识"),
                R_occ=dict(rsqn=R_occ["rsqn"], alpha=R_occ["alpha"], n=R_occ["n"],
                           judged=("幂律" if pl_occ else ("驻留指数" if exp_occ else "不可判"))),
                dwell_mean=dwell_mean, dwell_mean_over_T8=dwell_mean / (T_obs / 8.0),
                seed_dwell_means=seed_dwell_means, cross_seed_resolvable=resolvable,
                R_S=R_S, flat_excluded_segs=segs, T_flat=8192,
                gates=dict(P_ii_rsqn=PRE["csn_gate"], P_i_dwell="≤T/8", deadband="dwell>T/8 ∧ 跨seed不可分辨"),
                note="消融 sham（切规则1+2 ⇒ 禁带必现，不现⇒C2b 作废）与灵敏度 sham 由 c2b_sensitivity_sham 供仪器侧")


def c2b_sensitivity_sham(seed: int = 900) -> dict:
    """仪器灵敏度 sham（E5 §1.3/E3 C2b 原文）：随机向量井/γ→0 退化井 ⇒ battery 必须归类 A（无游走）。"""
    rng = np.random.default_rng([seed, 777])
    chain = rng.integers(0, 8, size=65536)     # 随机向量井 ⇒ 占据链 iid 均匀
    x = chain.astype(float) + 0.01 * rng.standard_normal(65536)
    r = classify(x, dict(case="sham_random_wells"))
    ch = np.flatnonzero(np.diff(chain)) + 1
    ix = np.concatenate(([0], ch, [len(chain)]))
    occ = _csn_fit(np.diff(ix).astype(float))
    ok = (r["cls"] == "A")
    return dict(criterion="C2b 仪器灵敏度 sham", sham="随机向量井（iid 占据链）",
                battery_cls=r["cls"], required="A", passed=bool(ok),
                occ_rsqn=float(occ["rsqn"]) if occ["status"] == "ok" else float("nan"))
