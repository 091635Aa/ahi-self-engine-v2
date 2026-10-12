# -*- coding: utf-8 -*-
"""ahi_theory/g_layer.py —— M6 · G-Layer 治理层（影子模式）：C-HOME v2 情境门控 + 守门信号族 + K3⊕K4 两段式
    + 离线相起飞门 + β_slow 信号（E5 G-Layer v2 终稿实装）

预登记：ahi_theory/verify_all.py `_CHANGELOG` 2026-09-30『M6 G-Layer 治理层预登记』（先于改码）。
规格出处：meeting11/95_架构定稿.md L2 治理层（行 129–194）+ §2.0 全局合同（行 63–70）+
§3.1 K1/K3⊕K4/K3′ 判据式（行 377/379/380）+ L2 参数表（行 164–194）+ 蓝图 §5-M6（行 468）。

影子模式纪律（红线）：
  · 纯新增；默认不加载（无模块级副作用、无内核接线——消费方显式 import 才实例化）；
  · 不 import brain/world/system/three_loop/mbp（影子纪律）；只读 import ahi_theory.k_battery
    （K1/DFA 仪器、六情形合成器、A11 σ̂_target 查表——定稿 L2"regime_ok 复用 K-BATTERY 的 DFA 估计器"，
    k_battery.k1_regime_ok/k4_scc/dfa_summary 均为公开仪器函数，直接只读复用不重实现）；
    ahi_theory.ledger 仅惰性 import（变更清单预授权扩充路径，ledger.py 源文件零改动——personality/selfthink 同律）；
  · 零拟合红线：全文件无 lstsq / linalg.inv·pinv / linalg.solve / scipy.optimize / 参数搜索构造
    （C4 审计器 tokenize 级可检，G6_selftest [2] 自证）；σ̂=条件中位数配对比（预注册统计量，非拟合）；
  · 账本纪律：预注册常数构造时以 kind="参数变更" 入账本（键前缀 "G层_"，候选/定稿身份逐条标注；
    k_p=0.06/η_i=3e-5 为 K5 时滞标定落定前的候选值）；控制环 authored 的账本写入仅在 k_c>0 时发生
    （k_c=0 切除 ⇒ 控制律旁路+控制环零账本写入——K3 验账口径；监测/审计通道事件不受此限）；
  · β 输出语义：只产出 β_slow 信号（三态龄期 5s 语义：新鲜/陈旧/出厂默认），不回写 brain——
    β_slow 状态化按 D-1 缓行（BREAKING 面），输出信号供三环消费、影子期无人消费（诚实边界）；
  · 诚实边界：本模块全部为影子期机制自证；K1–K4 引擎级测量、K3′ 单带 sham 表型、O1/O2/O3/O6/O8/O10
    开放项数值待 V2 引擎供数与所有者/E5 裁决后另行预注册接入，不虚设。

机制总览（95_架构定稿 行 133–162 逐条落实，判据式一字照抄）：
  前置链: K5 六情形自证 ≥90%【已过线：meeting12 #42–#57 三分区×双窗×六块五门全过 ⇒ 前置满足标注】
    → σ̂/β̂ 偏差标定（A11 查表，经 k_battery.sigma_target(W_o=4096, θ_a=0.9)）→ 时滞注入标定（k_p/η_i=候选）。
  C-HOME v2 情境门控:
    上下文 c_t∈{wake,offline} 双源合成 =
      ACh 源（影子代理=EMA(pe) 供数接口；ACh/median_wake(ACh)<0.5 持续≥100 R2 拍⇒offline-like，
      L3 §1-A4 原文；基线=最近 65536 个 wake 共识步 ACh 样本中位、≥512 样本方可用，未热⇒无降判据证据）
      ⊕ offline 旗标源（供数接口；滞回：旗标稳定≥5 R2 拍方接受切换）；
    冲突或 ACh 缺失⇒context_ambiguous⇒调节冻结至消歧；失配>500 R2 拍⇒账本登记。
    每 R_reg=200 步一观测拍（分域观测，域切换即断窗重开）：
      σ̂_c = 条件中位数配对比 ÷ σ̂_target(W_o)；e_c = log(σ̂_c)；
      wake 调节带: |e_c|>δ_dead=0.05 ⇒ I_c^wake←clip(I_c^wake+η_i·R_reg·e_c, ±I_max=0.2)
                                        β_slow←clip(β_base+k_p·e_c+I_c^wake, [0.3,1.2])（死区内不写）；
      offline 监测带: I_c^off 冻结、β 保持、越带[0.85,1.00]只报警（post_shy_audit 通道）——
        offline_steering=observe_only 预注册默认（active_band 须独立执行器论证=O2 开放，未实装）；
    切换协议: (i) 离开域 I_c 快照+(t,e_c) 入账本 (ii) 进入域 I_c 从快照续（不清零）(iii) 首拍只观测不写 β；
    积分口径注：η_i≈3e-5/步（E5 交付④单位=每步），每 R_reg 拍执行一次观测与更新，
    积分增量=η_i·R_reg·e_c（步率×拍长；与 k_battery.time_delay_calibration 的逐步积分同动力学口径）。
    影子步轴换算：1 R2 拍=N_conv=200 步[候选，O8 开放] ⇒ 持续/滞回/登记阈 100/5/500 R2 拍 = 100/5/500 观测拍，
    β 龄期上限 5s=25 R2 拍=5000 步。
  守门信号族:
    regime_ok(c)  = k_battery.k1_regime_ok 一字照抄：α̂∈[0.85,1.15] 于 n∈[16,512]（≥1.5 十进位）
                    ∧ 三窗 {16K,32K,64K} 漂移 ≤0.05 ∧ 去趋势双口径已报告 ∧ 洞占比 ≤10% 且无 G3 跳变；
    scc_state(t)  = {size, closed?, members_changed?, event∈{转换|死亡|失败}}（K4 边界=补偿图强连通分量，
                    复用 k_battery.k4_scc；size=最大 SCC 节点数（≥2 节点口径，k4_scc 原样））；
    recovery_ok   = K3⊕K4 两段式：T_rec^fast=2000 步内 e_c 回死区（快通道=W_fast=2048 快子窗 σ̂ 快回判读，
                    k_battery 双轴表原文；主通道兜底）∧ T_rec^slow=16384 步内 α 回带（体制环末 ≤64K 的
                    DFA alpha_band 复核）∧ SCC 保持 ⇒ 切换瞬态；否则真失稳；kill 注入接口=注入kill()；
                    恢复确认（切换瞬态）为控制环 authored 账本写入（仅 k_c>0）；
    windup_guard  = wake→offline→wake ×N_sw=10，每次回 wake 后 |ΔI_c^wake|≤δ_Ic=0.02 且回 wake 首拍
                    e_c∈死区；任一失败⇒windup_event 入账本（记录时即时登记）、K3′ FAIL；
                    记录不足 N_sw⇒不可辨识（不伪装）；
    post_shy_audit= K-lite（DFA α̂+谱指数 β̂）on C 驱动活性序列（供数接口）；α̂∈[0.85,1.15]；
                    连败 3 次⇒离线相暂停+账本（谱指数只报告不判门；连败数在通过时清零并解除暂停）；
    post_replay_quiet = 距最近离线相结束 ≥T_q=6144 步 ⇒ true（E1 T_rev 相位条件；
                    从未有离线相⇒true=无静默要求）；
    consolidation_permit = regime_ok(wake) ∧ ¬stale_window（stale=连续 ≥3 喂步缺帧，L1/E3 丢帧语义；
                    regime_ok(wake) 取最近一次 wake 判定缓存，从未测量⇒不放行）。
  离线相起飞门: E5 版=「regime_ok(wake) 为真才允许进入离线相」（S 压力接口，影子期代理供数）；
    降级条款预注册：offline 域 K1/K2 不可辨识 ⇒ wake 域 regime_ok ∧ offline 域无 SCC 崩解；
    E1 饥饿升级（S>3·S_crit 强制离线相）=O3 开放，未实装（诚实边界）。

WCET 合同：每 R_reg 观测拍 ≤2.0ms（σ̂ 主窗+快子窗估计+双源合成；G6_selftest [10] 实测）；β 信号产出 O(1)。
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
from ahi_theory import k_battery as _kb

__all__ = [
    "G层常量", "G层核心", "β限幅", "条件中位数配对比",
    "R_reg", "W_o", "W_fast", "δ_dead", "I_max", "β_base", "β带", "k_p", "η_i",
    "T_rec快", "T_rec慢", "N_sw", "δ_Ic", "T_q", "α带", "尺度带", "三窗", "三窗漂移阈",
    "洞容忍", "wake带", "offline带", "offline_steering默认", "post_shy连败阈",
    "β龄期上限步", "stale拍阈", "ACh比阈", "N_conv", "注册键数",
]

# ════════════════════════════════════════════════════════════════════════
# §0 预注册常量（95_架构定稿 L2 参数表；变更 = kind="参数变更" 账本结构事件，键前缀 "G层_"）
# ════════════════════════════════════════════════════════════════════════
R_reg = 200                    # 观测节拍（E5 §1.1；E2 A8 表态=结构常量）【定稿】
W_o = 4096                     # 观测窗（变更=判据语义变更=全体 _CHANGELOG）【定稿】
W_fast = 2048                  # 快通道子窗（σ̂ 快回判读）【定稿】
δ_dead = 0.05                  # 死区（E5 §1.2）【候选】
I_max = 0.2                    # 积分钳制 ±I_max（E5 §1.2）【候选】
β_base = 0.526                 # β_base（T9 临界面锚，与 k_battery.time_delay_calibration 同源）【候选】
β带 = (0.3, 1.2)               # β_slow 输出带（E5 §1.2）【候选】
k_p = 0.06                     # 比例增益（E5 交付④）【候选，K5 时滞标定落定前】
η_i = 3e-5                     # 积分增益/步（E5 交付④）【候选，K5 时滞标定落定前】
T_rec快 = 2000                 # K3 快通道（S10 联署）【定稿】
T_rec慢 = 16384                # K3 慢通道【定稿】
N_sw = 10                      # windup 循环数（E5 §1.2）【定稿】
δ_Ic = 0.02                    # windup 守卫（=0.1×I_max，E1 同值）【定稿】
T_q = 6144                     # 复核静默窗（E5；与 E1 500 R2 拍折算对齐=O6 开放）【候选】
α带 = (0.85, 1.15)             # K1 α 带【定稿】
尺度带 = (16, 512)             # K1 尺度带（≥1.5 十进位）【定稿】
三窗 = (16384, 32768, 65536)   # FSS 三窗【定稿】
三窗漂移阈 = 0.05              # 三窗漂移容差（E5 §2）【候选】
洞容忍 = 0.10                  # 洞占比 ≤10% 且无 G3 跳变⇒作数（E5 交付②；E4 >20%⊥ 合并=O10）【候选】
wake带 = (0.95, 1.10)          # σ̂ 轴 wake 带（E1 §1-A4）【定稿】
offline带 = (0.85, 1.00)       # σ̂ 轴 offline 带【定稿】
offline_steering默认 = "observe_only"   # 预注册开关（active_band=O2 开放，未实装）
post_shy连败阈 = 3             # K-lite 连败 3 次⇒离线相暂停【定稿】
N_conv = 200                   # 影子步轴换算 1 R2 拍=N_conv 步【候选，O8 开放】
β龄期上限步 = 25 * N_conv      # 产出龄期上限 5s=25 R2 拍 ⇒ 5000 步【候选（随 N_conv）】
stale拍阈 = 3                  # 连续 ≥3 喂步缺帧⇒stale_window（L1/E3 丢帧语义）【定稿】
ACh比阈 = 0.5                  # ACh/median_wake(ACh) < 0.5⇒降判据（L3 §1-A4）【定稿】
ACh持续R2拍 = 100              # 持续 ≥100 R2 拍⇒offline-like【定稿（拍数）；步轴换算候选】
旗标滞回R2拍 = 5               # 旗标稳定 ≥5 R2 拍方接受切换【定稿（拍数）】
失配登记R2拍 = 500             # 失配 >500 R2 拍⇒账本登记【定稿（拍数）】
ACh_EMA_α = 0.001              # ACh=EMA(pe) 的 α=1/τ，τ_ACh=1000 步（L3 未定 τ，实装预注册初值）【候选】
θ_a = 0.9                      # σ̂ 条件分位自参考（A11 标定口径）【定稿】
体制环容量 = 65536             # 体制观测环（=三窗最大窗；K3 慢通道 α 复核载体）
基线环容量 = 65536             # ACh wake 基线环（步轴；≥512 样本方可用）
ACh基线最少样本 = 512

#: 派生换算（R2 拍 → 观测拍，随 N_conv 候选）
ACh持续拍 = ACh持续R2拍 * N_conv // R_reg      # =100 观测拍
旗标滞回拍 = 旗标滞回R2拍 * N_conv // R_reg    # =5 观测拍
失配登记拍 = 失配登记R2拍 * N_conv // R_reg    # =500 观测拍


def _注册键值表():
    """预注册常数表（构造时逐条 kind="参数变更" 入账本；变更须再次结构事件）。"""
    return (
        ("G层_R_reg", R_reg, "定稿",
         "M6 #71 预注册：观测节拍 200 步（95_架构定稿 L2 参数表；E2 A8 表态=结构常量）"),
        ("G层_W_o", W_o, "定稿",
         "M6 #71 预注册：观测窗 4096 步（变更=判据语义变更=全体 _CHANGELOG）"),
        ("G层_W_fast", W_fast, "定稿",
         "M6 #71 预注册：快通道子窗 2048 步（σ̂ 快回判读，k_battery 双轴表同源）"),
        ("G层_δ_dead", δ_dead, "候选",
         "M6 #71 预注册：死区 0.05（E5 §1.2 [候选]）"),
        ("G层_I_max", I_max, "候选",
         "M6 #71 预注册：积分钳制 ±0.2（E5 §1.2 [候选]）"),
        ("G层_β_base与带", (β_base, β带[0], β带[1]), "候选",
         "M6 #71 预注册：β_base=0.526（T9 临界面锚，k_battery.time_delay_calibration 同源）"
         "β 带 [0.3,1.2]（E5 §1.2 [候选值]）"),
        ("G层_k_p与η_i", (k_p, η_i), "候选",
         "M6 #71 预注册：k_p≲0.06、η_i≈3e-5/步（E5 交付④ [候选，K5 时滞标定落定前为候选值]；"
         "积分增量=η_i·R_reg·e_c 每拍，步率×拍长口径）"),
        ("G层_T_rec两段", (T_rec快, T_rec慢), "定稿",
         "M6 #71 预注册：K3⊕K4 两段式 T_rec^fast=2000 / T_rec^slow=16384 步（S10 联署）"),
        ("G层_windup守卫", (N_sw, δ_Ic), "定稿",
         "M6 #71 预注册：windup_guard N_sw=10、δ_Ic=0.02（定稿 §3.1 K3′；E1 漂移容差 ≤0.1×I_max 同值）"),
        ("G层_T_q", T_q, "候选",
         "M6 #71 预注册：复核静默窗 6144 步（E5 [候选值]；与 E1 500 R2 拍折算对齐=O6 开放）"),
        ("G层_上下文双源",
         {"ACh比阈": ACh比阈, "ACh持续R2拍": ACh持续R2拍, "旗标滞回R2拍": 旗标滞回R2拍,
          "失配登记R2拍": 失配登记R2拍, "N_conv候选": N_conv, "基线环": 基线环容量,
          "基线最少样本": ACh基线最少样本},
         "候选",
         "M6 #71 预注册：双源判别常数（L3 §1-A4 原文：ACh/median_wake(ACh)<0.5 持续≥100 R2 拍⇒offline-like、"
         "旗标滞回 ≥5 R2 拍、失配 >500 R2 拍⇒登记）；影子步轴换算 1 R2 拍=N_conv=200 步 [候选，O8 开放]"
         "⇒ 持续/滞回/登记 = 100/5/500 观测拍；基线=最近 65536 个 wake 共识步 ACh 样本中位、≥512 方可用"
         "（实现口径，L3 未定基线窗）"),
        ("G层_α带与尺度带", (list(α带), list(尺度带)), "定稿",
         "M6 #71 预注册：K1 α 带 [0.85,1.15] 于 n∈[16,512]（≥1.5 十进位，定稿 §3.1 K1）"),
        ("G层_三窗漂移", (list(三窗), 三窗漂移阈), "候选",
         "M6 #71 预注册：FSS 三窗 {16K,32K,64K} 漂移 ≤0.05（E5 §2 [候选]）"),
        ("G层_洞容忍", 洞容忍, "候选",
         "M6 #71 预注册：洞占比 ≤10% 且无 G3 跳变⇒作数（E5 交付② [候选]；与 E4 >20%⊥ 合并=O10 开放）"),
        ("G层_wake带与offline带", (list(wake带), list(offline带)), "定稿",
         "M6 #71 预注册：σ̂ 轴 wake 带 [0.95,1.10]（含死区）/offline 带 [0.85,1.00]（E1 §1-A4）"),
        ("G层_offline_steering", offline_steering默认, "定稿",
         "M6 #71 预注册：预注册开关默认 observe_only（active_band 须独立执行器论证=O2 开放，未实装）"),
        ("G层_post_shy连败阈", post_shy连败阈, "定稿",
         "M6 #71 预注册：K-lite 连败 3 次⇒离线相暂停+账本（L2 机制原文）"),
        ("G层_β龄期上限步", β龄期上限步, "候选",
         "M6 #71 预注册：β_slow 产出龄期上限 5s=25 R2 拍 ×N_conv=200 [候选，O8] ⇒ 5000 步；"
         "三态龄期 {新鲜/陈旧/出厂默认}（R2 产出龄期上限 5s，L1 机制原文）"),
        ("G层_stale拍阈", stale拍阈, "定稿",
         "M6 #71 预注册：连续 ≥3 喂步缺帧⇒stale_window（L1/E3 丢帧 stale 语义：连续 ≥3 拍缺包冻结 M 写入）"),
        ("G层_起飞门模式", "regime_ok(wake)", "定稿",
         "M6 #71 预注册：E5 版起飞门=regime_ok(wake) 为真方允许进入离线相；降级条款预注册"
         "（offline 域 K1/K2 不可辨识⇒wake 域 regime_ok+offline 域无 SCC 崩解）；"
         "E1 饥饿升级强制通道=O3 开放未实装"),
        ("G层_ACh_EMA_α", ACh_EMA_α, "候选",
         "M6 #71 预注册：ACh=EMA(pe) 的 α=1/τ，τ_ACh=1000 步 [候选；L3 只定 EMA 形式未定 τ，实装预注册初值]"),
        ("G层_θ_a", θ_a, "定稿",
         "M6 #71 预注册：σ̂ 条件分位自参考 θ_a=窗内 0.9 分位（A11 标定口径，σ̂ 标定=定稿行 178）"),
    )


注册键值表 = _注册键值表()
注册键数 = len(注册键值表)


class G层常量:
    """只读常量暴露（判据/台架引用同一份，杜绝魔法数漂移）。"""

    R_reg = R_reg
    W_o = W_o
    W_fast = W_fast
    δ_dead = δ_dead
    I_max = I_max
    β_base = β_base
    β带 = β带
    k_p = k_p
    η_i = η_i
    T_rec快 = T_rec快
    T_rec慢 = T_rec慢
    N_sw = N_sw
    δ_Ic = δ_Ic
    T_q = T_q
    α带 = α带
    尺度带 = 尺度带
    三窗 = 三窗
    三窗漂移阈 = 三窗漂移阈
    洞容忍 = 洞容忍
    wake带 = wake带
    offline带 = offline带
    offline_steering默认 = offline_steering默认
    post_shy连败阈 = post_shy连败阈
    β龄期上限步 = β龄期上限步
    stale拍阈 = stale拍阈
    ACh比阈 = ACh比阈
    N_conv = N_conv


# ════════════════════════════════════════════════════════════════════════
# §1 纯函数（零拟合：预注册统计量）
# ════════════════════════════════════════════════════════════════════════

def β限幅(e_c: float, I_c: float) -> float:
    """β_slow ← clip(β_base + k_p·e_c + I_c, [0.3,1.2])（L2 机制原式；纯函数，单元可测）。"""
    lo, hi = β带
    return float(min(max(β_base + k_p * float(e_c) + float(I_c), lo), hi))


def 条件中位数配对比(x: np.ndarray, 分位: float = θ_a) -> float:
    """σ̂ 原始估计（C13 清算条目口径）：θ=窗内 分位 自参考（θ_a=0.9）；
    σ̂_raw = median{ x_{t+1}/x_t : x_t ≥ θ }（条件中位数配对比）。条件对 <10 或无正分母 ⇒ NaN（不伪装）。"""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 32:
        return float("nan")
    阈 = float(np.quantile(x, 分位))
    前, 后 = x[:-1], x[1:]
    m = 前 >= 阈
    den = 前[m]
    num = 后[m]
    ok = den > 0
    if int(ok.sum()) < 10:
        return float("nan")
    比 = num[ok] / den[ok]
    return float(np.median(比))


_A11表缓存 = {"table": None}


def _A11表() -> dict:
    """A11 偏差标定表（模块级一次性缓存；与 k_battery.sigma_target 内部缓存同源同 seed 流）。"""
    if _A11表缓存["table"] is None:
        _A11表缓存["table"] = _kb.a11_calibration()
    return _A11表缓存["table"]


# ════════════════════════════════════════════════════════════════════════
# §2 G 层核心（影子期不接管控制流；消费供数接口，导出 β_slow 信号与守门判定）
# ════════════════════════════════════════════════════════════════════════
class G层核心:
    """C-HOME v2 情境门控 + 守门信号族 + K3⊕K4 两段式 + 起飞门 + β_slow 信号。

    供数接口（影子期代理）：
      喂步(pe, offline_flag, x_act, x_regime, 帧新鲜) —— 每步一次，O(1)；每 R_reg 步自动触发观测拍；
      regime_ok(x, ctx, hole_log) —— K1 判据（复用 k_battery.k1_regime_ok）；
      scc_state(adj, 失稳) —— K4 边界遥测（复用 k_battery.k4_scc）；
      注入kill()/recovery_ok() —— K3⊕K4 两段式；
      post_shy_audit(活性序列) —— K-lite；
      post_replay_quiet(t)/记录离线相结束(t) —— T_q 相位条件；
      consolidation_permit() —— 巩固前置门；起飞门(S, S_crit) —— 离线相准入。
    输出：β_slow 信号（三态龄期 5s 语义）+ e_c/σ̂ 遥测（_e_c史）+ windup/kill/α 复核账。

    构造注意：首次构造会经 k_battery.sigma_target 触发 A11 偏差标定（一次性，模块级缓存，
    约 3–6s 纯 CPU——合成 C1 系生成+查表；同进程后续构造零开销）。
    """

    def __init__(self, 账本=None, 控制环: bool = True):
        self.账本 = 账本
        self.k_c = 1 if 控制环 else 0        # 控制环启用（0=切除：K3 sham 口径）
        self.offline_steering = offline_steering默认
        if self.offline_steering != "observe_only":
            raise AssertionError("offline_steering=%r 未获独立执行器论证（O2 开放）——影子期仅 observe_only"
                                 % (self.offline_steering,))
        # —— σ̂_target：A11 查表（W_o=4096、θ=0.9 运行口径）——
        # 查表次序：k_battery.sigma_target 双线性插值 →（若 NaN：其插值会被表中其它 W_o×θ 格的
        # NaN 以 0×NaN=NaN 污染，而运行口径 (4096,0.9) 恰在 A11 网格上）精确格读数 → 退化 1.0+登记。
        try:
            st = _kb.sigma_target(W_o, θ_a)
        except Exception:
            st = float("nan")
        来源 = "k_battery.sigma_target 双线性查表"
        if not np.isfinite(st) or st <= 0:
            try:
                tab = _A11表()
                Wg = list(tab["Wo_grid"])
                Tg = list(tab["theta_grid"])
                if W_o in Wg and θ_a in Tg:
                    cand = float(tab["sigma_target"][Wg.index(W_o)][Tg.index(θ_a)])
                    if np.isfinite(cand) and cand > 0:
                        st = cand
                        来源 = "A11 精确格读数（W_o=4096,θ_a=0.9 均在标定网格；插值层 NaN 污染绕行）"
            except Exception:
                pass
        if not np.isfinite(st) or st <= 0:
            st = 1.0
            来源 = "退化1.0（A11 表不可用）"
            self._登记("结构事件",
                       {"事件": "σ̂_target A11 查表不可用⇒退化1.0"},
                       "M6：A11 偏差表不可用时的诚实退化（登记可审计；正常路径经 K5 过线后查表成立）")
        self.σ̂_target = float(st)
        self.σ̂_target来源 = 来源
        # —— 上下文双源状态 ——
        self.ACh = float("nan")              # 影子代理：EMA(pe)
        self._pe供给数 = 0
        self._基线环 = np.full(基线环容量, 0.0)
        self._基线头 = 0
        self._基线计 = 0
        self._ACh低持续 = 0                  # 观测拍
        self._ACh票 = "缺失"                 # wake / offline_like / 缺失
        self._ACh比 = float("nan")
        self._旗标原始 = False
        self._旗标稳定 = 0                   # 观测拍
        self._旗标票 = "wake"
        self.c_t = "wake"                    # 上下文（共识域）
        self.ambiguous = True                # ACh 缺失⇒初始 ambiguous（调节冻结至消歧）
        self._失配持续 = 0
        self._失配已登记 = False
        # —— 调节器状态（双带双积分器）——
        self.I_c = {"wake": 0.0, "offline": 0.0}
        self._I_c快照 = {"wake": 0.0, "offline": 0.0}
        self.β_slow = float(β_base)          # β_base 初值（未写入态）
        self._β写入拍 = None
        # —— 窗与环 ——
        self._win = np.full(W_o, 0.0)        # 当前域 σ̂ 观测窗（域切换即断窗重开）
        self._win头 = 0
        self._win计 = 0
        self._体制环 = np.full(体制环容量, 0.0)   # 体制观测环（K3 慢通道 α 复核载体；不断窗）
        self._体制头 = 0
        self._体制计 = 0
        self._域步计数 = 0                   # 当前域内自上次观测拍的步数
        self._首拍挂起 = False
        self._首拍延迟 = False
        # —— 账/遥测 ——
        self.拍数 = 0                        # 全局步轴
        self._tick数 = 0
        self._e_c史: deque = deque(maxlen=512)    # 每观测拍记录（K3 两段式扫描载体）
        self._最近拍: Optional[dict] = None
        self._windup记录: list = []
        self._windup待判: Optional[dict] = None
        self._windup待结算 = False
        self._kill账: list = []
        self._α复核账: list = []             # (t星, α)
        self._regime_wake = None             # (passed, status) 最近 wake 判定
        self._offline不可辨识 = False        # 降级条款触发器（预注册，不事后挑）
        self._离线相结束步 = None
        self._离线越带态 = False
        self._shy连败 = 0
        self.离线相暂停 = False
        self._stale连续 = 0
        self._scc_size = None
        self._scc_closed = None
        self._scc_members = None
        self._控制环写入数 = 0
        self._参数预注册入账()

    # —— 账本 ——
    def _登记(self, kind: str, payload: dict, 理由: str):
        if self.账本 is None:
            return None
        return self.账本.登记(kind, payload=payload, 理由=理由)

    def _参数预注册入账(self) -> None:
        """预注册常量以 kind="参数变更" 入账本（键前缀 "G层_"）；账本 None⇒跳过。
        变更清单扩充走 ledger 预授权路径：「扩充清单本身=结构事件」（ledger.py 源文件零改动）。"""
        if self.账本 is None:
            return
        try:
            from ahi_theory import ledger as _ledger
        except Exception:
            _ledger = None
        if _ledger is not None and hasattr(_ledger, "变更清单"):
            新键 = [k for k, _, _, _ in 注册键值表 if k not in _ledger.变更清单]
            if 新键:
                self.账本.登记("结构事件",
                               payload={"事件": "变更清单扩充", "键族": "G层_", "新增键": 新键},
                               理由="M6 #71 预注册：G 层参数键族入清单"
                                    "（ledger 预授权：扩充清单本身=结构事件；运行时扩充，"
                                    "ledger.py 源文件零改动）")
                _ledger.变更清单 = _ledger.变更清单 | frozenset(新键)
        for 键, 值, 身份, 理由 in 注册键值表:
            self.账本.登记("参数变更", payload={"键": 键, "值": 值, "身份": 身份},
                           理由=理由 + "（先验不调；变更须再次结构事件）")

    # —— 环提取（时间序） ——
    @staticmethod
    def _环现值(ring: np.ndarray, 头: int, 计: int, 容量: int, 只取尾: Optional[int] = None) -> np.ndarray:
        n = 计 if 只取尾 is None else min(只取尾, 计)
        if n <= 0:
            return np.empty(0, dtype=float)
        索引 = (头 - n + np.arange(n)) % 容量
        return ring[索引]

    # ── 供数接口（每步 O(1)） ────────────────────────────────────────────────
    def 喂步(self, pe: Optional[float] = None, offline_flag: bool = False,
             x_act: Optional[float] = None, x_regime: Optional[float] = None,
             帧新鲜: bool = True) -> Optional[dict]:
        """每步供数（O(1)）：ACh=EMA(pe)（pe=None⇒EMA 保持）、旗标原始值、活动观测、体制观测、帧新鲜性。
        x_regime=None⇒复用 x_act（单流模式）。x_act=None 或 帧新鲜=False ⇒ 该步无活动样本（窗不推进，
        步轴与拍计数照走）。每 R_reg 步自动触发一个观测拍（返回该拍记录，否则 None）。"""
        self.拍数 += 1
        if pe is not None:
            self._pe供给数 += 1
            a = self.ACh
            self.ACh = float(pe) if not np.isfinite(a) else a + ACh_EMA_α * (float(pe) - a)
        self._旗标原始 = bool(offline_flag)
        if 帧新鲜:
            self._stale连续 = 0
        else:
            self._stale连续 += 1
        if x_act is not None and 帧新鲜:
            i = self._win头
            self._win[i] = float(x_act)
            self._win头 = (i + 1) % W_o
            if self._win计 < W_o:
                self._win计 += 1
            j = self._体制头
            self._体制环[j] = float(x_act if x_regime is None else x_regime)
            self._体制头 = (j + 1) % 体制环容量
            if self._体制计 < 体制环容量:
                self._体制计 += 1
        # ACh wake 基线（共识 wake 且非 ambiguous 时逐步收样——域切换/冲突期不收，防衰减尾污染）
        if (not self.ambiguous) and self.c_t == "wake" and np.isfinite(self.ACh):
            b = self._基线头
            self._基线环[b] = self.ACh
            self._基线头 = (b + 1) % 基线环容量
            if self._基线计 < 基线环容量:
                self._基线计 += 1
        self._域步计数 += 1
        if self._域步计数 >= R_reg:
            self._域步计数 = 0
            return self._观测拍()
        return None

    # ── 观测拍（每 R_reg 步一次） ────────────────────────────────────────────
    def _观测拍(self) -> dict:
        self._tick数 += 1
        rec = {"t": self.拍数, "拍": self._tick数, "域": self.c_t, "ambiguous": self.ambiguous,
               "σ̂": float("nan"), "σ̂_fast": float("nan"), "e_c": float("nan"), "e_c_fast": float("nan"),
               "首拍": False, "I_c写": False, "β写": False, "k_c": self.k_c}
        # 1) 上下文双源合成（可能触发域切换协议/失配登记）
        self._上下文合成(rec)
        # 2) 首拍守门（一拍守门：只观测不写 β）
        if self._首拍挂起 and not self.ambiguous:
            if self._首拍延迟:
                self._首拍延迟 = False           # 切换拍本体：窗已断窗重开⇒空窗，自然无写
            else:
                rec["首拍"] = True
                self._首拍挂起 = False
                self._tickσ̂(rec)
                self._windup首拍结算(rec)
                self._e_c史.append(rec)
                self._最近拍 = rec
                return rec
        # 3) σ̂ 观测（主窗 W_o + 快子窗 W_fast）
        self._tickσ̂(rec)
        # 4) 分域控制/监测（ambiguous ⇒ 调节冻结至消歧）
        if not self.ambiguous:
            if self.c_t == "wake":
                self._wake调节(rec)
            else:
                self._offline监测(rec)
        # 5) 记录
        self._e_c史.append(rec)
        self._最近拍 = rec
        return rec

    def _tickσ̂(self, rec: dict) -> None:
        主 = self._环现值(self._win, self._win头, self._win计, W_o)
        raw = 条件中位数配对比(主)
        rec["σ̂"] = (raw / self.σ̂_target) if np.isfinite(raw) else float("nan")
        rec["e_c"] = math.log(rec["σ̂"]) if np.isfinite(rec["σ̂"]) and rec["σ̂"] > 0 else float("nan")
        n_fast = min(W_fast, self._win计)
        if n_fast >= 256:
            快 = self._环现值(self._win, self._win头, self._win计, W_o, 只取尾=n_fast)
            rawf = 条件中位数配对比(快)
            rec["σ̂_fast"] = (rawf / self.σ̂_target) if np.isfinite(rawf) else float("nan")
            rec["e_c_fast"] = (math.log(rec["σ̂_fast"])
                               if np.isfinite(rec["σ̂_fast"]) and rec["σ̂_fast"] > 0 else float("nan"))

    # —— 上下文双源合成 ——
    def _上下文合成(self, rec: dict) -> None:
        # ACh 源（影子代理=EMA(pe) 供数接口；ACh/median_wake(ACh)<0.5 持续≥100 R2 拍⇒offline-like）
        if self._pe供给数 == 0 or not np.isfinite(self.ACh):
            ach票 = "缺失"
        else:
            基线 = self._ACh基线()
            if 基线 is None or 基线 <= 0:
                ach票 = "wake"               # 基线未热⇒降判据无证据（诚实缺省 wake）
            else:
                self._ACh比 = self.ACh / 基线
                if self._ACh比 < ACh比阈:
                    self._ACh低持续 += 1
                else:
                    self._ACh低持续 = 0
                ach票 = "offline_like" if self._ACh低持续 >= ACh持续拍 else "wake"
        self._ACh票 = ach票
        # 旗标源（滞回：旗标稳定 ≥旗标滞回拍 方接受切换）
        原 = "offline" if self._旗标原始 else "wake"
        if 原 != self._旗标票:
            self._旗标稳定 += 1
            if self._旗标稳定 >= 旗标滞回拍:
                self._旗标票 = 原
                self._旗标稳定 = 0
        else:
            self._旗标稳定 = 0
        # 双源合成（ACh 电平阈值化 ⊕ MBP offline 旗标，双源一致性校验）
        if ach票 == "缺失":
            self.ambiguous = True            # ACh 缺失⇒context_ambiguous⇒调节冻结至消歧
            self._失配持续 = 0
            rec["上下文"] = "ambiguous(ACh缺失)"
        else:
            ach域 = "offline" if ach票 == "offline_like" else "wake"
            if ach域 == self._旗标票:
                self._失配持续 = 0
                self._失配已登记 = False
                if self.ambiguous:
                    self.ambiguous = False   # 消歧
                    if ach域 != self.c_t:
                        self._域切换协议(ach域)
                elif ach域 != self.c_t:
                    self._域切换协议(ach域)
                rec["上下文"] = "一致"
            else:
                self.ambiguous = True        # 冲突⇒ambiguous（调节冻结至消歧）
                self._失配持续 += 1
                if self._失配持续 > 失配登记拍 and not self._失配已登记:
                    self._失配已登记 = True
                    self._登记("结构事件",
                               {"事件": "上下文双源失配登记", "ACh票": ach票, "旗标票": self._旗标票,
                                "持续R2拍": self._失配持续},
                               "L3 §1-A4：ACh 与旗标失配 >500 R2 拍 ⇒ 账本登记（95_架构定稿 行 246）")
                rec["上下文"] = "ambiguous(冲突)"
        rec["域"] = self.c_t

    def _ACh基线(self) -> Optional[float]:
        """median_wake(ACh)：最近 ≤65536 个 wake 共识步 ACh 样本中位；<512 样本⇒None（基线未热）。
        实现口径（WCET）：以等步长子采样（步长=max(1, 计//4096)）估计中位数——EMA 慢信号的比例
        统计等价，每拍 ≤4097 点（G6_selftest [10] ≤2.0ms 观测预算内）。"""
        if self._基线计 < ACh基线最少样本:
            return None
        n = self._基线计
        步 = max(1, n // 4096)
        索引 = (self._基线头 - n + np.arange(0, n, 步)) % 基线环容量
        v = self._基线环[索引]
        v = v[np.isfinite(v) & (v > 0)]
        if v.size < ACh基线最少样本:
            return None
        return float(np.median(v))

    # —— 域切换协议（L2 机制行 148 一字照抄） ——
    def _域切换协议(self, 新域: str) -> None:
        旧域 = self.c_t
        # (i) 离开域 I_c 快照 + (t, e_c) 入账本
        self._I_c快照[旧域] = float(self.I_c[旧域])
        尾e = None
        for r in reversed(self._e_c史):
            if np.isfinite(r.get("e_c", float("nan"))):
                尾e = float(r["e_c"])
                break
        if self.账本 is not None and self.k_c > 0:
            self._登记("结构事件",
                       {"事件": "域切换I_c快照", "离开域": 旧域, "进入域": 新域,
                        "I_c快照": self._I_c快照[旧域], "t": self.拍数, "e_c": 尾e},
                       "C-HOME v2 切换协议(i)：当前域 I_c 快照+(t,e_c) 入账本（95_架构定稿 行 148）")
            self._控制环写入数 += 1
        # (ii) 进入域 I_c 从快照续（不清零）
        self.I_c[新域] = float(self._I_c快照[新域])
        # 断窗重开（分域观测，域切换即断窗重开）
        self._win计 = 0
        self._win头 = 0
        self._域步计数 = 0
        self._首拍挂起 = True
        self._首拍延迟 = True
        self.c_t = 新域
        # 域伴随状态
        if 新域 == "offline":
            self._windup待判 = {"I_c出发": float(self._I_c快照["wake"]), "t_start": self.拍数}
            self._windup待结算 = False
        else:
            self._离线相结束步 = self.拍数
            self._离线越带态 = False
            self._windup待结算 = True        # offline→wake：待回 wake 首拍结算 windup 循环

    def _windup首拍结算(self, rec: dict) -> None:
        """回 wake 首拍：|ΔI_c^wake| ≤ δ_Ic 且回 wake 首拍 e_c∈死区；任一失败⇒windup_event 入账本。"""
        if (not self._windup待结算) or self._windup待判 is None:
            return
        w = self._windup待判
        w["首拍e_c"] = rec["e_c"]
        w["ΔI_c"] = float(self.I_c["wake"]) - w["I_c出发"]
        ok = bool(np.isfinite(w["首拍e_c"]) and abs(w["首拍e_c"]) <= δ_dead
                  and abs(w["ΔI_c"]) <= δ_Ic)
        self._windup记录.append(dict(ok=ok, ΔI_c=w["ΔI_c"],
                                     首拍e_c=(float(w["首拍e_c"]) if np.isfinite(w["首拍e_c"])
                                              else float("nan")),
                                     t_start=w["t_start"], t_end=self.拍数))
        if not ok:
            self._登记("结构事件",
                       {"事件": "windup_event", "ΔI_c": w["ΔI_c"], "首拍e_c": w["首拍e_c"],
                        "t_start": w["t_start"], "t_end": self.拍数},
                       "K3′：wake→offline→wake 循环 |ΔI_c^wake|>δ_Ic 或 回 wake 首拍 e_c 越死区"
                       "（95_架构定稿 行 157/380）")
            self._控制环写入数 += 1
        self._windup待结算 = False
        self._windup待判 = None

    # —— wake 调节带（驱动执行器） ——
    def _wake调节(self, rec: dict) -> None:
        e = rec["e_c"]
        if not np.isfinite(e):
            return                           # 不可辨识拍：不写（诚实）
        rec["wake带越界"] = bool(not (wake带[0] <= rec["σ̂"] <= wake带[1]))
        if abs(e) > δ_dead:                  # 死区内不动（L2 机制行 144–145 一字照抄）
            if self.k_c > 0:
                self.I_c["wake"] = float(np.clip(self.I_c["wake"] + η_i * R_reg * e, -I_max, I_max))
                self.β_slow = β限幅(e, self.I_c["wake"])
                self._β写入拍 = self.拍数
                rec["I_c写"] = True
                rec["β写"] = True
        # k_c=0 切除：控制律旁路（观测照常、I_c/β 零写）

    # —— offline 监测带（只观测不控制） ——
    def _offline监测(self, rec: dict) -> None:
        # I_c^off 冻结；β_slow 保持；越带只报警（post_shy_audit 通道）——L2 机制行 146–147
        σ = rec["σ̂"]
        if not np.isfinite(σ):
            return
        越带 = bool(not (offline带[0] <= σ <= offline带[1]))
        rec["offline带越界"] = 越带
        if 越带 and not self._离线越带态:
            self._离线越带态 = True
            self._登记("结构事件",
                       {"事件": "offline带越带报警", "σ̂": float(σ), "t": self.拍数},
                       "L2：offline 监测带越带只报警（post_shy_audit 通道；95_架构定稿 行 147）")
        elif not 越带:
            self._离线越带态 = False

    # ── 守门信号族 ──────────────────────────────────────────────────────────
    def regime_ok(self, x, ctx: str = "wake", hole_log: Optional[dict] = None) -> dict:
        """regime_ok(c)（L2 机制行 153–155 判据式原文）＝K1 判据——直接复用 k_battery.k1_regime_ok
        （只读 import；DFA 估计器同源）。K5 已过线（meeting12 #42–#57）⇒ 前置满足。
        ctx="offline" 且判为不可辨识 ⇒ 触发起飞门降级条款（预注册，不事后挑）。"""
        res = _kb.k1_regime_ok(np.asarray(x, float), ctx=ctx, hole_log=hole_log)
        if ctx == "wake":
            self._regime_wake = (bool(res["passed"]), str(res["status"]))
        elif ctx == "offline" and res["status"] == "不可辨识":
            self._offline不可辨识 = True
        return res

    def scc_state(self, adj=None, 失稳: bool = False) -> dict:
        """scc_state(t)={size, closed?, members_changed?, event∈{转换|死亡|失败}}（L2 机制行 155）。
        adj=补偿流邻接阵（M5/M7 供数接口；K4 边界=A 图强连通分量，复用 k_battery.k4_scc，
        size=最大 SCC 节点数（≥2 节点口径，k4_scc 原样））。adj=None⇒复用上次邻接（从未供给⇒不可辨识）。"""
        if adj is not None:
            self._上次adj = np.asarray(adj)
        a = getattr(self, "_上次adj", None)
        if a is None:
            return dict(criterion="scc_state", status="不可辨识", size=0, closed=False,
                        members_changed=False, event="无", note="补偿流邻接阵未供给（M5/M7 供数接口）")
        sccs = _kb.k4_scc(a)
        主 = max(sccs, key=len) if sccs else []
        size = len(主)
        closed = size >= 1
        members = frozenset(主)
        changed = bool(self._scc_members is not None and members != self._scc_members)
        if size == 0:
            event = "死亡"
        elif changed:
            event = "转换"
        elif 失稳:
            event = "失败"
        else:
            event = "无"
        self._scc_members = members
        self._scc_closed = closed
        self._scc_size = size
        return dict(criterion="scc_state", status="ok", size=size, closed=closed,
                    members_changed=changed, event=event, sccs=[list(s) for s in sccs])

    # —— K3⊕K4 两段式 ——
    def 注入kill(self, t_步: Optional[int] = None) -> int:
        """K3 kill 事件注入接口：记录扰动事件（步轴；影子期 kill/演练标记登记归 E4，此处仅内存账）。
        返回 kill_id。"""
        t = self.拍数 if t_步 is None else int(t_步)
        self._kill账.append({"id": len(self._kill账), "t_kill": t, "域": self.c_t})
        return len(self._kill账) - 1

    def recovery_ok(self, kill_id: int, scc_alive: Optional[bool] = None) -> dict:
        """K3⊕K4 两段式（定稿 §3.1 / E5 §2-E4-1，判据式一字照抄）：
          T_rec^fast=2000 步内 e_c 回死区（"回"=同通道先离开后回归；快通道=W_fast=2048 快子窗
          σ̂ 快回判读——k_battery 双轴表原文，主通道兜底）
          ∧ T_rec^slow=16384 步内 α 回带（体制环末 ≤64K 的 DFA alpha_band 复核，k_battery 仪器）
          ∧ SCC 保持 ⇒ 切换瞬态；否则真失稳。
          scc_alive=None⇒取 scc_state 最近判定（从未评估⇒按 False=未评估不通过，诚实）。
          恢复确认（切换瞬态）为控制环 authored 账本写入（仅 k_c>0）；k_c=0 切除⇒控制环零账本写入
          （K3 验账口径：k_c=0 ⇒ 账本增量=0）。"""
        if int(kill_id) >= len(self._kill账):
            raise IndexError("kill_id 越界")
        k = self._kill账[int(kill_id)]
        t0 = int(k["t_kill"])
        # 快段：扫描 (t0, t0+T_rec快] 内观测拍——「回死区」按**同通道**判读：该通道先离开
        #       （|e_c|>δ_dead 的扰动签名）后回归（其后首个 |e_c|≤δ_dead 拍）；仅 ∃死区拍或跨通道
        #       混判不构成回归（主/快通道窗长不同、翻转有时差——跨通道混判会把主通道滞后误读为回归）。
        #       快通道=W_fast 快子窗 σ̂ 优先（k_battery 双轴表"σ̂ 快回判读"），主通道兜底。
        通道态: dict = {}
        for ch in ("e_c_fast", "e_c"):
            签拍 = 签值 = 回拍 = None
            for r in self._e_c史:
                tt = r["t"]
                if tt <= t0 or tt > t0 + T_rec快:
                    continue
                v = r.get(ch, float("nan"))
                if not np.isfinite(v):
                    continue
                if abs(v) <= δ_dead:
                    if 签拍 is not None and 回拍 is None:
                        回拍 = tt
                elif 签拍 is None:
                    签拍 = tt
                    签值 = float(v)
            通道态[ch] = (签拍, 签值, 回拍)
        fast_ok = False
        用通道 = None
        首回 = None
        签名 = None
        for ch in ("e_c_fast", "e_c"):
            签拍, 签值, 回拍 = 通道态[ch]
            if 签拍 is not None and 回拍 is not None:
                fast_ok = True
                用通道 = ch
                首回 = 回拍
                签名 = (签拍, 签值)
                break
        # 慢段：t0+T_rec慢 处（或其后首个已到时点）α 复核
        t星 = t0 + T_rec慢
        alpha = float("nan")
        slow_ok = False
        slow状态 = "未到时点"
        if self.拍数 >= t星:
            if self._体制计 >= 4096:
                环 = self._环现值(self._体制环, self._体制头, self._体制计, 体制环容量)
                s = _kb.dfa_summary(环)
                alpha = float(s["alpha_band"])
                self._α复核账.append((t星, alpha))
                if np.isfinite(alpha):
                    slow_ok = bool(α带[0] <= alpha <= α带[1])
                    slow状态 = "ok"
                else:
                    slow状态 = "不可辨识"
            else:
                slow状态 = "不可辨识（体制样本不足）"
        scc判 = bool(scc_alive) if scc_alive is not None else bool(self._scc_closed)
        scc已评 = bool(scc_alive is not None or self._scc_closed is not None)
        verdict = "切换瞬态" if (fast_ok and slow_ok and scc判) else "真失稳"
        if verdict == "切换瞬态" and self.k_c > 0:
            self._登记("结构事件",
                       {"事件": "K3⊕K4 恢复确认", "kill_id": int(kill_id), "t_kill": t0,
                        "fast通道": 用通道, "fast首回步": 首回, "alpha复核": alpha},
                       "K3⊕K4 两段式：T_rec^fast 内 e_c 回死区 ∧ T_rec^slow 内 α 回带 ∧ SCC 保持"
                       "⇒ 切换瞬态（控制环确认写入；95_架构定稿 行 156/379）")
            self._控制环写入数 += 1
        return dict(criterion="K3⊕K4", verdict=verdict, kill_id=int(kill_id), t_kill=t0,
                    fast=dict(ok=bool(fast_ok), 通道=用通道, 首回步=首回,
                              窗=T_rec快, deadband=δ_dead, 扰动签名=签名),
                    slow=dict(ok=bool(slow_ok), alpha=alpha, 状态=slow状态,
                              窗=T_rec慢, band=list(α带)),
                    scc_alive=scc判, scc_assessed=scc已评, k_c=self.k_c)

    # —— windup_guard（K3′） ——
    def windup_guard(self) -> dict:
        """windup_guard（定稿 §3.1 K3′）：wake→offline→wake ×N_sw=10，每次回 wake 后
        |ΔI_c^wake|≤δ_Ic=0.02 且回 wake 首拍 e_c∈死区；任一失败⇒windup_event（记录时即时登记）+FAIL。
        记录不足 N_sw⇒不可辨识（不伪装）。"""
        rec = list(self._windup记录)[-N_sw:]
        if len(rec) < N_sw:
            return dict(criterion="windup_guard", status="不可辨识", passed=False,
                        记录数=len(rec), 需要=N_sw)
        ok = all(r["ok"] for r in rec)
        return dict(criterion="windup_guard", passed=bool(ok),
                    status=("通过" if ok else "失败"), 记录数=len(rec), 需要=N_sw,
                    明细=[dict(序=i, ΔI_c=r["ΔI_c"], 首拍e_c=r["首拍e_c"],
                               t_start=r["t_start"], t_end=r["t_end"])
                          for i, r in enumerate(rec)])

    # —— post_shy_audit（K-lite） ——
    def post_shy_audit(self, 活性序列) -> dict:
        """post_shy_audit（L2 机制行 158）：K-lite（DFA α̂+谱指数 β̂）on C 驱动活性序列（供数接口）；
        α̂∈[0.85,1.15]（β̂ 只报告不判门）；连败 3 次⇒离线相暂停+账本；通过⇒连败清零并解除暂停。"""
        x = np.asarray(活性序列, float)
        x = x[np.isfinite(x)]
        if x.size < 4096:
            return dict(criterion="post_shy_audit", status="不可辨识", passed=False,
                        alpha=float("nan"), beta=float("nan"),
                        连败=self._shy连败, 离线相暂停=self.离线相暂停)
        s = _kb.dfa_summary(x)
        alpha = float(s["alpha_band"])
        beta = float(_kb.psd_channels(x)["beta_raw"])
        passed = bool(np.isfinite(alpha) and α带[0] <= alpha <= α带[1])
        if passed:
            if self._shy连败 >= post_shy连败阈 and self.离线相暂停:
                self.离线相暂停 = False
                self._登记("结构事件",
                           {"事件": "post_shy_audit 恢复（离线相暂停解除）", "alpha": alpha},
                           "L2：post_shy_audit 通过⇒连败清零、离线相暂停解除")
            self._shy连败 = 0
        else:
            self._shy连败 += 1
            if self._shy连败 >= post_shy连败阈 and not self.离线相暂停:
                self.离线相暂停 = True
                self._登记("结构事件",
                           {"事件": "post_shy_audit 连败3次⇒离线相暂停", "连败": self._shy连败,
                            "alpha": alpha},
                           "L2：post_shy_audit K-lite 连败 3 次⇒离线相暂停+账本（95_架构定稿 行 158）")
        return dict(criterion="post_shy_audit", status=("通过" if passed else "失败"), passed=passed,
                    alpha=alpha, beta=beta, 连败=self._shy连败, 离线相暂停=self.离线相暂停)

    # —— post_replay_quiet / consolidation_permit ——
    def 记录离线相结束(self, t_步: Optional[int] = None) -> int:
        """供数接口：登记最近离线相结束步（域切换 offline→wake 时自动调用；影子期亦可直注）。"""
        t = self.拍数 if t_步 is None else int(t_步)
        self._离线相结束步 = t
        return t

    def post_replay_quiet(self, t_步: Optional[int] = None) -> bool:
        """post_replay_quiet（L2 机制行 159）：距最近离线相结束 ≥T_q=6144 步 ⇒ true
        （E1 T_rev 相位条件）。从未有离线相⇒true（无静默要求）。"""
        t = self.拍数 if t_步 is None else int(t_步)
        if self._离线相结束步 is None:
            return True
        return (t - self._离线相结束步) >= T_q

    def consolidation_permit(self) -> dict:
        """consolidation_permit = regime_ok(wake) ∧ ¬stale_window（L2 机制行 160；E3 巩固写入前置门）。
        regime_ok(wake) 取最近一次 regime_ok(ctx="wake") 判定；从未测量⇒不放行（无证据不放行，诚实）。"""
        stale = self._stale连续 >= stale拍阈
        rw = self._regime_wake
        regime_wake = bool(rw is not None and rw[0])
        return dict(permit=bool(regime_wake and not stale), regime_ok_wake=regime_wake,
                    regime状态=(rw[1] if rw else "未测量"), stale_window=bool(stale),
                    stale连续=self._stale连续)

    # —— 离线相起飞门 ——
    def 起飞门(self, S: float, S_crit: float) -> dict:
        """离线相起飞门（L2 机制行 150/162；S 压力接口，影子期代理供数）：
          E5 版=「regime_ok(wake) 为真才允许进入离线相」；S≤S_crit⇒拒绝（压力不足）。
          降级条款（预注册，不事后挑）：offline 域 K1/K2 不可辨识⇒wake 域 regime_ok ∧ offline 域无 SCC 崩解。
          E1 饥饿升级（S>3·S_crit 强制离线相）=O3 开放，未实装（诚实边界）。"""
        模式 = ("降级：wake regime_ok ∧ offline 无 SCC 崩解" if self._offline不可辨识
                else "regime_ok(wake)")
        if S <= S_crit:
            return dict(允许=False, 理由="S 压力不足（S≤S_crit）", 模式=模式,
                        S=float(S), S_crit=float(S_crit))
        regime_wake = bool(self._regime_wake is not None and self._regime_wake[0])
        scc_ok = (self._scc_closed is not False)   # 未评估/保持⇒True；死亡(False)⇒False
        if self._offline不可辨识:
            允许 = bool(regime_wake and scc_ok)
            理由 = ("放行（降级条款：wake regime_ok ∧ SCC 未崩解）" if 允许
                    else ("拦截：SCC 崩解（降级条款第二支）" if regime_wake
                          else "拦截：wake regime_ok 不可辨识/未通过"))
        else:
            允许 = regime_wake
            理由 = ("放行（regime_ok(wake)=True）" if 允许
                    else "拦截：regime_ok(wake) 未通过或未测量")
        return dict(允许=允许, 理由=理由, 模式=模式, S=float(S), S_crit=float(S_crit))

    # —— β 信号（三态龄期 5s 语义；不回写 brain——影子期无人消费） ——
    def β状态(self) -> str:
        if self._β写入拍 is None:
            return "出厂默认"
        return "新鲜" if (self.拍数 - self._β写入拍) <= β龄期上限步 else "陈旧"

    # —— 遥测只读面 ——
    def e_c史(self) -> list:
        return list(self._e_c史)

    def 控制环写入数(self) -> int:
        return self._控制环写入数
