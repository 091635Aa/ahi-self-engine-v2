# -*- coding: utf-8 -*-
"""ahi_theory/ecae.py —— M7 · ECAE 生理层（六层架构 L3，影子模式）：H 印迹库 + C 皮层库双库
    + 价态四通道 + viability 包络 + 睡眠/离线重放 + A7 双约束接口 + G-Layer 供数面

预登记：ahi_theory/verify_all.py `_CHANGELOG` #73（2026-10-01『M7 ECAE 生理层预登记』，先于改码）。
规格出处：meeting11/95_架构定稿.md §2 L3 生理层（行 196–254，机制块+参数表一字照抄）
  + §3.1 V1（行 388）/V3（行 391）/V4（行 392）/V5（行 393）/V2′·K3′（行 390/380）。
参照实现风格：ahi_theory/g_layer.py（最新风格基准）、hdc_core.py、personality.py、selfthink.py。

影子模式纪律（红线）：
  · 纯新增；默认不加载（无模块级副作用、无内核接线——消费方显式 import 才实例化）；
  · 不 import brain/world/system/three_loop/mbp（影子纪律）；运行时仅惰性 import
    ahi_theory.ledger（变更清单预授权扩充路径，ledger.py 源文件零改动——g_layer 同律）；
    g_layer/hdc_core 为构造注入级类型引用（治理层实例经 治理层= 注入），模块本体不 import；
  · 零拟合红线：全文件无 lstsq / linalg.inv·pinv / linalg.solve / scipy.optimize / 参数搜索构造
    （C4 tokenize 级可检，E7_selftest [2] 自证）；softmax / CUSUM / EMA / 局部 delta 为既有机制原语；
  · SHY 归属注明：SHY 统一乘性因子只作用于本模块 C 库权重（W_act/W_free）；
    hdc_core.W_selfHop 属 E3 井阵，本模块不持有、不触碰（L3 参数表 SHY 行 + 定稿 Round3 终局）；
  · 账本纪律：预注册常数构造时以 kind="参数变更" 入账本（键前缀 "ECAE_"，候选/定稿身份逐条标注；
    run0 分位带校准前的 θ_w/θ_v 与实装初值为候选，校准后须再次结构事件方可变更）；
  · 诚实边界：影子期全部为机制自证（合成 pe/世界流驱动）；照顾者流 M9 未建 ⇒ R_care 子域接口
    预留+未发育退化（设备ID=None⇒主域，恒未发育路径）；离线占比目标带 [0.2,0.6] 为发育期目标
    （影子期只报告不调节）；run0 校准窗为实装口径（L3 未定窗长）；四通道引擎级接线（g_layer 的
    ACh 代理切换本模块真值）待 V2 引擎供数，g_layer 一行不改，联动自证在 E7_selftest [9] 台架完成。

机制总览（95_架构定稿 行 200–230 逐条落实，判据式一字照抄）：
  H 印迹库（海马式，容量 M 槽位）:
    事件 e_t = (z_t 摘要, a_t 动作, 结果摘要, tag v_t)；novelty_t = pe_t/(EMA(pe)+ε)；v_t = DA_t − 5HT_t
    if novelty_t > θ_w or |v_t| > θ_v:
        i = argmin_i(|tag_i|·exp(−age_i/τ_H))；H[i] = (key=kwta(z_t,k), payload, tag, age=0)
    age_i += 1 ∀i（R2 拍轴，禁裸步数；τ_H=5×10⁴ R2 拍；资格线 age>10·τ_H 且 |tag| 弱优先换出）
    ——换出两级判读：存在资格线内槽位（age>10·τ_H）⇒ 只在资格线内取 |tag| 最弱（弱优先）；
      否则全域 argmin(|tag_i|·exp(−age_i/τ_H))（机制原式兜底）。
  C 皮层库（慢统计）: 在线 ε_on≈0（ACh 门控：ACh 低于阈⇒ε_on=0）；
    C_act 动作条件子空间（efference 预测器：ẑ′=W_act@(z⊛a)，键=z⊛a；R_care 子域接口预留）/
    C_free 无条件子空间（ẑ′=W_free@z）；局部 delta 学习 C ← C − ε·(pe_r ⊗ pre)（零反向传播）。
  价态四通道（取代 L 线性混合与标量 mom）:
    DA = max(0, −Δpe_rel)   # 门控写入 tag 强度、充当重放采样
    NE = CUSUM(pe 分布)      # 抬增益、中断离线相（觉醒）：c←max(0,c+(pe−μ0)/σ0−k_ne)，c>h_ne⇒突变（报警后复位）
    ACh = EMA(pe)            # 门控在线学习率 ε_on；兼作情境判别交叉校验源
    5HT = e_int              # 降 engagement、写负 tag
  viability（x 的慢 readout，τ_q∈[200,2000]）:
    x = [节律违反度, H 占用率, sat, ‖W‖ 增长]；包络 [lo,hi] 构造给定（预注册）；
    e_int = Σ max(0,lo−x)+max(0,x−hi)；e_int>0 ⇒ 5HT↑+注意捕获+负 tag；回稳⇒DA 小信号；
    降级版（R2 死）仅在 {x1,x3} 上计算仍触发强制优先。
  睡眠/离线重放:
    S ← S + novelty_t；S > S_crit ⇒ 请求离线相（起飞门=g_layer 供给）；R_r = clip(κ_S·(S−S_crit), 5 s, 600 s)
    睡毕 S←S·(1−完成度)（防打断销账）
    离线相每步: i ~ softmax((DA·v_i + ACh·novelty_i − age_i/τ_H)/T_DA)     # 巩固选择律（可消融）
      pe_r = C_act(a)(z) − z′；C ← C − ε_off·(pe_r ⊗ pre)                  # 局部 delta，零反向传播
      W_C ← W_C·(W_target/‖W_C‖_F)                                        # SHY 统一乘性因子（Round3 修正；
                                                                           #  W_target=SHY_系数·z维度，维度缩放）
      tag_i ← tag_i·(1−η_f)；NE 突变 ⇒ 立即中断离线相
    离线占比目标带 [0.2, 0.6]（发育期）；S_crit = k_S·⟨novelty⟩_run0，k_S=10³
  A7 双约束: τ_e = 20 ms（=1 R1 拍）资格迹——照顾者应答 δ≥40 ms 时资格迹已衰减 e^{−40/20}≈0.135
    <迹写入阈（候选 0.2）⇒ 照顾者条件按构造不可写入 C_act（L3 价态注入行：照顾者条件照常学进
    C_free/M 的 R_care 子域——M9 未建，接口预留）；pe_self 只在身体域子空间计算（R_care 分域堵读取）。

WCET 合同：每 R2 拍喂步 ≤2.0ms（基线 ε_on=0 ⇒ C 零矩阵写，快路径=四通道+H+viability）；
离线相每步 ≤2.0ms（W_act@pre + 秩1 delta + SHY 范数，float32 GEMV）；四通道读出 O(1)。
WCET 实测口径 = 默认 z维度=128（影子期实现口径【候选】——VM 内存带宽受限下 1024² 矩阵算子
≈4-5ms/次，128 维使合同留有 ≥3× 裕度；引擎接线若升维须重新实测并经结构事件）。
（E7_selftest [10] 实测。）
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

__all__ = [
    "ECAE常量", "ECAE核心", "kwta", "注册键数",
    "ECAE种子", "τ_H拍", "资格线倍数", "M默认", "k稀疏度", "θ_w", "θ_v",
    "ε_on基线", "ε_off", "ACh门控阈", "ACh_EMA_α", "NE_k", "NE_h", "run0窗",
    "包络", "τ_q拍", "k_S", "R_r下界秒", "R_r上界秒", "R2拍秒", "κ_S",
    "占比带", "T_DA", "SHY_系数", "η_f", "τ_e毫秒", "迹写入阈", "回稳小信号", "novelty_ε",
]

# ════════════════════════════════════════════════════════════════════════
# §0 预注册常量（95_架构定稿 L3 参数表 + 机制块；变更 = kind="参数变更" 账本结构事件，键前缀 "ECAE_"）
# ════════════════════════════════════════════════════════════════════════
ECAE种子 = 2048                 # W 初始化/重放采样单 rng 派生（预注册顺序，不挑）【定稿（结构）】
τ_H拍 = 5e4                     # H 印迹时间常数（R2 拍；折算墙钟 ≥1h 下限；E4 附加条件：不跨机制比较）【定稿】
资格线倍数 = 10                 # 槽位资格线 age > 10·τ_H 且 |tag| 弱优先换出（E1 §2#2）【定稿】
M默认 = 128                     # H 容量 M（toy 扫描 {64,128,256,512}+V1 单调容差=E1 §6-2；扫描前预注册默认）【候选】
k稀疏度 = 32                    # kwta 稀疏度（机制块 key=kwta(z_t,k)，预注册 k=32）【定稿】
θ_w = 2.0                       # novelty 写入阈（run0 分位带校准后入账本，校准前候选）【候选】
θ_v = 0.5                       # |tag v_t| 写入阈（run0 分位带校准后入账本，校准前候选）【候选】
ε_on基线 = 0.0                  # 在线学习率基线 ≈0（ACh 门控乘子；基线字面 0——门控真值经预注册刀口 ε_on刀口 验证）【候选】
ε_off = 1e-4                    # 离线学习率（逐级放大作单调性干预=预注册干预面，E1 §3.1/§1-A3）【候选】
ACh门控阈 = 0.05                # ACh 低于此阈 ⇒ ε_on=0（实装预注册初值；run0 分位带校准前候选）【候选】
ACh_EMA_α = 0.001               # ACh=EMA(pe) 的 α=1/τ（τ_ACh=1000 拍；L3 只定 EMA 形式未定 τ；与 g_layer 同源同值）【候选】
NE_k = 0.5                      # NE CUSUM 允许量（L3 只定 CUSUM 形式；实装预注册初值）【候选】
NE_h = 5.0                      # NE 突变判决阈（CUSUM 越阈⇒觉醒中断信号）【候选】
run0窗 = 1000                   # ⟨novelty⟩_run0 与 NE 参考段（μ0/σ0）共用校准窗（R2 拍；实装口径，L3 未定窗长）【候选】
包络 = {"x1节律违反度": (0.0, 0.05), "x2H占用率": (0.0, 0.95),
        "x3sat": (0.2, 1.0), "x4W增长": (0.0, 0.05)}   # viability 包络 [lo,hi]（构造给定，预注册）【候选】
τ_q拍 = 500                     # x 慢 readout τ_q∈[200,2000]（L3 参数表），构造取带内中点【候选】
k_S = 1e3                       # S_crit = k_S·⟨novelty⟩_run0（E1 §2#6，E1 认领）【定稿】
R_r下界秒 = 5.0                 # R_r clip 下界（C14 闭合）【定稿】
R_r上界秒 = 600.0               # R_r clip 上界（C14 闭合）【定稿】
R2拍秒 = 0.2                    # 1 R2 拍=200ms（三时钟域合同 R0 1ms/R1 20ms/R2 200ms）【定稿（结构）】
κ_S = 1.0                       # R_r 斜率（"κ_S 使满 S⇒300 s"标定待 run0；候选初值）【候选】
占比带 = (0.2, 0.6)             # 离线占比目标带（发育期；E1 认领、E2 可经账本请求调整；影子期只报告不调节）【定稿】
T_DA = 0.1                      # 重放采样温度 T_DA（巩固选择律分母；实装预注册初值）【候选】
SHY_系数 = 0.01                 # SHY 目标范数系数：W_target = SHY_系数·z维度（1024 口径=10.24；与 W 初始化 σ 同源）【候选】
η_f = 0.01                      # 离线重放 tag 衰减率 tag←tag·(1−η_f)（实装预注册初值）【候选】
τ_e毫秒 = 20.0                  # A7 资格迹时间常数（=1 R1 拍；A7 闭合）【定稿】
迹写入阈 = 0.2                  # C_act 写入资格阈：δ≥40ms⇒e^{−2}≈0.135<阈⇒按构造不可写（实装口径）【候选】
回稳小信号 = 0.05               # e_int 回稳⇒DA 小信号幅度（机制块"回稳⇒DA 小信号"；实装预注册初值）【候选】
novelty_ε = 1e-9                # novelty=pe/(EMA(pe)+ε) 分母保护（结构常量）【定稿】


def _注册键值表():
    """预注册常数表（构造时逐条 kind="参数变更" 入账本；变更须再次结构事件）。"""
    return (
        ("ECAE_种子", ECAE种子, "定稿",
         "M7 #73 预注册：W 初始化/重放采样单 rng 派生种子（结构常量，预注册顺序固定不挑）"),
        ("ECAE_τ_H", τ_H拍, "定稿",
         "M7 #73 预注册：τ_H=5×10⁴ R2 拍（95_架构定稿 L3 参数表；E1 认领、E4 附加条件=不跨机制比较）"),
        ("ECAE_槽位资格线", (资格线倍数, τ_H拍), "定稿",
         "M7 #73 预注册：age > 10·τ_H 且 |tag| 弱优先换出（E1 §2#2）；"
         "换出两级判读=资格线内 |tag| 弱优先、否则全域 argmin(|tag|·exp(−age/τ_H)) 原式兜底"),
        ("ECAE_H容量M", M默认, "候选",
         "M7 #73 预注册：容量 M=128（toy 扫描 {64,128,256,512}、V1 裕度应随 M 变紧+预注册单调容差=E1 §6-2；"
         "扫描前预注册默认 128，扫描后以结构事件落定）【候选】"),
        ("ECAE_k稀疏度", k稀疏度, "定稿",
         "M7 #73 预注册：kwta 稀疏度 k=32（L3 机制块 key=kwta(z_t,k) 预注册值）"),
        ("ECAE_写入触发", (θ_w, θ_v), "候选",
         "M7 #73 预注册：写入触发 novelty_t>θ_w ∨ |v_t|>θ_v，θ_w=2.0/θ_v=0.5 为预注册初值"
         "（run0 分位带校准后入账本——E1 §3.1，校准前为候选）"),
        ("ECAE_ε族", (ε_on基线, ε_off, ACh门控阈), "候选",
         "M7 #73 预注册：ε_on 基线≈0（ACh 门控：ACh 低于阈⇒ε_on=0；基线字面 0，门控真值经预注册刀口"
         " ε_on刀口 构造参数在台架验证）；ε_off=1e-4（逐级放大作单调性干预=预注册干预面，E1 §3.1/§1-A3）；"
         "ACh 门控阈=0.05（实装预注册初值，run0 分位带校准前候选）"),
        ("ECAE_ACh_EMA_α", ACh_EMA_α, "候选",
         "M7 #73 预注册：ACh=EMA(pe) 的 α=1/τ，τ_ACh=1000 拍（L3 只定 EMA 形式未定 τ；"
         "与 g_layer.ACh_EMA_α 同源同值——四通道真值化联动的一致性前提）【候选】"),
        ("ECAE_NE_CUSUM", (NE_k, NE_h), "候选",
         "M7 #73 预注册：NE=CUSUM(pe 分布)：c←max(0,c+(pe−μ0)/σ0−k_ne)，k_ne=0.5；c>h_ne=5.0⇒突变"
         "（觉醒中断信号，报警后复位）；μ0/σ0=run0 参考段均值/σ（σ0 下移=max(std,0.1|μ0|,1e-6) 实现口径）；"
         "L3 只定 CUSUM 形式，参数为实装预注册初值【候选】"),
        ("ECAE_viability包络", 包络, "候选",
         "M7 #73 预注册：x=[节律违反度,H占用率,sat,‖W‖增长]；包络 [lo,hi] 构造给定（预注册）；"
         "e_int=Σmax(0,lo−x)+max(0,x−hi)；降级版（R2 死）仅 {x1,x3} 上计算仍触发强制优先【候选】"),
        ("ECAE_τ_q", τ_q拍, "候选",
         "M7 #73 预注册：x 的慢 readout τ_q∈[200,2000]（L3 参数表），构造取带内中点 500 R2 拍【候选】"),
        ("ECAE_k_S", k_S, "定稿",
         "M7 #73 预注册：S_crit = k_S·⟨novelty⟩_run0，k_S=10³（E1 §2#6，E1 认领）"),
        ("ECAE_run0窗", run0窗, "候选",
         "M7 #73 预注册：⟨novelty⟩_run0 与 NE 参考段共用校准窗 1000 R2 拍（实装口径，L3 未定窗长）【候选】"),
        ("ECAE_R_r_clip", (R_r下界秒, R_r上界秒, R2拍秒), "定稿",
         "M7 #73 预注册：R_r=clip(κ_S·(S−S_crit), 5s, 600s)（C14 闭合）；1 R2 拍=200ms（三时钟域合同）"),
        ("ECAE_κ_S", κ_S, "候选",
         "M7 #73 预注册：κ_S=1.0 s/单位（'κ_S 使满 S⇒300 s' 标定待 run0；候选初值）"),
        ("ECAE_离线占比带", 占比带, "定稿",
         "M7 #73 预注册：离线占比目标带 [0.2,0.6]（发育期；E1 认领、E2 可经账本请求调整；影子期只报告不调节）"),
        ("ECAE_T_DA", T_DA, "候选",
         "M7 #73 预注册：重放采样温度 T_DA=0.1（巩固选择律分母；实装预注册初值）【候选】"),
        ("ECAE_SHY", (SHY_系数, η_f), "候选",
         "M7 #73 预注册：SHY 统一乘性因子 W←W·(W_target/‖W‖_F)（形式=定稿 E1 终局，逐行异因子版废除）；"
         "目标范数系数 SHY_系数=0.01（W_target=SHY_系数·z维度：1024 口径=10.24、默认 128 口径=1.28，"
         "与 W 初始化 σ=0.01 同源——维度缩放为实现口径【候选】）、η_f=0.01【候选，实装预注册初值】；"
         "作用域=本模块 C 库权重（W_act/W_free），不动 hdc_core.W_selfHop（归属注明）"),
        ("ECAE_A7_τe", τ_e毫秒, "定稿",
         "M7 #73 预注册：τ_e=20ms 资格迹（=1 R1 拍；消融判据：放宽至 500ms⇒ρ 漂移与否双向可失败，A7 闭合）"),
        ("ECAE_A7_迹阈", (迹写入阈, 40.0), "候选",
         "M7 #73 预注册：C_act 写入资格阈 0.2 与 40ms 判据（δ≥40ms⇒e^{−40/20}≈0.135<阈⇒照顾者条件"
         "按构造不可写入 C_act；实装口径，run0 校准前候选）"),
        ("ECAE_回稳小信号", 回稳小信号, "候选",
         "M7 #73 预注册：e_int 回稳⇒DA 小信号幅度 0.05（机制块原文无数值，实装预注册初值）【候选】"),
        ("ECAE_novelty_ε", novelty_ε, "定稿",
         "M7 #73 预注册：novelty=pe/(EMA(pe)+ε) 分母保护（结构常量）"),
    )


注册键值表 = _注册键值表()
注册键数 = len(注册键值表)


class ECAE常量:
    """只读常量暴露（判据/台架引用同一份，杜绝魔法数漂移）。"""

    种子 = ECAE种子
    τ_H = τ_H拍
    资格线倍数 = 资格线倍数
    M = M默认
    k = k稀疏度
    θ_w = θ_w
    θ_v = θ_v
    ε_on基线 = ε_on基线
    ε_off = ε_off
    ACh门控阈 = ACh门控阈
    ACh_EMA_α = ACh_EMA_α
    NE_k = NE_k
    NE_h = NE_h
    run0窗 = run0窗
    包络 = 包络
    τ_q = τ_q拍
    k_S = k_S
    R_r下界秒 = R_r下界秒
    R_r上界秒 = R_r上界秒
    R2拍秒 = R2拍秒
    κ_S = κ_S
    占比带 = 占比带
    T_DA = T_DA
    SHY系数 = SHY_系数
    η_f = η_f
    τ_e毫秒 = τ_e毫秒
    迹写入阈 = 迹写入阈
    novelty_ε = novelty_ε


# ════════════════════════════════════════════════════════════════════════
# §1 纯函数（零拟合）
# ════════════════════════════════════════════════════════════════════════

def kwta(z: np.ndarray, k: int = k稀疏度) -> np.ndarray:
    """k-winner-take-all 稀疏键（H 印迹 key，L3 机制块 key=kwta(z_t,k)，k 预注册=32）：
    |z| top-k 分量保留 sign、其余置 0。并列值按 argpartition 确定性次序（无随机）。
    纯函数（零拟合）；k 超界钳制；零输入⇒零键。"""
    z = np.asarray(z, dtype=np.float32).ravel()
    键 = np.zeros_like(z)
    k = int(min(max(int(k), 0), z.size))
    if k == 0 or z.size == 0:
        return 键
    索引 = np.argpartition(-np.abs(z), k - 1)[:k]
    键[索引] = np.sign(z[索引])
    return 键


def _softmax(scores: np.ndarray) -> np.ndarray:
    """数值稳定 softmax（既有机制原语，非拟合）。"""
    s = np.asarray(scores, dtype=np.float64)
    s = s - s.max()
    e = np.exp(s)
    总 = e.sum()
    if 总 <= 0 or not np.isfinite(总):
        return np.full(s.shape, 1.0 / s.size)
    return e / 总


def _熵(p: np.ndarray) -> float:
    """Shannon 熵 nats（V4-DA 影子版读出用）。"""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


# ════════════════════════════════════════════════════════════════════════
# §2 ECAE 核心（影子期不接管控制流；四通道/C 双库/H 印迹/离线重放机制本体）
# ════════════════════════════════════════════════════════════════════════
class ECAE核心:
    """L3 生理层影子核心（95_架构定稿 行 200–230 机制本体）。

    供数接口（影子期）：
      喂步(pe, z, a, z_next, 节律违反度, sat, 帧新鲜) —— 每 R2 拍一次；pe=预测误差标量（L4 e_t 或等价量），
        z=状态摘要向量（与 z维度 同维；HDC 侧可取 步进R1()["x"]），a=动作向量（与 z 同维；HDC 侧经
        Φ_act 投影；None⇒零动作），z_next=结果摘要向量；节律违反度/sat=viability 供数；帧新鲜=False⇒R2 死
        （降级版 viability 仅 {x1,x3}）。返回该拍记录（四通道/novelty/写入/e_int/注意捕获/S…）。
      压力()/请求离线相()/进入离线相()/离线步(pe)/结束离线相() —— S 压力→离线相状态机；
        起飞门经 治理层（g_layer.G层核心 实例，构造注入）.起飞门(S, S_crit) 联动；
        结束离线相 自动调 治理层.记录离线相结束()（post_replay_quiet 相位联动）。
      get_四通道() —— G-Layer 消费面：DA/NE/ACh/5HT 读出（g_layer ACh 代理切换本模块真值的
        联动自证在 E7_selftest [9]；g_layer 一行不改）。
      A7资格(δ毫秒, 设备ID) —— A7 双约束接口（τ_e=20ms 资格迹+设备 ID 分域）。
      选择分布(DA, ACh)/查事件pe(i)/C范数()/列印迹() —— 判据读出面（V1/V3/V4 手算对照）。
      消融通道(名)/通道恢复(名) —— V4 失效矩阵消融刀口（DA/NE/ACh/5HT 各自可切除）；
      巩固剥夺() —— E3 巩固剥夺消融刀口（ε_off=0 + H 清空，定稿 L3→L4 接口）。

    构造注意：账本非 None 时惰性 import ahi_theory.ledger 做变更清单预授权扩充（运行时扩充，
    ledger.py 源文件零改动）；ε_on刀口=None⇒预注册基线 ε_on基线=0（基线≈0 字面；台架验证门控真值
    时可设 >0——预注册干预面，同 g_layer 控制环 k_c 之律）。
    """

    def __init__(self, z维度: int = 128, 容量M: Optional[int] = None,
                 账本=None, 治理层=None, 种子: int = ECAE种子,
                 ε_on刀口: Optional[float] = None):
        self.z维度 = int(z维度)       # 默认 128=影子期实现口径【候选】（引擎接线时与 L4 摘要维度对齐，
                                      #  须经结构事件；HDC 联动台架可显式传 1024）
        self.M = int(M默认 if 容量M is None else 容量M)
        self.账本 = 账本
        self.治理层 = 治理层          # g_layer.G层核心 类型引用（起飞门/记录离线相结束 联动；不 import）
        self.ε_on实例 = float(ε_on基线 if ε_on刀口 is None else float(ε_on刀口))
        self._ε_off实例 = float(ε_off)
        self.W_target实例 = float(SHY_系数 * self.z维度)   # SHY 目标范数（维度缩放，SHY_系数 口径）
        rng = np.random.default_rng(种子)          # 单 rng 顺序派生（预注册顺序，不挑）
        self._rng = rng
        # —— C 双库（float32；SHY 作用域=C 库权重；hdc_core.W_selfHop 不在本模块——归属注明）——
        self.W_act = (0.01 * rng.standard_normal((self.z维度, self.z维度))).astype(np.float32)
        self.W_free = (0.01 * rng.standard_normal((self.z维度, self.z维度))).astype(np.float32)
        self._W范数缓存 = [float(np.linalg.norm(self.W_act)), float(np.linalg.norm(self.W_free))]
        self._W初始范数 = self._W范数缓存[0] + self._W范数缓存[1]
        # —— H 印迹库 ——
        self.H: list[dict] = []       # {"key","z","a","z_next","tag","age","novelty","拍"}
        # —— 价态四通道状态 ——
        self.ACh = float("nan")
        self.DA = 0.0
        self.NE = 0.0
        self._5HT = 0.0
        self._NE_c = 0.0
        self._pe_prev: Optional[float] = None
        self._μ0 = float("nan")
        self._σ0 = float("nan")
        self._run0pe: list[float] = []
        self._run0nov: list[float] = []
        self.run0热 = False
        self.S = 0.0
        self.S_crit: Optional[float] = None
        # —— viability ——
        self._x慢 = np.zeros(4, dtype=np.float64)
        self._上拍e_int = 0.0
        self.注意捕获 = False
        # —— 消融刀口（V4 失效矩阵；预注册接口）——
        self.消融 = {"DA": False, "NE": False, "ACh": False, "5HT": False}
        # —— 离线相状态机 ——
        self._离线态: Optional[dict] = None
        # —— 计数与遥测 ——
        self.拍数 = 0
        self.写入数 = 0
        self.在线写事件 = 0
        self.离线步数 = 0
        self.离线相数 = 0
        self.突变数 = 0
        self._参数预注册入账()

    # —— 账本 ——
    def _登记(self, kind: str, payload: dict, 理由: str):
        if self.账本 is None:
            return None
        return self.账本.登记(kind, payload=payload, 理由=理由)

    def _参数预注册入账(self) -> None:
        """预注册常量以 kind="参数变更" 入账本（键前缀 "ECAE_"）；账本 None⇒跳过。
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
                               payload={"事件": "变更清单扩充", "键族": "ECAE_", "新增键": 新键},
                               理由="M7 #73 预注册：ECAE 生理层参数键族入清单"
                                    "（ledger 预授权：扩充清单本身=结构事件；运行时扩充，"
                                    "ledger.py 源文件零改动）")
                _ledger.变更清单 = _ledger.变更清单 | frozenset(新键)
        for 键, 值, 身份, 理由 in 注册键值表:
            self.账本.登记("参数变更", payload={"键": 键, "值": 值, "身份": 身份},
                           理由=理由 + "（先验不调；变更须再次结构事件）")

    # —— 内部辅助 ——
    def _W范数(self) -> float:
        return self._W范数缓存[0] + self._W范数缓存[1]

    def _E_on_eff(self) -> float:
        """ε_on ACh 门控：ε_on_eff = ε_on实例·1[ACh ≥ ACh门控阈]（ACh 缺失/冻结于阈下⇒0）。"""
        if not np.isfinite(self.ACh) or self.ACh < ACh门控阈:
            return 0.0
        return self.ε_on实例

    def _NE步(self, pe: float) -> bool:
        """NE=CUSUM(pe 分布)（觉醒中断信号）：c←max(0,c+(pe−μ0)/σ0−k_ne)；c>h_ne⇒突变（报警后复位）。
        消融 NE⇒通道置 0 恒不突变（V4-NE 中断失效刀口）；run0 未热⇒参考段缺失恒不突变（不伪装）。"""
        if self.消融["NE"]:
            self.NE = 0.0
            return False
        if not self.run0热 or not np.isfinite(self._μ0):
            return False
        zscore = (float(pe) - self._μ0) / self._σ0
        self._NE_c = max(0.0, self._NE_c + zscore - NE_k)
        self.NE = float(self._NE_c)
        if self._NE_c > NE_h:
            self._NE_c = 0.0
            self.突变数 += 1
            return True
        return False

    def _e_int(self, 帧新鲜: bool) -> tuple:
        """viability 包络：x=[x1 节律违反度,x2 H占用率,x3 sat,x4 ‖W‖增长]（τ_q 慢 readout 已在喂步更新）；
        e_int=Σ max(0,lo−x)+max(0,x−hi)；帧新鲜=False（R2 死）⇒降级版仅 {x1,x3}（仍触发强制优先）。"""
        x = self._x慢
        域 = (0, 1, 2, 3) if 帧新鲜 else (0, 2)     # 降级版 {x1 节律违反度, x3 sat}
        e = 0.0
        for i in 域:
            lo, hi = 包络[("x1节律违反度", "x2H占用率", "x3sat", "x4W增长")[i]]
            e += max(0.0, lo - float(x[i])) + max(0.0, float(x[i]) - hi)
        降级 = not 帧新鲜
        return e, 降级

    def _换出位(self) -> int:
        """换出：资格线 age>10·τ_H 的槽位内 |tag| 弱优先；无资格线槽位⇒全域 argmin(|tag_i|·exp(−age_i/τ_H))
        （L3 机制块原式兜底）。"""
        资格 = [i for i, s in enumerate(self.H) if s["age"] > 资格线倍数 * τ_H拍]
        if 资格:
            return min(资格, key=lambda i: abs(self.H[i]["tag"]))
        分 = [abs(self.H[i]["tag"]) * math.exp(-self.H[i]["age"] / τ_H拍) for i in range(len(self.H))]
        return int(np.argmin(np.asarray(分)))

    def _写入事件(self, z: np.ndarray, a: Optional[np.ndarray],
                  z_next: Optional[np.ndarray], tag: float, novelty: float) -> None:
        z32 = np.asarray(z, dtype=np.float32).ravel().copy()
        if a is None:
            a32 = np.zeros(self.z维度, dtype=np.float32)
        else:
            a32 = np.asarray(a, dtype=np.float32).ravel().copy()
        zn32 = z32.copy() if z_next is None else np.asarray(z_next, dtype=np.float32).ravel().copy()
        槽 = {"key": kwta(z32, k稀疏度), "z": z32, "a": a32, "z_next": zn32,
              "tag": float(tag), "age": 0, "novelty": float(novelty), "拍": self.拍数}
        if len(self.H) < self.M:
            self.H.append(槽)
        else:
            i = self._换出位()
            self.H[i] = 槽
        self.写入数 += 1

    # ── 每 R2 拍更新（喂步） ──────────────────────────────────────────────────
    def 喂步(self, pe: float, z: Optional[np.ndarray] = None, a: Optional[np.ndarray] = None,
             z_next: Optional[np.ndarray] = None, 节律违反度: float = 0.0,
             sat: float = 1.0, 帧新鲜: bool = True) -> dict:
        """每 R2 拍一次（拍轴=R2 拍，禁裸步数）：四通道→viability→S 压力→H 写入→C 在线（ACh 门控）。
        返回该拍记录。"""
        self.拍数 += 1
        pe = float(pe)
        # (1) age_i += 1 ∀i（R2 拍轴；新写入槽在本拍写入处置 age=0）
        for s in self.H:
            s["age"] += 1
        # (2) ACh = EMA(pe)（首拍 ACh=pe；消融冻结⇒保持现值——V4-ACh 刀口）
        if not np.isfinite(self.ACh):
            self.ACh = pe
        elif not self.消融["ACh"]:
            self.ACh = self.ACh + ACh_EMA_α * (pe - self.ACh)
        novelty = pe / (self.ACh + novelty_ε)
        # (3) DA = max(0, −Δpe_rel)
        if self._pe_prev is None:
            DA = 0.0
        else:
            Δpe_rel = (pe - self._pe_prev) / (abs(self._pe_prev) + novelty_ε)
            DA = max(0.0, -Δpe_rel)
        self._pe_prev = pe
        # (4) viability x 慢 readout（τ_q）+ e_int（含 R2 死降级 {x1,x3}）
        x4 = max(0.0, (self._W范数() / self._W初始范数) - 1.0)   # ‖W‖增长（仅增长侧，SHY 缩不违规）
        x_raw = (float(节律违反度), len(self.H) / float(self.M), float(sat), x4)
        self._x慢 += (np.asarray(x_raw) - self._x慢) / float(τ_q拍)
        e_int, 降级 = self._e_int(帧新鲜)
        # (5) 5HT = e_int（消融⇒0）；回稳⇒DA 小信号
        self._5HT = 0.0 if self.消融["5HT"] else e_int
        回稳 = bool(self._上拍e_int > 0.0 and e_int == 0.0)
        DA_eff = DA + (回稳小信号 if 回稳 else 0.0)
        self.DA = 0.0 if self.消融["DA"] else DA_eff
        self.注意捕获 = bool(e_int > 0.0)
        self._上拍e_int = e_int
        # (6) NE = CUSUM(pe)（wake 侧遥测；离线相内由 离线步 驱动）
        NE突变 = self._NE步(pe)
        # (7) S ← S + novelty_t（wake 累积；run0 校准窗收集）
        self.S += novelty
        if not self.run0热:
            self._run0pe.append(pe)
            self._run0nov.append(novelty)
            if len(self._run0pe) >= run0窗:
                arr = np.asarray(self._run0pe, dtype=np.float64)
                self._μ0 = float(arr.mean())
                self._σ0 = max(float(arr.std()), 0.1 * abs(self._μ0), 1e-6)   # σ 下移（实现口径，防除零）
                self.S_crit = k_S * float(np.asarray(self._run0nov, dtype=np.float64).mean())
                self.run0热 = True
                self._run0pe = []
                self._run0nov = []
        # (8) H 写入触发：novelty_t > θ_w ∨ |v_t| > θ_v（v_t = DA_t − 5HT_t）
        v_t = self.DA - self._5HT
        写 = False
        if z is not None and (novelty > θ_w or abs(v_t) > θ_v):
            self._写入事件(z, a, z_next, v_t, novelty)
            写 = True
        # (9) C 在线局部 delta（ε_on ACh 门控；基线 ε_on=0 ⇒ 零矩阵写——快路径）
        在线写 = False
        ε_on_eff = self._E_on_eff()
        if (ε_on_eff > 0.0 and 帧新鲜 and z is not None and a is not None
                and z_next is not None):
            z32 = np.asarray(z, dtype=np.float32).ravel()
            a32 = np.asarray(a, dtype=np.float32).ravel()
            zn32 = np.asarray(z_next, dtype=np.float32).ravel()
            pre_act = z32 * a32                                    # efference 键 z⊛a
            pe_act = self.W_act @ pre_act - zn32
            self.W_act -= (ε_on_eff * np.outer(pe_act, pre_act)).astype(np.float32)
            self._W范数缓存[0] = float(np.linalg.norm(self.W_act))
            pe_free = self.W_free @ z32 - zn32
            self.W_free -= (ε_on_eff * np.outer(pe_free, z32)).astype(np.float32)
            self._W范数缓存[1] = float(np.linalg.norm(self.W_free))
            self.在线写事件 += 1
            在线写 = True
        return dict(拍=self.拍数, pe=pe, novelty=float(novelty), DA=float(self.DA),
                    NE=float(self.NE), ACh=float(self.ACh), **{"5HT": float(self._5HT)},
                    e_int=float(e_int), v=float(v_t), 写入=写, 在线写=在线写,
                    注意捕获=self.注意捕获, 回稳=回稳, 降级=降级, NE突变=NE突变,
                    S=float(self.S), S_crit=(float(self.S_crit) if self.S_crit is not None else None),
                    run0热=self.run0热)

    # ── 四通道读出（G-Layer 消费面） ──────────────────────────────────────────
    def get_四通道(self) -> dict:
        """供数接口：DA/NE/ACh/5HT（影子期；g_layer 的 ACh 代理将切换本模块真值——
        g_layer 一行不改，联动自证在 E7_selftest [9]）。O(1)。"""
        return dict(DA=float(self.DA), NE=float(self.NE),
                    ACh=float(self.ACh), **{"5HT": float(self._5HT)})

    # ── S 压力 → 离线相状态机 ────────────────────────────────────────────────
    def 压力(self) -> dict:
        """S 压力读出：S/S_crit/请求（S>S_crit 且 run0 热；未热⇒不可辨识不伪装）。"""
        请求 = bool(self.run0热 and self.S_crit is not None and self.S > self.S_crit)
        return dict(S=float(self.S), S_crit=(float(self.S_crit) if self.S_crit is not None else None),
                    请求=请求, 可辨识=bool(self.run0热))

    def 请求离线相(self) -> dict:
        """S>S_crit ⇒ 请求离线相；准入=起飞门（治理层.起飞门(S, S_crit)，E5 版 regime_ok(wake) 供数）。
        治理层缺失⇒影子期默认不放行（起飞门=g_layer 供给，诚实缺省）。"""
        p = self.压力()
        if not p["请求"]:
            理由 = ("run0 未热（S_crit 不可辨识，不伪装）" if not p["可辨识"]
                    else "S 压力不足（S≤S_crit）")
            return dict(允许=False, 请求=False, 理由=理由, 门=None)
        if self.治理层 is None:
            return dict(允许=False, 请求=True,
                        理由="无治理层供数（起飞门=g_layer 供给；影子期默认不放行）", 门=None)
        门 = self.治理层.起飞门(p["S"], p["S_crit"])
        return dict(允许=bool(门.get("允许")), 请求=True, 理由=门.get("理由", ""), 门=门)

    def 进入离线相(self) -> dict:
        """进入离线相：R_r = clip(κ_S·(S−S_crit), 5s, 600s)；计划拍=ceil(R_r/R2拍秒)；
        DA/ACh 快照供巩固选择律（重放采样）。"""
        if self._离线态 is not None:
            return dict(进入=False, 理由="已在离线相")
        p = self.压力()
        if not p["请求"]:
            return dict(进入=False, 理由="S 压力不足（S≤S_crit 或 run0 未热）")
        R_r秒 = min(max(κ_S * (p["S"] - p["S_crit"]), R_r下界秒), R_r上界秒)
        计划拍 = int(math.ceil(R_r秒 / R2拍秒))
        self._离线态 = dict(计划拍=计划拍, 已步=0, 中断=False,
                            DA快照=float(self.DA), ACh快照=float(self.ACh), 选择log=[])
        self.离线相数 += 1
        return dict(进入=True, 计划拍=计划拍, R_r秒=float(R_r秒),
                    DA快照=float(self.DA), ACh快照=float(self.ACh))

    def 离线步(self, pe: Optional[float] = None) -> dict:
        """离线相每步：NE 供数（pe 给出时更新 CUSUM）→ NE 突变⇒立即中断；否则按巩固选择律
        i~softmax((DA·v_i+ACh·novelty_i−age_i/τ_H)/T_DA) 选槽 → pe_r=C_act(a)(z)−z′ →
        C_act←C_act−ε_off·(pe_r⊗pre)（局部 delta，零反向传播）→ SHY 统一乘性因子（C 库权重；
        不动 hdc_core.W_selfHop）→ tag_i←tag_i·(1−η_f)。步满计划拍⇒自动结束（完成度=1）。"""
        st = self._离线态
        if st is None:
            return dict(离线=False, 理由="未进入离线相（起飞门未放行）")
        突变 = False
        if pe is not None:
            突变 = self._NE步(pe)
            if 突变:
                st["中断"] = True
        选择 = None
        if not 突变 and self.H:
            p = self.选择分布(st["DA快照"], st["ACh快照"])
            i = int(self._rng.choice(len(p), p=p))
            选择 = i
            st["选择log"].append(i)
            槽 = self.H[i]
            pre = 槽["z"] * 槽["a"]                                # efference 键 z⊛a
            pe_r = self.W_act @ pre - 槽["z_next"]
            self.W_act -= (self._ε_off实例 * np.outer(pe_r, pre)).astype(np.float32)
            self._W范数缓存[0] = float(np.linalg.norm(self.W_act))
            # SHY 统一乘性因子（C 库权重 W_act/W_free；hdc_core.W_selfHop 不在本模块——归属注明）
            for j, W in enumerate((self.W_act, self.W_free)):
                n = self._W范数缓存[j]
                if n > 0.0 and abs(n - self.W_target实例) > 0.0:
                    f = np.float32(self.W_target实例 / n)
                    W *= f
                    self._W范数缓存[j] = float(np.linalg.norm(W))
            槽["tag"] *= (1.0 - η_f)                               # tag 衰减
        st["已步"] += 1
        self.离线步数 += 1
        rec = dict(离线=True, 步=st["已步"], 选择=选择, NE突变=突变, 中断=突变)
        if 突变 or st["已步"] >= st["计划拍"]:
            rec.update(self.结束离线相())
        return rec

    def 结束离线相(self) -> dict:
        """睡毕：完成度=已步/计划拍；S←S·(1−完成度)（防打断销账）；治理层.记录离线相结束() 联动
        （post_replay_quiet 相位条件）。"""
        st = self._离线态
        if st is None:
            return dict(离线=False, 理由="无进行中离线相")
        完成度 = (st["已步"] / st["计划拍"]) if st["计划拍"] > 0 else 0.0
        self.S = float(self.S * (1.0 - 完成度))
        self._离线态 = None
        中断 = bool(st["中断"])
        if self.治理层 is not None and hasattr(self.治理层, "记录离线相结束"):
            self.治理层.记录离线相结束()
        return dict(离线结束=True, 完成度=float(完成度), S销账后=float(self.S),
                    中断=中断, 步数=st["已步"], 计划拍=st["计划拍"])

    def 离线占比(self) -> dict:
        """离线占比遥测（发育期目标带 [0.2,0.6]；影子期只报告不调节）。"""
        总 = self.拍数 + self.离线步数
        占比 = (self.离线步数 / 总) if 总 > 0 else 0.0
        return dict(占比=float(占比), 带=list(占比带), 越带=bool(not (占比带[0] <= 占比 <= 占比带[1])),
                    wake拍=self.拍数, 离线拍=self.离线步数)

    # ── A7 双约束接口 ────────────────────────────────────────────────────────
    def A7资格(self, δ毫秒: float, 设备ID=None) -> dict:
        """A7 双约束（τ_e=20ms 资格迹+设备 ID 分域）——影子期照顾者流 M9 未建：接口预留+未发育退化。
        资格迹 e=exp(−δ/τ_e)；δ≥40ms⇒e^{−2}≈0.135<迹写入阈⇒照顾者条件按构造不可写入 C_act
        （L3 价态注入行：转 C_free/M 的 R_care 子域——M9 未建，登记面预留）。
        设备ID=None⇒主域+未发育退化（恒当前路径）；设备ID≠None⇒R_care 分域标记（堵读取语义预留）。"""
        e = math.exp(-float(δ毫秒) / τ_e毫秒)
        return dict(资格迹=float(e), 可写C_act=bool(e >= 迹写入阈),
                    分域=("R_care" if 设备ID is not None else "主域"),
                    未发育退化=(设备ID is None), τ_e毫秒=τ_e毫秒, 迹写入阈=迹写入阈)

    # ── 判据读出面 ──────────────────────────────────────────────────────────
    def 选择分布(self, DA: Optional[float] = None, ACh: Optional[float] = None) -> np.ndarray:
        """巩固选择律分布（判据读出；手算对照用）：softmax((DA·v_i+ACh·novelty_i−age_i/τ_H)/T_DA)。"""
        d = float(self.DA) if DA is None else float(DA)
        ac = float(self.ACh) if ACh is None else float(ACh)
        if not self.H:
            return np.empty(0, dtype=np.float64)
        s = np.asarray([d * it["tag"] + ac * it["novelty"] - it["age"] / τ_H拍
                        for it in self.H], dtype=np.float64)
        return _softmax(s / T_DA)

    def 查事件pe(self, i: int) -> float:
        """V1 影子版读出：事件 i 的 efference 预测误差范数 ‖C_act(a)(z)−z′‖_2。"""
        槽 = self.H[int(i)]
        pre = 槽["z"] * 槽["a"]
        return float(np.linalg.norm(self.W_act @ pre - 槽["z_next"]))

    def C范数(self) -> dict:
        """C 库权重范数读出（SHY 收敛判据用；‖·‖_F）。"""
        return dict(act=float(np.linalg.norm(self.W_act)), free=float(np.linalg.norm(self.W_free)))

    def 列印迹(self) -> list:
        """H 印迹只读视图（tag/age/novelty/拍；向量载荷不导出——内存与判据面分离）。"""
        return [dict(拍=s["拍"], tag=float(s["tag"]), age=int(s["age"]),
                     novelty=float(s["novelty"])) for s in self.H]

    # ── 消融刀口（V4 失效矩阵；预注册接口） ──────────────────────────────────
    def 消融通道(self, 名: str) -> dict:
        """单通道静默（V4 影子版刀口）：DA/NE/ACh/5HT 各自可切除——
        DA→0（tag 恒零+softmax 均匀表型）/NE→0（中断失效）/ACh→冻结（值冻结⇒门控闭⇒C 在线写=0）/
        5HT→0（e_int 反馈断）。"""
        if 名 not in self.消融:
            raise ValueError("未知通道：%r（许可 {DA,NE,ACh,5HT}）" % (名,))
        self.消融[名] = True
        if 名 == "DA":
            self.DA = 0.0
        elif 名 == "NE":
            self.NE = 0.0
            self._NE_c = 0.0
        elif 名 == "5HT":
            self._5HT = 0.0
        return dict(通道=名, 静默=True)

    def 通道恢复(self, 名: str) -> dict:
        if 名 not in self.消融:
            raise ValueError("未知通道：%r" % (名,))
        self.消融[名] = False
        return dict(通道=名, 静默=False)

    def 巩固剥夺(self) -> dict:
        """巩固剥夺消融刀口（定稿 L3→L4 接口：ε_off=0 / H 清空——V1-B/B6 判据的剥夺臂）。"""
        self._ε_off实例 = 0.0
        n = len(self.H)
        self.H = []
        return dict(ε_off=0.0, H清空=n)
