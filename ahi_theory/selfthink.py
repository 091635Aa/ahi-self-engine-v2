# -*- coding: utf-8 -*-
"""ahi_theory/selfthink.py —— M15 · 自我思考层（影子模式）：自省循环内核 + 外显自主推理

预登记：ahi_theory/verify_all.py `_CHANGELOG` 2026-09-30『M15 自我思考层预登记』（先于改码）。
设计定稿：meeting15/00_自我思考会议纪要.md（第 15 次会议五专家 MoE 两轮收敛）。
规格出处：纪要 §2.1 自省代数定义（E3 终稿）/ §2.2 触发矩阵（E4 终稿）/ §2.3 存储裁决 /
§2.4 反哺通路表 / §2.5 外显推理 / §2.6 工程执行栈 / §2.7 发育门 / §4 参数表。

影子模式纪律（红线）：
  · 纯新增；**默认不加载**（无模块级副作用、无任何内核接线——消费方显式 import 才实例化）；
  · **不 import brain/world/system/three_loop/mbp**（影子纪律红线）；只读消费构造传入的
    HDC核心 实例（cleanup/unbind 代数、W_selfHop、IM、占据链、置换 ρ、M 读路径——全部
    只读访问，**不改 hdc 任何状态、不增 IM、不建井**）；
  · **零拟合红线**：全文件无 lstsq / linalg.inv·pinv / solve / scipy.optimize / 参数搜索构造
    （C4 审计器 tokenize 级可检，meeting15/SELF_selftest [2] 自证）；近并列下四分位门与
    滞回阈均为预注册统计量（窗口与定义写死，封死后调参通道——纪要 §5#2）；
  · **账本纪律**：登记自由度不登记自由度的解——预注册常数（SELF_SEED/W_NARR/FIFO/查询链
    步数/λ 带与速率闸/统计窗/L 与候选 m/作业预算/快照上限/滞回/熔断双门/立档阈/枚举标记）
    构造时以 kind="参数变更" 入账本（键前缀 "自省_"）；清单扩充走 ledger 预授权路径
    （「扩充清单本身=结构事件」，ledger.py 源文件零改动——personality 同律）；
  · **键域**：SELF_SEED=3072 单 rng 顺序派生自省角色键族（R_intro_act/R_intro_err/
    R_intro_val——影子期保留未消费、接口预注册，供阶段 B 反哺通道 b/c 结论绑定）；
    **不动 hdc_core 的 MASTER_SEED=1024 与 personality 的 PERS_SEED=2048 消耗序**；
    纪要 §2.1 的 S_introp filler 键全部复用 hdc 既有键（R_act/R_q/R_err/R_g/R_care/R_val），
    自省键族只补缺；
  · 人格≠活性证据同律扩展：**严禁以自省产出反证 R2 活性**（纪要 §2.6 kill 交互条款）；
  · 诚实边界：本模块全部为**影子期机制自证**；引擎级真值（长时程发育数据、F1–F3/INT1–INT2/
    RT6-a-b 判据测量、六 seed 个体）待 V2 引擎供数另行预注册接入，不虚设（纪要 §5#6）；
    观察者效应（自省 episode 改变被测系统）以枚举标记隔离，K 判据剔除自省驻留窗（纪要 §5#1）。

机制总览（纪要 §1 定稿总览）：
  R2 尾片（巩固后）─触发矩阵─▶ 自省作业（查询链 5 步 + 自传体叙事 N_auto）─▶ S_introp 自省迹
                                │                                │
                    ┌───────────┴───────────┐          立档账本（阶段A）/ 反哺（阶段B）
                    ▼                       ▼
              叙事快照（链式 hash）      外显推理（step 6 扩展：近并列 lookahead L∈{2,3}）

自省代数（纪要 §2.1，零新参数、零拟合——全走 hdc 既有键/cleanup 门 θ=0.5/M 读路径）：
  Q_act = cleanup(unbind(S̃,R_act))「我做了什么」（D1 未通过 ⇒ R_act 分量前置零=未发育查询退化，
          与 hdc step7 同律）；Q_q = unbind(S̃,R_q)「我感觉如何」；
  Q_err = unbind(S̃,R_err)「哪里异常」；Q_mem = cleanup(M@(a⊛H))「我记得什么」；
  Q_self = IM[k*]「我在哪个井」。
  N_auto = sign(Σ_{τ∈W} ρ^{t−τ}·(IM[k*(τ)]⊛R_act + 𝟙[e_τ 局部峰]·IM[k*(τ)]⊛R_err
                                  + 𝟙[v_τ 局部峰]·IM[k*(τ)]⊛R_val))   （ρ 既作角色置换又作时间轴）
  S_introp = sign(Q_act⊛R_act + Q_q⊛R_q + Q_err⊛R_err + Q_self⊛R_g + N_auto⊛R_care)
  局部峰 = 3 点局部极大（严格大于两侧邻；窗边界不判峰；NaN 邻不判峰）。
  求和实现 = Horner（acc=ρ(acc)+v_τ，旧→新）：因 ρ 为精确指标置换（逐元素、无舍入），
  与显式 ρ^p 幂次按同一（旧→新）顺序求和在 float64 下**逐位等价**（SELF_selftest [4][7] 独立
  重构路径实证 cos=1.0）。IM[k*(τ)] 取 episode 时点 IM 现值（井写入后不被修改，驱逐失效⇒零项）。

触发矩阵（纪要 §2.2，E4 终稿）：
  自省触发 = 公共工程闸 ∧ 分型源 ∧ 速率闸
    公共闸：R2 队列空 ∧ 无未决 R1 请求 ∧ 非离线相 ∧ consolidation_permit 互锁（与巩固/离线
            重放同窗互斥）∧ 滞回（空闲型 2 R2 基拍/失配型 1 基拍=更短，预注册）
    空闲型 = e_t < 滚动中位 ∧ 无警戒带（S_警戒 ∨ viability 越界均否）⇒ 叙事型产出
    失配型 = S_警戒 ∨ viability 越界 ∨ NE 突变 ∨ e_t > 高分位 ⇒ 内容型产出（事件驱动，D-5 仲裁序，
            与空闲型同现时失配优先）
    速率闸 = 距上次自省 ≥ 100 R1 拍（=1/λ带上界 1e-2；带 [1e-3,1e-2]，λ_target=5e-3/R1 拍）
存储（纪要 §2.3，|IM|≤150 红线下）：不增 IM、不建井；S_introp 以 tag='introp' 入**本模块自持的
  影子快照列表**（hdc_core 一行不许改——hdc 快照缓冲不可写入，影子期等价承载，逐出只损档龄
  不损叙事代数）；新增 20 槽 FIFO 环缓冲（每 R1 拍记 k*/e/v/a星序号，环满覆写，预分配 O(1)
  无分配）承载 N_auto 有序历史；立档账本记 (t, cos(S_introp, IM[k*])) 命中率（阶段 A 契约，
  命中阈=cleanup 同阈 0.5）。
外显推理（纪要 §2.5，step 6 扩展）：一级 G(a) 原式不动（归 hdc）；近并列=top1−top2 G 评分差
  落入 100 R1 拍滚动下四分位（统计窗未满 ⇒ 统计量不可用⇒不触发，诚实）；触发后仅 top-m=2
  候选做 step8 同款弛豫仿真（缺省用 hdc.W_selfHop 只读复制：x̂_a=sign(M@(av_a⊛H)) 起、C=5 次
  弛豫，confidence(a)=max_k cos(S̃_a,IM[k])；仿真函数可由调用方注入）；confidence 重排破并列，
  翻转 argmin ⇒ 犹豫签名；深度 L∈{2,3}（L=2 默认；L=3 仅跨井结构事件——候选落井彼此相异——
  升格再加 C 次弛豫；k=4（候选 m>2）拒绝入环，断言强制，降格环外一次性诊断）；自我纠正=
  执行后 e_t>滚动中位 ∧ k*≠k̃ ⇒ 剔除已试候选按 G 重选（换向朝更小 G）。
结论三态分类（纪要 §2.4/§2.6）：A 快量（B7 校准/行为参数，适用 5s 三态龄期）/ B 慢量
  （viability 目标带/θ_w 解位移，不适用龄期走账本 WAL，仅阶段 B）/ C 纯读出（自传体 tag，
  仅入叙事日志）；未知通道 ⇒ ValueError（诚实拒绝，不默认归 C）。
熔断协议（纪要 §3 E5 终稿）：双门任一即熔——门1 λ 连续 2 窗越带（>1e-2 反刍 或 <1e-3 停摆）；
  门2 体制切换后 Jaccard(触发构型)<0.5；动作=反哺关闭回退阶段 A+episode 置非发育位继续运行；
  恢复=连续 2 窗 λ 回带 ∧ Jaccard≥0.5 ∧ 最短驻留 1 窗（防乒乓）。
阶段纪律（纪要 §2.4）：阶段 A 立档（只记命中率账本）；阶段 B 需命中率 > 主井占比（Π 既有机会
  水平）且 M14 阶段 B 已启用（人格剖面器.stage=="B"）——默认关，接口预注册（阶段B晋升判定）。
WCET（纪要 §4）：喂R1 O(1) 无分配；自省episode O(查询链+叙事窗求和+C 次 1024² 弛豫)；
  lookahead O(每候选 1 次 M 读+L×C 次 1024² 弛豫)——预算断言在 meeting15/SELF_selftest [10]
  （喂R1 ≤0.1ms / episode ≤2.0ms / L=2 ≤5.0ms / L=3 ≤10.0ms，均 ≪ 作业预算 6 R2 基拍=1.2s）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np
# 影子纪律许可的类型/常量引用（ahi_theory.hdc_core / ahi_theory.personality 均在红线许可清单；
# 人格谱系同构复用=不设第三谱系防身份域分裂，纪要 §2.6；personality.py 源文件零改动）
from ahi_theory.hdc_core import C_弛豫, D_VSA, THETA_CLEANUP, 快照缓冲容量
from ahi_theory.personality import canonical, 人格谱系

__all__ = ["自省核心", "叙事谱系", "SELF_SEED", "W_NARR", "FIFO槽位", "查询链步数",
           "LAMBDA_TARGET", "λ带", "速率闸间隔R1拍", "近并列统计窗", "e统计窗",
           "L_DEFAULT", "L_MAX", "候选m上限", "作业预算基拍", "快照KB上限",
           "滞回空闲基拍", "滞回失配基拍", "熔断越带窗数", "JACCARD阈", "最短驻留窗",
           "恢复回带窗", "立档命中阈", "立档期窗数", "高分位q", "标记枚举", "非发育标记",
           "发育经历标记", "结论三态映射", "注册键数"]

# ── §0 预注册常量（纪要 §4 参数表；变更 = kind="参数变更" 账本结构事件，键前缀 "自省_"） ──
SELF_SEED = 3072              # 自省角色键族单 rng 种子（R_intro_* 顺序派生；不动 hdc 1024/
                              # personality 2048 消耗序——三 seed 域永不相交）
W_NARR = 20                   # 自传体叙事窗（R1 拍；峰=3 点局部极大判据）
FIFO槽位 = 20                 # FIFO 环缓冲槽位（每 R1 拍记 k*/e/v/a星序号，环满覆写）
查询链步数 = 5                # Q_act/Q_q/Q_err/Q_mem/Q_self 固定 5 步
LAMBDA_TARGET = 5e-3          # λ_target / R1 拍（E5 口径：experience-sampling 自发走神分钟量级；
                              # R2 尾片是允许时钟域不是频率承诺）
λ带 = (1e-3, 1e-2)            # λ 频率带（越带连续 2 窗⇒熔断门1）
速率闸间隔R1拍 = 100           # =1/λ带上界 1e-2（按带下界口径预注册写死，封死调参）
近并列统计窗 = 100            # G 评分差滚动下四分位窗（R1 拍）
e统计窗 = 100                 # e 滚动中位/高分位统计窗（自我纠正与空闲型同律承载）
L_DEFAULT = 2                 # lookahead 深度默认（2 候选×C=5 次弛豫）
L_MAX = 3                     # 深度上限（仅跨井结构事件升格）；k=4（m>2）拒绝入环（断言）
候选m上限 = 2                 # top-m 候选上限（破并列只需一阶置信差，非全局规划）
作业预算基拍 = 6              # 自省作业预算=6 R2 基拍=1.2s（超 2s 无进展⇒登记降级，纪要 §4）
快照KB上限 = 16               # 叙事快照 ≤16KB（canonical 序列化字节数断言）
滞回空闲基拍 = 2              # 滞回 2 R2 基拍（空闲型）
滞回失配基拍 = 1              # 失配型滞回更短（事件驱动，D-5 仲裁序）
熔断越带窗数 = 2              # 门1：λ 连续 2 窗越带
JACCARD阈 = 0.5               # 门2：体制切换后 Jaccard(触发构型)<0.5 即熔
最短驻留窗 = 1                # 熔断最短驻留（防乒乓）
恢复回带窗 = 2                # 恢复=连续 2 窗 λ 回带 ∧ Jaccard≥0.5 ∧ 驻留满足
立档命中阈 = THETA_CLEANUP    # 立档命中率阈值=cleanup 同阈 0.5（单一来源防漂移）
立档期窗数 = 3                # 阶段 A 立档 ≥3 窗（同 M14；机制级以立档样本数承载，
                              # W_pers 窗口径待 V2 供数）
高分位q = 0.9                 # e 高分位（失配型 e_t>高分位；滚动 e 统计窗上分位，预注册）

# 枚举标记（纪要 §2.6 E4 表，替代双 bit）：1–4 非发育（不进 Π、不计发育史），5 属发育经历
标记枚举 = {0: "常态发育流", 1: "行为门施测", 2: "人格施测D6", 3: "演练",
            4: "诱发自省D7", 5: "自发自省"}
非发育标记 = frozenset({1, 2, 3, 4})
发育经历标记 = frozenset({0, 5})

# 结论三态分类（纪要 §2.4 反哺通路表 × §2.6 三态）：未知通道 ⇒ ValueError（诚实拒绝）
结论三态映射 = {"自传体tag": "C纯读出",          # 通道 a：IM 条目 tag 字段扩展（阶段 A 即可）
                "θ_w解位移": "B慢量",            # 通道 b：W_selfHop(θ_w)（仅阶段 B，WAL 先行）
                "viability目标带": "B慢量",      # 通道 c：目标带（仅阶段 B，WAL 结构事件）
                "B7校准": "A快量",               # B7 校准量（5s 三态龄期）
                "行为参数": "A快量"}             # 行为层参数（5s 三态龄期）


def _注册键值表():
    """预注册常数表（构造时逐条 kind="参数变更" 入账本；变更须再次结构事件）。"""
    return (
        ("自省_SELF_SEED", SELF_SEED,
         "M15 #69 预注册初值：自省角色键族单 rng 种子（R_intro_* 顺序派生；"
         "独立于 hdc MASTER_SEED=1024 / personality PERS_SEED=2048 消耗序）"),
        ("自省_W_NARR", W_NARR,
         "M15 #69 预注册初值：自传体叙事窗 20 R1 拍（ρ 时间轴；峰=3 点局部极大）"),
        ("自省_FIFO槽位", FIFO槽位,
         "M15 #69 预注册初值：FIFO 环缓冲 20 槽（每 R1 拍记 k*/e/v/a星序号，环满覆写）"),
        ("自省_查询链步数", 查询链步数,
         "M15 #69 预注册初值：查询链固定 5 步（Q_act/Q_q/Q_err/Q_mem/Q_self，零新参数）"),
        ("自省_λ_target与带", (LAMBDA_TARGET, λ带[0], λ带[1]),
         "M15 #69 预注册初值：λ_target=5e-3/R1 拍、带 [1e-3,1e-2]"
         "（experience-sampling 口径；R2 尾片是允许时钟域不是频率承诺）"),
        ("自省_速率闸间隔R1拍", 速率闸间隔R1拍,
         "M15 #69 预注册初值：速率闸最小间隔 100 R1 拍（=1/λ带上界 1e-2，写死封死调参）"),
        ("自省_近并列统计窗R1拍", 近并列统计窗,
         "M15 #69 预注册初值：G 评分差滚动下四分位统计窗 100 R1 拍"),
        ("自省_e统计窗R1拍", e统计窗,
         "M15 #69 预注册初值：e 滚动中位/高分位统计窗 100 R1 拍（空闲型/自我纠正承载）"),
        ("自省_lookaheadL与候选m", (L_DEFAULT, L_MAX, 候选m上限),
         "M15 #69 预注册初值：lookahead L∈{2,3}（L=2 默认/3 跨井升格）、top-m=2；"
         "k=4 拒绝入环（断言强制，147 MFLOP 树上界失控不可预注册 WCET）"),
        ("自省_作业预算基拍", 作业预算基拍,
         "M15 #69 预注册初值：自省作业预算 6 R2 基拍=1.2s（超 2s 无进展⇒登记降级）"),
        ("自省_快照KB上限", 快照KB上限,
         "M15 #69 预注册初值：叙事快照 ≤16KB（canonical 字节断言；人格谱系 64KB 同构收窄）"),
        ("自省_滞回基拍", (滞回空闲基拍, 滞回失配基拍),
         "M15 #69 预注册初值：滞回 空闲型 2 / 失配型 1 R2 基拍（失配型更短，事件驱动）"),
        ("自省_熔断双门", (熔断越带窗数, JACCARD阈, 最短驻留窗, 恢复回带窗),
         "M15 #69 预注册初值：熔断双门（λ 连续 2 窗越带 ∨ Jaccard<0.5）+最短驻留 1 窗"
         "+恢复连回带 2 窗（防乒乓）"),
        ("自省_立档命中阈与立档窗数", (立档命中阈, 立档期窗数),
         "M15 #69 预注册初值：立档命中阈=cleanup 同阈 0.5；阶段 A 立档 ≥3 窗"
         "（命中率 > 主井占比 ∧ M14 阶段 B 已启用方晋升，默认关）"),
        ("自省_枚举标记", {str(k): v for k, v in 标记枚举.items()},
         "M15 #69 预注册：枚举标记 0–5（0 常态发育流/1 行为门施测/2 人格施测 D6/3 演练/"
         "4 诱发自省 D7/5 自发自省；1–4 非发育，5 属发育经历）"),
    )


注册键值表 = _注册键值表()
注册键数 = len(注册键值表)


def _双极(rng: np.random.Generator, n: int) -> np.ndarray:
    """双极 ±1 向量（dtype=float32；⊛=逐元素乘即 XNOR 保持律，与 hdc_core._双极 同构）。"""
    return np.sign(rng.standard_normal(n)).astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """余弦（零向量⇒0.0；与 hdc_core._cos 同构）。"""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── §1 叙事谱系（人格谱系同构复用，16KB 收窄；不设第三谱系） ────────────────────────
class 叙事谱系(人格谱系):
    """叙事快照追加-only 链式 hash 谱系——**直接继承** personality.人格谱系 的链式 hash
    本体（H_i = sha256(H_{i-1} ‖ canonical(快照_i) ‖ seq_i ‖ 标记_i)），personality.py
    源文件零改动；本类只做两件事：
      ① 快照上限收窄覆盖 64KB→16KB（纪要 §4：叙事快照 ≤16KB，先断言再入链）；
      ② 事件标记缺省 "自省:*"（与人格条目同域追加、以标记区分——纪要 §2.6「与人格谱系
         交叉引用（不设第三谱系）防身份域分裂」的落地：叙事条目就在人格谱系链上）。"""

    def 追加(self, 快照dict, 事件标记: str = "自省") -> dict:
        文 = canonical(快照dict).encode("utf-8")
        if len(文) > 快照KB上限 * 1024:
            raise AssertionError("叙事快照 %d 字节 > %dKB 上限（M15 #69 预注册）"
                                 % (len(文), 快照KB上限))
        return super().追加(快照dict, 事件标记)


# ── §2 自省核心 ──────────────────────────────────────────────────────────────
class 自省核心:
    """自省循环内核 + 外显自主推理（影子期不接管控制流；消费 HDC核心/人格剖面器 引用与导出量）。

    O(1) 纪律：喂R1 只写预分配环（20 槽 FIFO + 100 槽 G 差环 + 100 槽 e 环），无分配；
    近并列分位数/中位数只在 R2 决策点（近并列判定/自我纠正）按需计算，不进 R1 热路径。
    """

    def __init__(self, hdc, 人格=None, 账本=None, 种子: int = SELF_SEED):
        self._hdc = hdc
        self.人格 = 人格
        self.账本 = 账本
        self.种子 = int(种子)
        rng = np.random.default_rng(self.种子)          # 单 rng 顺序派生（预注册顺序，不挑）
        # 自省角色键族（只补缺：S_introp filler 键全复用 hdc 既有键，纪要 §2.1；
        # 影子期保留未消费——阶段 B 反哺通道 b/c 结论绑定接口预注册）
        self.R_intro_act = _双极(rng, D_VSA)
        self.R_intro_err = _双极(rng, D_VSA)
        self.R_intro_val = _双极(rng, D_VSA)
        # 20 槽 FIFO 环（预分配，环满覆写，O(1) 无分配）：每 R1 拍记 (k星, e_t, v_t, a星序号)
        self._环k = np.full(FIFO槽位, -1, dtype=np.int64)
        self._环e = np.full(FIFO槽位, np.nan, dtype=np.float64)
        self._环v = np.zeros(FIFO槽位, dtype=np.float64)
        self._环a = np.zeros(FIFO槽位, dtype=np.int64)
        self._环头 = 0                                  # 下一写入槽
        self._环满 = 0                                  # 已填充数（≤FIFO槽位）
        # 近并列滚动统计器（100 拍 G 评分差下四分位；预分配环）
        self._G窗 = np.zeros(近并列统计窗, dtype=np.float64)
        self._G头 = 0
        self._G计 = 0
        # e 滚动统计窗（中位/高分位；预分配环）
        self._e窗 = np.full(e统计窗, np.nan, dtype=np.float64)
        self._e头 = 0
        self._e计 = 0
        # 状态与账
        self.拍数 = 0                                   # R1 拍计数（喂R1 推进）
        self._上次自省拍 = -1                            # 速率闸起点（-1=从未自省）
        self._滞回剩余 = 0                              # 滞回计数（R2 基拍）
        self.stage = "A"                                # 阶段 A 立档（只记命中率账本）
        self._introp影子: list = []                      # 影子快照列表 tag='introp'（自持，不碰 hdc）
        self._立档账: list = []                          # (t, cos(S_introp, IM[k*])|None)
        self._熔断中 = False
        self._熔断驻留窗 = 0
        self.叙事 = 叙事谱系()                           # 叙事谱系（人格谱系同构链）
        self._最近链: dict = {}                          # 最近 episode 查询链存证（判据/自检消费）
        self._参数预注册入账()

    # —— 构造辅助 ——
    def _参数预注册入账(self) -> None:
        """预注册常量以 kind="参数变更" 入账本（键前缀 "自省_"）；账本 None ⇒ 跳过。
        变更清单扩充走 ledger 预授权路径：「扩充清单本身 = 结构事件」（ledger.py 文档
        预授权；运行时模块对象扩充，ledger.py 源文件一行不改——personality 同律）。"""
        if self.账本 is None:
            return
        try:
            from ahi_theory import ledger as _ledger
        except Exception:
            _ledger = None
        if _ledger is not None and hasattr(_ledger, "变更清单"):
            新键 = [k for k, _, _ in 注册键值表 if k not in _ledger.变更清单]
            if 新键:
                self.账本.登记("结构事件",
                               payload={"事件": "变更清单扩充", "键族": "自省_", "新增键": 新键},
                               理由="M15 #69 预注册：自省参数键族入清单"
                                    "（ledger 预授权：扩充清单本身=结构事件；运行时扩充，"
                                    "ledger.py 源文件零改动）")
                _ledger.变更清单 = _ledger.变更清单 | frozenset(新键)
        for 键, 值, 理由 in 注册键值表:
            self.账本.登记("参数变更", payload={"键": 键, "值": 值},
                           理由=理由 + "（先验不调；变更须再次结构事件）")

    # —— R1 侧：O(1) 环更新 ——
    def 喂R1(self, e_t, a星序号, v_t: float = 0.0, G差: Optional[float] = None) -> None:
        """每 R1 拍一次：FIFO 环 + e 统计窗 + G 差环 各 O(1) 推进（无分配）。
        k星 只读取自 hdc.占据链尾（R2 维护低频 append；两次维护间保持驻留值——确定性）。
        e_t=None（丢帧/stale）⇒ 环记 NaN（不判峰、不入统计）。G差=top1−top2 G 评分差幅度
        （≥0，调用方自 hdc 步进R1 的 G表 计算）；None⇒G 环不推进。"""
        if self._hdc.占据链:
            k星 = int(self._hdc.占据链[-1])
        else:
            k星 = -1
        i = self._环头
        self._环k[i] = k星
        self._环e[i] = np.nan if e_t is None else float(e_t)
        self._环v[i] = float(v_t)
        self._环a[i] = int(a星序号)
        self._环头 = (i + 1) % FIFO槽位
        if self._环满 < FIFO槽位:
            self._环满 += 1
        j = self._e头
        self._e窗[j] = np.nan if e_t is None else float(e_t)
        self._e头 = (j + 1) % e统计窗
        if self._e计 < e统计窗:
            self._e计 += 1
        if G差 is not None:
            g = self._G头
            self._G窗[g] = float(G差)
            self._G头 = (g + 1) % 近并列统计窗
            if self._G计 < 近并列统计窗:
                self._G计 += 1
        self.拍数 += 1

    def _环序(self) -> list:
        """环内时序（旧→新）的槽位下标。"""
        n = self._环满
        头 = self._环头
        return [(头 - n + k) % FIFO槽位 for k in range(n)]

    def _e现值(self) -> Optional[float]:
        """最新环 e（NaN/空 ⇒ None）。"""
        if self._环满 == 0:
            return None
        e = self._环e[(self._环头 - 1) % FIFO槽位]
        return None if np.isnan(e) else float(e)

    def e滚动中位(self) -> Optional[float]:
        """e 统计窗滚动中位（NaN 剔除；样本 0 ⇒ None）。"""
        有 = self._e窗[:self._e计] if self._e计 < e统计窗 else self._e窗
        有 = 有[~np.isnan(有)]
        return float(np.median(有)) if 有.size else None

    def 滚动高分位(self, q: float = 高分位q) -> Optional[float]:
        """e 统计窗滚动高分位（缺省 q=0.9 预注册；NaN 剔除；样本 0 ⇒ None）。"""
        有 = self._e窗[:self._e计] if self._e计 < e统计窗 else self._e窗
        有 = 有[~np.isnan(有)]
        return float(np.quantile(有, q)) if 有.size else None

    # —— R2 决策点：触发矩阵（纪要 §2.2） ——
    def 触发判定(self, e_滚动中位: float, e_高分位: Optional[float] = None,
                 S_警戒: bool = False, viability_越界: bool = False,
                 NE_突变: bool = False, 队列空: bool = True, 无未决R1: bool = True,
                 非离线: bool = True, consolidation_permit: bool = True) -> Optional[str]:
        """自省触发 = 公共工程闸 ∧ 分型源 ∧ 速率闸（每 R2 基拍调用一次；返回
        "空闲型"/"失配型"/None）。滞回内置计数（每调用推进一基拍；空闲 2/失配 1）。
        失配型与空闲型同现 ⇒ 失配优先（D-5 仲裁序）。触发即置速率闸起点与滞回——
        调用方应随即执行 自省episode()（被抢占作废重排，D-5 同律；中断也是自省史）。"""
        if self._滞回剩余 > 0:                          # 滞回（公共闸组成；先推进计数）
            self._滞回剩余 -= 1
            return None
        if not (队列空 and 无未决R1 and 非离线 and consolidation_permit):
            return None                                  # 公共工程闸不满足
        if self._上次自省拍 >= 0 and (self.拍数 - self._上次自省拍) < 速率闸间隔R1拍:
            return None                                  # 速率闸（最小间隔 100 R1 拍）
        e_now = self._e现值()
        if e_now is None or e_滚动中位 is None:
            return None                                  # 无 e 可判（诚实不触发）
        失配 = bool(S_警戒 or viability_越界 or NE_突变
                    or (e_高分位 is not None and e_now > e_高分位))
        空闲 = (not 失配) and (e_now < e_滚动中位) and not (S_警戒 or viability_越界)
        if 失配:
            分型 = "失配型"
        elif 空闲:
            分型 = "空闲型"
        else:
            return None
        self._上次自省拍 = self.拍数
        self._滞回剩余 = 滞回失配基拍 if 失配 else 滞回空闲基拍
        return 分型

    # —— 自省作业：查询链 + 自传体叙事 + S_introp（纪要 §2.1） ——
    def 自省episode(self, 触发源: str = "自发自省", 标记: int = 5):
        """查询链 5 步（全走 hdc cleanup/unbind 代数，零新参数）→ N_auto（FIFO 环+ρ 时间轴，
        Horner 求和）→ S_introp → 影子快照（tag='introp'，自持不碰 hdc）→ 立档记录
        (t, cos(S_introp,IM[k*])|None) → 叙事快照入谱系（链式 hash，≤16KB 断言）。
        返回 (S_introp, 立档记录)。标记∈枚举标记（缺省 5=自发自省，属发育经历）。
        只读 hdc：S/W_selfHop/IM/占据链/H/动作候选/R_*/置换/M 读路径，状态零改动。"""
        hdc = self._hdc
        # S̃ 重构（step8 同款：C 次弛豫；只读复制）
        S̃ = np.sign(hdc.S).astype(np.float32)
        for _ in range(C_弛豫):
            S̃ = np.sign(hdc.W_selfHop @ S̃)
        k星 = int(hdc.占据链[-1]) if hdc.占据链 else -1
        # 查询链（未发育退化：D1 前 Q_act 权重前置零——与 hdc step7 同律联动）
        if hdc.D1通过:
            Q_act = hdc._cleanup(S̃ * hdc.R_act)
        else:
            Q_act = None
        Q_q = S̃ * hdc.R_q
        Q_err = S̃ * hdc.R_err
        a序 = int(self._环a[(self._环头 - 1) % FIFO槽位]) if self._环满 else 0
        键 = hdc._动作向量(hdc.动作候选[a序]) * np.sign(hdc.H)
        Q_mem = hdc._cleanup(hdc._M读(键.reshape(D_VSA, 1))[:, 0].astype(np.float32))
        Q_self = (hdc.IM[k星]["井"].astype(np.float32)
                  if 0 <= k星 < len(hdc.IM) else np.zeros(D_VSA, dtype=np.float32))
        # 自传体叙事 N_auto（ρ 既作角色置换又作时间轴；Horner 旧→新，与显式 ρ^p 求和逐位等价）
        e序 = [self._环e[i] for i in self._环序()]
        v序 = [self._环v[i] for i in self._环序()]
        k序 = [int(self._环k[i]) for i in self._环序()]
        峰e = self._局部峰(e序)
        峰v = self._局部峰(v序)

        def _井(k: int) -> np.ndarray:
            return (hdc.IM[k]["井"].astype(np.float64)
                    if 0 <= k < len(hdc.IM) else np.zeros(D_VSA, dtype=np.float64))

        acc = np.zeros(D_VSA, dtype=np.float64)
        for idx in range(len(k序)):
            井v = _井(k序[idx])
            项 = 井v * hdc.R_act.astype(np.float64)
            if 峰e[idx]:
                项 = 项 + 井v * hdc.R_err.astype(np.float64)
            if 峰v[idx]:
                项 = 项 + 井v * hdc.R_val.astype(np.float64)
            acc = hdc._ρ(acc) + 项                       # ρ=精确指标置换：Horner≡显式幂次和
        N_auto = np.sign(acc).astype(np.float32)
        # S_introp（filler 全是对自身的查询，二阶绑定即「自我」）
        z = ((Q_act if Q_act is not None else np.zeros(D_VSA, dtype=np.float32)).astype(np.float64)
             * hdc.R_act + Q_q.astype(np.float64) * hdc.R_q
             + Q_err.astype(np.float64) * hdc.R_err + Q_self.astype(np.float64) * hdc.R_g
             + N_auto.astype(np.float64) * hdc.R_care)
        S_introp = np.sign(z).astype(np.float32)
        # 影子快照（tag='introp'；自持列表，逐出最旧只损档龄不损叙事代数——不碰 hdc）
        self._introp影子.append({"拍": self.拍数, "S": S_introp.copy(), "tag": "introp"})
        if len(self._introp影子) > 快照缓冲容量:
            del self._introp影子[0]
        # 立档记录（阶段 A 契约：只记命中率账本）
        立档 = (self.拍数,
                (_cos(S_introp, hdc.IM[k星]["井"]) if 0 <= k星 < len(hdc.IM) else None))
        self._立档账.append(立档)
        # 叙事快照（链式 hash 入谱系；≤16KB 断言在 叙事谱系.追加）
        def _vhash(v) -> str:
            return hashlib.sha256(np.ascontiguousarray(
                v, dtype=np.float32).tobytes()).hexdigest()[:16]

        步hash = [_vhash(Q_act if Q_act is not None else np.zeros(D_VSA, np.float32)),
                  _vhash(Q_q), _vhash(Q_err),
                  _vhash(Q_mem if Q_mem is not None else np.zeros(D_VSA, np.float32)),
                  _vhash(Q_self), _vhash(N_auto)]
        尾 = min(5, self._环满)
        快照 = dict(
            触发源=str(触发源),
            输入摘要=dict(拍=int(self.拍数),
                          k星尾=[int(k序[len(k序) - 尾 + i]) for i in range(尾)],
                          e尾=[(None if np.isnan(e序[len(e序) - 尾 + i])
                                else float(e序[len(e序) - 尾 + i])) for i in range(尾)]),
            推理链步骤hash=步hash,
            结论向量标记=hashlib.sha256(np.ascontiguousarray(
                S_introp, dtype=np.float32).tobytes()).hexdigest()[:16],
            置信=立档[1],
            标记=int(标记))
        self.叙事.追加(快照, 事件标记="自省:" + str(触发源))
        # 查询链存证（判据/自检消费； episode 本身可重入——确定性重建）
        self._最近链 = dict(S̃=S̃, k星=k星,
                            Q_act=(Q_act if Q_act is not None
                                   else np.zeros(D_VSA, dtype=np.float32)),
                            Q_act退化=(Q_act is None), Q_q=Q_q, Q_err=Q_err,
                            Q_mem=(Q_mem if Q_mem is not None
                                   else np.zeros(D_VSA, dtype=np.float32)),
                            Q_self=Q_self, N_auto=N_auto, S_introp=S_introp,
                            峰e=峰e, 峰v=峰v)
        self._上次自省拍 = self.拍数                   # 诱发调用（未经触发判定）同样计入速率闸
        return S_introp, 立档

    @staticmethod
    def _局部峰(序列: list) -> list:
        """3 点局部极大（严格大于两侧邻；边界/NaN 邻不判峰）。"""
        n = len(序列)
        out = [False] * n
        for i in range(1, n - 1):
            a, b, c = 序列[i - 1], 序列[i], 序列[i + 1]
            if np.isnan(a) or np.isnan(b) or np.isnan(c):
                continue
            out[i] = (b > a) and (b > c)
        return out

    # —— 外显推理（step 6 扩展，纪要 §2.5） ——
    def 近并列判定(self, G表) -> tuple:
        """(是否近并列, 候选序)：本拍 top1−top2 G 评分差 ≤ 100 拍滚动下四分位 ⇒ 近并列
        （统计量非拟合，窗口与定义预注册封死后调参）；统计窗未满 ⇒ 统计量不可用 ⇒
        (False, [])（诚实不触发）。候选序=按 G 升序的前 top-m=2 序号。"""
        if self._G计 < 近并列统计窗:
            return (False, [])
        g = np.sort(np.asarray(G表, dtype=np.float64).ravel())
        if g.size < 2:
            return (False, [])
        差 = float(g[1] - g[0])
        q1 = float(np.percentile(self._G窗, 25))
        序 = [int(i) for i in np.argsort(g)[:候选m上限]]
        return (差 <= q1, 序)

    def _弛豫仿真(self, a序号: int) -> np.ndarray:
        """缺省弛豫仿真（W_selfHop 只读复制）：x̂_a=sign(M@(av_a⊛H)) 起、C=5 次弛豫
        （step8 同款；hdc 状态零改动）。"""
        hdc = self._hdc
        av = hdc._动作向量(hdc.动作候选[int(a序号)])
        键 = av * np.sign(hdc.H)
        S̃ = np.sign(hdc._M读(键.reshape(D_VSA, 1))[:, 0]).astype(np.float32)
        for _ in range(C_弛豫):
            S̃ = np.sign(hdc.W_selfHop @ S̃)
        return S̃

    def 推理lookahead(self, G表, 候选键仿真回调=None, L: Optional[int] = None,
                      m: int = 候选m上限) -> Optional[dict]:
        """近并列 ⇒ 二级 lookahead：仅 top-m=2 候选做弛豫仿真，confidence(a)=max_k
        cos(S̃_a,IM[k]) 重排破并列；重排 top1 ≠ 原 G argmin ⇒ 犹豫签名 True。
        L=None ⇒ 自适应（L_DEFAULT=2 起；候选落井彼此相异=跨井结构事件 ⇒ 升格 L_MAX=3，
        各候选再续 C 次弛豫）；显式 L∈{2,3}。m≠2（如 k=4）⇒ AssertionError（拒绝入环，
        147 MFLOP 树上界失控不可预注册 WCET，降格环外一次性诊断）。非近并列 ⇒ None。"""
        assert int(m) == 候选m上限, "k=%d 拒绝入环（M15 #69 预注册：top-m 固定 2，" \
                                    "树上界失控不可预注册 WCET；降格环外一次性诊断）" % int(m)
        assert L is None or L in (L_DEFAULT, L_MAX), \
            "L=%r ∉ {2,3} 拒绝入环（M15 #69 预注册断言）" % (L,)
        近并列, 候选序 = self.近并列判定(G表)
        if (not 近并列) or len(候选序) < 2:
            return None
        hdc = self._hdc
        S̃表 = {}
        conf = {}
        井标记 = {}
        for a in 候选序:
            S̃表[a] = (候选键仿真回调(a) if 候选键仿真回调 is not None
                       else self._弛豫仿真(a))
        L效 = L_DEFAULT if L is None else int(L)

        def _评():
            for a in 候选序:
                v = S̃表[a]
                if hdc.IM:
                    sims = [_cos(v, it["井"]) for it in hdc.IM]
                    conf[a] = float(np.max(sims))         # confidence=max_k cos（纪要 §2.5 原式）
                    井标记[a] = int(np.argmax(np.abs(sims)))
                else:
                    conf[a] = 0.0
                    井标记[a] = -1

        _评()
        跨井 = len(set(井标记.values())) > 1
        升格 = (L is None and 跨井 and L效 < L_MAX)    # L=3 仅跨井结构事件升格（自适应时）
        if 升格:
            L效 = L_MAX
        if L效 > L_DEFAULT:                             # 深一层：各候选再续 C 次弛豫
            for a in 候选序:
                v = S̃表[a]
                for _ in range(C_弛豫):
                    v = np.sign(hdc.W_selfHop @ v)
                S̃表[a] = v
            _评()
        重排序 = sorted(候选序, key=lambda a: -conf[a])
        犹豫 = (重排序[0] != 候选序[0])
        return dict(近并列=True, 候选序=list(候选序), 重排序=重排序,
                    confidence={int(a): conf[a] for a in 候选序},
                    井标记={int(a): 井标记[a] for a in 候选序},
                    犹豫签名=bool(犹豫), L=L效, 跨井=bool(跨井))

    def 自我纠正(self, e_last, k星_执行后, k星_预期, 已试候选序号,
                 G表=None) -> dict:
        """执行后 e_last > 滚动中位 ∧ k*≠k̃ ⇒ 剔除已试候选按 G 重选一次（换向朝更小 G）；
        否则不纠正（诚实返回触发=False）。G表 缺省 None ⇒ 只报触发不重选。"""
        中位 = self.e滚动中位()
        触发 = bool(中位 is not None and e_last is not None
                    and float(e_last) > 中位 and k星_执行后 != k星_预期)
        out = dict(触发=触发, e中位=中位, 重选序号=None, 重选G=None)
        if (not 触发) or G表 is None:
            return out
        已试 = set(int(i) for i in 已试候选序号)
        剩 = [(float(g), i) for i, g in enumerate(G表) if i not in 已试]
        if not 剩:
            out["备注"] = "候选耗尽"
            return out
        g, i = min(剩)                                   # 换向朝更小 G（纪要 §2.5）
        out["重选序号"] = int(i)
        out["重选G"] = g
        return out

    # —— 结论三态 / 熔断 / 阶段 ——
    def 结论三态分类(self, 结论dict) -> str:
        """A 快量（5s 三态龄期）/ B 慢量（账本 WAL，豁免龄期，仅阶段 B）/ C 纯读出（仅日志）。
        未知通道 ⇒ ValueError（诚实拒绝；纪要 §2.4 通路表预注册映射）。"""
        通道 = 结论dict.get("通道") if isinstance(结论dict, dict) else None
        if 通道 not in 结论三态映射:
            raise ValueError("结论通道 %r 不在反哺通路表（纪要 §2.4；拒绝默认归类）" % (通道,))
        return 结论三态映射[通道]

    def 熔断判定(self, λ历史, jaccard触发构型: Optional[float] = None) -> dict:
        """双门熔断状态机（每报一窗调用一次）：门1=λ 连续 2 窗越带（>1e-2 反刍/<1e-3 停摆）；
        门2=体制切换后 Jaccard(触发构型)<0.5；恢复=连续 2 窗回带 ∧ Jaccard≥0.5 ∧
        最短驻留 1 窗（防乒乓）。动作语义（反哺关闭回退阶段 A/episode 置非发育位）由
        调用方按返回的 熔断 态执行——本模块影子期不接管控制流。"""
        下, 上 = λ带
        越上 = 越下 = 回带 = 0
        for λ in reversed(list(λ历史)):
            if λ > 上:
                if 回带 or 越下:
                    break
                越上 += 1
            elif λ < 下:
                if 回带 or 越上:
                    break
                越下 += 1
            else:
                if 越上 or 越下:
                    break
                回带 += 1
        恢复 = False
        门 = None
        if self._熔断中:
            self._熔断驻留窗 += 1
            if (self._熔断驻留窗 >= 最短驻留窗 and 回带 >= 恢复回带窗
                    and (jaccard触发构型 is None or jaccard触发构型 >= JACCARD阈)):
                self._熔断中 = False
                self._熔断驻留窗 = 0
                恢复 = True
        else:
            if 越上 >= 熔断越带窗数 or 越下 >= 熔断越带窗数:
                self._熔断中 = True
                门 = 1
                self._熔断驻留窗 = 0
            elif jaccard触发构型 is not None and jaccard触发构型 < JACCARD阈:
                self._熔断中 = True
                门 = 2
                self._熔断驻留窗 = 0
        return dict(熔断=self._熔断中, 门=门, 恢复=恢复, 连续越上=越上, 连续越下=越下,
                    连续回带=回带, 驻留窗=self._熔断驻留窗)

    def 命中率(self) -> tuple:
        """立档命中率（cos ≥ cleanup 同阈 0.5）与有效样本数（无井拍=不可比对，不计入）。"""
        样本 = [c for _, c in self._立档账 if c is not None]
        if not 样本:
            return (0.0, 0)
        return (sum(1 for c in 样本 if c >= 立档命中阈) / len(样本), len(样本))

    def 阶段B晋升判定(self, 主井占比: float) -> bool:
        """阶段 A 立档 → 阶段 B 反哺的门（纪要 §2.4）：命中率 > 主井占比（Π 既有机会水平）
        ∧ 立档样本 ≥ 立档期窗数 ∧ M14 阶段 B 已启用（人格剖面器.stage=="B"——默认关）。
        接口预注册：本判定**不自动翻转** self.stage（影子期反哺本体不实现，消融臂保留）。"""
        if self.stage != "A":
            return False
        率, n = self.命中率()
        人格B = (self.人格 is not None
                 and getattr(self.人格, "stage", "A") == "B")
        return bool(n >= 立档期窗数 and 率 > float(主井占比) and 人格B)
