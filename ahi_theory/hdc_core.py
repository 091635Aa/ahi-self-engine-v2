# -*- coding: utf-8 -*-
"""ahi_theory/hdc_core.py —— M5 · hdc_core 影子模式（E3 九步闭环新内核影子）

预登记：verify_all._CHANGELOG #65（2026-09-28）。规格出处：95_架构定稿 L4（行 256–304）、
§3.1 C1–C4（行 382–386）、L6（行 358–359）。

影子模式纪律：
  · 纯新增；不 import brain/world/system/three_loop（影子期零侵入控制流）；
    只读消费 ahi_theory.k_battery（C2b 判读）、ahi_theory.ledger（结构事件）、ahi_theory.mbp（指纹）。
  · brain.py 本体一行不改；旧 P1–P8 体检链照常跑。
  · 参数先验照抄 E3 §2 不调：γ_H=0.9 / γ_S=0.98 / λ_M=0.999 / C≈5 / η(e)=η₀·min(e,1)；
    θ_w/ε_f/θ_share/η₀/γ_e 为实装预注册初值，构造时以 kind="参数变更" 入账本（键∈M4 预注册清单）。
  · D=1024 结构先验：C1 sham-A 不过 ⇒ 换稀疏角色分解，**不许升 D 扫参**。

九步闭环（steps 1–6 条件落位 R1，steps 7–9 归 R2；95_架构定稿 行 260–283 一字照抄）：
 1) x_t = sign(Φ_fixed @ u_t)
 2) H_t = γ_H·ρ(H_{t-1}) + x_t                        （丢帧期 stale：H 只衰减不加新项）
 3) x̂_t = cleanup(M @ (a_{t-1} ⊛ H_{t-1}))            （丢帧期按 stale 语义用 H_{t-1}）
 4) e_t = 1 − cos(x_t, x̂_t) ∈ [0,2]
 5) M ← λ_M·M + η(e_t)·x_t ⊗ (a_{t-1} ⊛ H_{t-1})      （惊讶门控：预测准不写）
 6) G(a) = ê(a) − γ_e·(1 − cos(x̂_a, x̂_default))；a* = argmin G(a)
 7) S_t = sign(γ_S·ρ(S_{t-1}) + (q⊛R_q) + (a*⊛R_act) + (Φ_body⊛R_body) + (e_t⊛R_err))
    查询=代数: cleanup(unbind(S_t, R_act))…；D1 门通过前 R_act 分量权重前置零（未发育查询退化）
 8) S̃ 弛豫 C≈5 次: S̃_{c+1}=sign(W_selfHop @ S̃_c)；k* = argmax_k cos(S̃_C, IM[k])
 9) 巩固（R2 维护窗口内，前置=consolidation_permit）——W-HOME 终稿（max 消歧形式）:
    规则1 新颖性门：max_k |cos(S,IM[k])| ≥ θ_w ⇒ 跳过（防井合并）
    规则2 反主导压平：dwell(k) > θ_share·T_obs 且 tag(k) < median(tag) ⇒ W −= ε_f·(IM[k]⊗IM[k] − I)
    规则3 容量驱逐：|井集合| > 0.1·D=102 ⇒ 删 重叠最小优先、同分 |tag| 最小
    快照缓冲准入（P-D①等价选择律）: 按累积 tag 降序、容量顶格逐出尾部

 #84 E3 会签 A-修订版（G0b IM 风暴缓解，2026-10-01 预登记）——_W_HOME 冷却/合并语义：
 满载（|IM|≥井容量=102）时规则1 写入×规则3 驱逐 churn（IM 风暴，维护拍 0.5→12.7s，#76/#78 实录）
 由驱逐滞回压制：规则3 驱逐沿置位冷却余量 C_cool=20 维护拍，冷却期内新颖 S 不写新井、改就近合并
 （IM[k*]←sign(IM[k*]+λ·S) 用 S 原快照、W_selfHop+=λ·S⊗S 零对角保持，λ=0.5），冷却期不驱逐；
 冷却到期仍满⇒现行规则3 驱逐 1 井并重置冷却（慢渗 ~21 维护拍一次）。冷却计数器逐 _W_HOME 调用
 开头减一。满载段井集合非平稳（合并漂移）⇒ 下游以 IM_SATURATED 前向标记剔除测量
 （k_engine 测量窗_v21，#84(b)(ii)；只前向标记不回填改史）。未饱和 run 动力学逐位不变
 （冷却余量恒 0 时合并/免驱逐分支永不触发，与原实现等价）。

 #86 G0b-2（W-HOME 向量化，性能不改语义；2026-10-01 预登记，所有者授权重构）：
 storm_diag 定位根因=性能：规则3 驱逐链 O(n²) Python _cos 循环（1.83M 次/拍 15.9s）+冷却期
 合并外积串（~120 次/拍 np.outer(1024²)，0.78s）。修法（语义不变纪律）：
   ①重叠表 GEMM 化（规则3 进 while 前一次 |X@X.T|（f64）+驱逐增量删行列）；
   ②规则1 门 GEMM 化（快照×井 cos 矩阵一次 GEMM+写/并列 GEMV 增量；S_perp 投影求和照旧逐井 _cos）；
   ③冷却期合并外积批量 GEMM（W += λ·(S_batch.T@S_batch) 一次替代逐次 np.outer；
     λ·±1 外积累积全为 0.5 倍距精确值——f32 无舍入，两实现逐位同）；
   ④判定边界精确重算兜底：阈带（W_HOME_阈带=1e-6 相对）内候选用原 _cos 逐对精确重算再判定
     （阈值比较/tie-break/argmax 语义与循环版一致）；
   ⑤IM 上限补漏：_W_HOME 写入限流 ≤ IM上限−|IM|（#84 附带发现：同拍先写后驱中间态峰值 186>150；
     溢出新颖 S 本调用不写入不合并、留置快照缓冲；计数入返回 dict 键『规则1限流』；
     首触发账本结构事件）。
 循环参考实现原样保留为 _W_HOME_循环（模块开关 W_HOME_向量化 切换；G0B2_bitcheck 对照用）；
 等价性自检：非饱和路径固定 seed 逐位比对、饱和路径 50 拍决策序列+全状态逐位一致、
 GEMM vs _cos 数值差异界报告（详见 verify_all._CHANGELOG #86 与 meeting16/G0B2_bitcheck.py）。

 #92 G0f 饱和吸引子修复（2026-10-02 正式预登记，#91 预案会签转正式；先于改码）：
 G0f-1 慢渗语义修复——_W_HOME 两实现写入额计算处同步钳制：|IM|≥井容量(102) 时
 写入额=min(写入额,1)（恢复 #84 声明的「慢渗 ~21 维护拍驱逐 1 井」渗流率；#91 定谳
 冷却到期拍一次写 48 井=写爆管击穿；未饱和路径分支不触发逐位不变）。
 G0f-2 离线相 W-HOME 压实——新方法 离线压实()：贪心取 |cos| 最大井对，|cos|≥θ_w(复用)
 ⇒ 合并（w_a←sign(w_a+w_b)、pop 较大索引、dwell 键位移对齐、首触发结构事件），
 重复至最大 |cos|<θ_w；由 v2_engine 每离线相末调用一次（睡眠依赖记忆巩固，
 95_架构定稿 SHY 叙事）；干净 seed 井近正交⇒零触发零改动。验收链：G0F_bitcheck
 （meeting16/autonomy/G0F_bitcheck.py）→selftest 全矩阵→12-seed 规则 v3 复跑。
"""
from __future__ import annotations

import hashlib
import math
from typing import Optional

import numpy as np

__all__ = ["HDC常量", "HDC核心"]

# ── §0 预注册常量（E3 §2 先验照抄 + 实装预注册初值；变更=账本结构事件） ──────────
D_VSA = 1024                 # 结构先验（C1 sham-A 不过⇒换稀疏角色分解，不许升 D）
GAMMA_H = 0.9                # E3 §2 先验
GAMMA_S = 0.98               # E3 §2 先验（迹长 1/(1−γ_S)≈50 R1 拍）
LAMBDA_M = 0.999             # E3 §2 先验
C_弛豫 = 5                   # E3 §2 "≈5"
IM上限 = 150                 # E3 §2/E4 §2#10（井位 102+角色键 ≤8+模块角色 ≤32+余量 8）
井容量 = 102                 # 0.1·D（定理容量 0.138D≈141 的 72% 裕度）
THETA_W = 0.30               # 实装预注册：双极随机向量 cos≈0（σ≈1/√D≈0.031），0.3≈10σ 重叠门
EPS_F = 0.01                 # 实装预注册：反主导压平步长
THETA_SHARE = 0.40           # 实装预注册：dwell > 0.4·T_obs 判主导
ETA0 = 0.1                   # 实装预注册：η(e)=η₀·min(e,1)
GAMMA_E = 0.5                # 实装预注册：D5-① 峰值档 {0.3,0.5,0.7} 中值（承重检验另行预登记）
THETA_CLEANUP = 0.5          # 实装预注册：cleanup 命中阈（|cos|≥0.5 ⇒ 返回对齐 IM 项）
T_OBS_五窗 = (4096, 8192, 16384, 32768, 65536)
MASTER_SEED = 1024           # 预注册 master 种子（角色族/Φ/ρ 全部单 rng 派生，顺序固定）
快照缓冲容量 = 150           # 与 |IM| 同律
R_MOD_上限 = 32              # 模块角色上限（L4 参数表）
# ── #84 E3 会签 A-修订版（G0b IM 风暴缓解）预注册常量块（定值冻结不凑数；变更=账本结构事件） ──
W_HOME_冷却 = 20             # C_cool：|IM|≥井容量 触发的冷却维护拍数（方案 A 修订版，#84 预登记）
W_HOME_合并步长 = 0.5        # λ：冷却期就近合并步长 IM[k*]←sign(IM[k*]+λ·S)、W+=λ·S⊗S（#84 预登记）
# ── #86 G0b-2（W-HOME 向量化）预注册常量块（定值冻结不凑数；变更=账本结构事件） ──
W_HOME_向量化 = True         # True=_W_HOME 走 GEMM 向量化实现；False=循环参考实现（G0B2_bitcheck 对照开关）
W_HOME_阈带 = 1e-6           # 向量化阈带：GEMM 预筛 vs _cos 精确重算切换带（相对值；#86 预登记）
# ── #122 N3（自修改问责）预注册常量块（定值冻结不凑数；变更=账本结构事件） ──
门控满门阈 = 0.995           # gate∈(0.995,1] 视为满门零调制（不入账）；<0.995 为异常调制
INT4桶数 = 16                # INT-4 量化桶数：gate∈[0.2,1.0] 均分 16 桶（桶号=clip(int((g−0.2)/0.8·16),0,15)）
# ── #130 肌肉升级 v1 预注册常量块（四分块 L=4 多级编码；变更=账本结构事件） ──
HDC_LEVEL = 1                # 编码级数开关：1=原 1-bit sign（缺省逐位不变）；4=四分块多级编码
L4块宽 = 256                 # D=1024 分 4 块×256
L4阈值 = (-1.1503, -0.3186, 0.3186, 1.1503)   # 高斯四分位阈值（解析零拟合）


class HDC常量:
    """只读常量暴露（判据/台架引用同一份，杜绝魔法数漂移）。"""
    D = D_VSA
    γ_H = GAMMA_H
    γ_S = GAMMA_S
    λ_M = LAMBDA_M
    C = C_弛豫
    IM上限 = IM上限
    井容量 = 井容量
    θ_w = THETA_W
    ε_f = EPS_F
    θ_share = THETA_SHARE
    η0 = ETA0
    γ_e = GAMMA_E
    θ_cleanup = THETA_CLEANUP
    T_obs五窗 = T_OBS_五窗
    master种子 = MASTER_SEED
    快照缓冲容量 = 快照缓冲容量
    R_mod上限 = R_MOD_上限
    W_HOME冷却 = W_HOME_冷却            # #84 E3 会签 A-修订版：冷却维护拍数 C_cool
    W_HOME合并步长 = W_HOME_合并步长    # #84 E3 会签 A-修订版：就近合并步长 λ
    W_HOME向量化 = W_HOME_向量化        # #86 G0b-2：向量化开关（bitcheck 对照）
    W_HOME阈带 = W_HOME_阈带            # #86 G0b-2：GEMM vs 精确重算切换带（相对值）


def _双极(rng: np.random.Generator, n: int) -> np.ndarray:
    """双极 ±1 向量（dtype=float32，bind 保持律：⊛=逐元素乘）。"""
    return np.sign(rng.standard_normal(n)).astype(np.float32)


# ── §1 九步闭环核心 ──────────────────────────────────────────────────────────
class HDC核心:
    """E3 九步闭环影子核心（95_架构定稿 行 260–283 机制本体）。

    影子期职责：消费 world 状态流 u_t / 动作流 a_{t-1} / 内感受 q / 模块注册快照，
    维护 H/M/S/W_selfHop/IM/占据链，导出 C1–C4 判据所需全部量；不接管控制流。
    """

    def __init__(self, 输入维度: int, 动作维度: int, q维度: int = 8, 账本=None,
                 种子: int = MASTER_SEED):
        self.账本 = 账本
        rng = np.random.default_rng(种子)          # 单 rng 顺序派生（预注册顺序，不挑）
        # 角色向量族（固定随机、预注册种子）
        self.R_act = _双极(rng, D_VSA)
        self.R_q = _双极(rng, D_VSA)
        self.R_body = _双极(rng, D_VSA)
        self.R_err = _双极(rng, D_VSA)
        self.R_g = _双极(rng, D_VSA)
        self.R_care = _双极(rng, D_VSA)
        self.R_val = _双极(rng, D_VSA)
        self.R_mod = np.stack([_双极(rng, D_VSA) for _ in range(R_MOD_上限)])
        # 固定随机投影（不学习）：Φ_u(u→x)、Φ_act(a→动作向量)、Φ_q(q→内感受向量)
        self.Φ_u = rng.standard_normal((D_VSA, 输入维度)).astype(np.float32)
        self.Φ_act = rng.standard_normal((D_VSA, 动作维度)).astype(np.float32)
        self.Φ_q = rng.standard_normal((D_VSA, q维度)).astype(np.float32)
        # ρ(·)=固定随机置换（VSA 保持律；预注册，不引入新超参）
        self._置换 = rng.permutation(D_VSA).astype(np.int64)
        # 状态
        self.H = np.zeros(D_VSA, dtype=np.float32)          # 双极化前连续累加
        self.S = np.zeros(D_VSA, dtype=np.float32)
        # M 关联记忆 = 稠密物化基（衰减至物化拍） + 秩1缓冲（精确等价 λ 指数衰减线性性）；
        # 写入 O(1)（预分配行缓冲），读取/物化全 GEMM 向量化——避免每拍 1024² 逐项循环（#65 WCET 条款）
        # #132 C*‴：M 实例化——M_CTX=1（缺省）单实例逐位不变；=4 时按私有上下文 c_{t−1} 分实例
        # （写/读都选实例，键回到 a⊛H；上下文内关联零稀释——修乘性键绑定的摊薄缺陷）
        self._M项物化阈值 = 100
        self.M_CTX = 1
        self._M活上下文 = 0
        self._M实例 = [self._M新实例() for _ in range(self.M_CTX)]
        self.W_selfHop = np.zeros((D_VSA, D_VSA), dtype=np.float32)   # 对称零对角
        self.IM: list[dict] = []                            # {"井": vec, "tag": float, "写入拍": int}
        self.快照缓冲: list[tuple[np.ndarray, float]] = []   # (S双极, 累积tag)
        self.占据链: list[int] = []                          # k*(t)（每次 R2 维护 append；-1=无井）
        self.井dwell: dict[int, int] = {}                    # 井 k 的累计驻留 R2 维护数
        self.压平窗尾 = -1                                   # 最近一次压平事件的维护序（O4 排除用）
        self._冷却余 = 0        # #84 E3 会签 A-修订版：W-HOME 冷却余量（维护拍数；规则3 驱逐沿置位 C_cool=20，逐 _W_HOME 调用递减）
        self._合并已登记 = False  # #84 E3 会签 A-修订版：冷却期就近合并首触发结构事件登记旗（沿 _退化已登记 风格）
        self._限流已登记 = False  # #86 G0b-2：IM 上限写入限流首触发结构事件登记旗（沿 _合并已登记 风格）
        self._压实已登记 = False  # #92 G0f-2：离线相 IM 压实首触发结构事件登记旗（沿 _合并已登记 风格）
        # #122 N3 自修改问责：写-账对账摘要 + INT-4 门控调制直方图（缺省路径零扰动——
        # 摘要/直方图只增侧账不改语义数组；学习门=None ⇒ 调制计数恒 0 ⇒ 账本零新增条目；
        # #132：σ/Σ 移入 M 实例（逐实例对账），_写事件数 保持全量口径）
        self._写事件数 = 0                                            # 经路径写事件单调计数（全实例合计）
        self._幽灵锁存 = False                                        # 粘性：一旦检出失配即锁存（问责不留恢复后门）
        self._首捕获坏槽: list[int] = []                              # 首次捕获时的坏槽列表
        self._门控桶 = [0] * INT4桶数                                 # 本维护拍窗 INT-4 门控直方图
        self._门控窗调制数 = 0                                        # 本维护拍窗调制写数（>0 ⇒ 维护拍聚合入账）
        # #130 肌肉升级 v1：L=4 四分块多级编码（独立派生流 seed+90002——不插入既有 rng 序列；
        # HDC_LEVEL=1 缺省 ⇒ 本矩阵零消费，编码走原 Φ_u 路径逐位不变）
        self._rng_lvl = np.random.default_rng(int(种子) + 90002)
        self._Φ_L = self._rng_lvl.standard_normal((D_VSA, 输入维度)).astype(np.float32)
        c = np.zeros(D_VSA, dtype=np.float32)
        for ℓ in range(4):
            c[ℓ * L4块宽:(ℓ + 1) * L4块宽] = L4阈值[ℓ]
        self._c_vec = c
        # 发育门与动作候选库
        self.D1通过 = False                                  # 未发育查询退化开关（D1 门=行为层，影子期默认 False）
        self._退化已登记 = False
        self.动作候选 = self._建动作候选(动作维度)
        self.动作维度 = 动作维度
        # 计数与账本登记（θ_w/ε_f/θ_share/η₀/γ_e 预注册 → kind="参数变更"）
        self.拍数 = 0
        self.维护数 = 0
        self.超预算降环 = None                               # K:8→4 结构事件载体（WCET 触发时置 True）
        self._参数预注册入账()

    # —— 构造辅助 ——
    def _建动作候选(self, 动作维度: int) -> list[np.ndarray]:
        """固定量化动作候选库（预注册）：{零动作} ∪ {±单位轴×dim}；连续 a 映射最近候选（L2 距离）。"""
        候选 = [np.zeros(动作维度, dtype=np.float32)]
        for i in range(动作维度):
            for s in (1.0, -1.0):
                v = np.zeros(动作维度, dtype=np.float32)
                v[i] = s
                候选.append(v)
        return 候选

    def _参数预注册入账(self) -> None:
        """θ_w/ε_f/θ_share/η₀/γ_e 初值以 kind="参数变更" 入账本（键∈M4 预注册 16 键清单）。"""
        if self.账本 is None:
            return
        for 键, 值, 理由 in (
            ("θ_w", THETA_W, "M5 #65 预注册初值：新颖性门 10σ 重叠阈（E3 §1.2 预注册机制，实装定值）"),
            ("ε_f", EPS_F, "M5 #65 预注册初值：反主导压平步长（E3 §1.2 预注册机制，实装定值）"),
            ("压平份额阈值", THETA_SHARE, "M5 #65 预注册初值：dwell>0.4·T_obs 判主导（E3 §1.2）"),
            ("η尺度", (ETA0, GAMMA_E), "M5 #65 预注册初值：η₀=0.1（η(e)=η₀·min(e,1)）、γ_e=0.5（D5-① 峰值档中值）"),
        ):
            self.账本.登记("参数变更", payload={"键": 键, "值": 值},
                           理由=理由 + "（先验不调；变更须再次结构事件）")
        # #84 E3 会签 A-修订版：W-HOME 冷却/合并机制常量以结构事件入账（账本零改动——
        # 键"W-HOME冷却"∉ledger 变更清单，走 kind="参数变更" 将被账本纪律拒绝并抛异常；
        # 预登记本体=verify_all._CHANGELOG #84 + §0 预注册常量块，同 井容量/快照缓冲容量 先例）
        self.账本.登记("结构事件",
                       payload={"事件": "W-HOME 冷却机制预注册", "键": "W-HOME冷却",
                                "值": {"C_cool": W_HOME_冷却, "合并步长λ": W_HOME_合并步长}},
                       理由="#84 E3 会签 A-修订版（G0b IM 风暴缓解方案 A）：|IM|≥井容量 进入冷却 "
                            "C_cool=20 维护拍，冷却期规则1 新颖 S 就近合并（IM[k*]←sign(IM[k*]+λ·S)、"
                            "W+=λ·S⊗S，λ=0.5 预注册）；θ_w 本体不动；变更清单扩充须动 ledger.py，"
                            "本轮红线不许——留待授权轮另行结构事件")

    # —— 绑定代数 ——
    def _ρ(self, v: np.ndarray) -> np.ndarray:
        return v[self._置换]

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _cleanup(self, z: np.ndarray) -> Optional[np.ndarray]:
        """cleanup：IM 最近邻对齐（|cos|≥θ_cleanup ⇒ 返回符号对齐的 IM 项）；否则 sign(z)；零向量⇒None。"""
        nz = float(np.linalg.norm(z))
        if nz == 0.0:
            return None
        if self.IM:
            sims = [abs(self._cos(z, it["井"])) for it in self.IM]
            k = int(np.argmax(sims))
            if sims[k] >= THETA_CLEANUP:
                井 = self.IM[k]["井"]
                return 井 if self._cos(z, 井) >= 0 else -井
        return np.sign(z)

    def _动作向量(self, a: np.ndarray) -> np.ndarray:
        return np.sign(self.Φ_act @ a.astype(np.float32))

    def _最近候选(self, a: np.ndarray) -> int:
        d = [float(np.sum((a - c) ** 2)) for c in self.动作候选]
        return int(np.argmin(d))

    # —— M 关联记忆（秩1缓冲+稠密物化基；λ 指数衰减线性性下与稠密阵逐项精确等价） ——
    def _M新实例(self) -> dict:
        """#132 C*‴：M 实例构造（秩1缓冲+稠密基+物化拍+#122 逐实例摘要）。"""
        I = dict(X=np.zeros((self._M项物化阈值, D_VSA), dtype=np.float32),
                 K=np.zeros((self._M项物化阈值, D_VSA), dtype=np.float32),
                 η=np.zeros(self._M项物化阈值, dtype=np.float64),
                 t=np.zeros(self._M项物化阈值, dtype=np.float64),
                 n=0, 稠密=np.zeros((D_VSA, D_VSA), dtype=np.float32), 物化拍=0,
                 槽摘要=np.zeros(self._M项物化阈值, dtype=np.uint64), 写事件数=0)
        I["稠密摘要"] = self._状态摘要(I["稠密"])
        return I

    @property
    def _M活(self) -> dict:
        """当前激活的 M 实例（M_CTX=1 恒为实例 0——缺省路径逐位不变）。"""
        return self._M实例[self._M活上下文]

    def 设M上下文(self, c: int) -> None:
        """#132：M_CTX>1 时按私有上下文切换激活实例（写/读同索引=真核条件化对齐）。"""
        if self.M_CTX > 1:
            self._M活上下文 = int(c) % self.M_CTX

    def 启用M上下文(self, n: int = 4) -> None:
        """#132：扩实例至 n 份（实验臂开关；缺省构造即单实例，不调用则逐位不变）。"""
        while len(self._M实例) < int(n):
            self._M实例.append(self._M新实例())
        self.M_CTX = int(n)

    def _状态摘要(self, *数组) -> int:
        """#122 N3：blake2b8 状态摘要（对账仪器；uint64。只读不改语义数组）。"""
        h = hashlib.blake2b(digest_size=8)
        for a in 数组:
            h.update(np.ascontiguousarray(a).tobytes())
        return int.from_bytes(h.digest(), "big")

    def 审计对账(self) -> dict:
        """#122 N3 写-账对账（#132 扩多实例）：逐实例重算槽摘要 σ′_i（i<n）与稠密摘要 Σ′。

        任何绕路径直写（改槽 X/K/η/t、改 _M项数 后未经路径补摘要、改 _M稠密）必致失配。
        粘性锁存：一旦检出失配 _幽灵锁存=True 永久保持（问责不留恢复后门；后续合法覆写
        不洗白历史），_首捕获坏槽 记录首次捕获的坏槽列表（含实例前缀）。"""
        坏槽: list = []
        稠密一致 = True
        for mi, I in enumerate(self._M实例):
            n = I["n"]
            for i in range(n):
                if I["槽摘要"][i] != self._状态摘要(I["X"][i], I["K"][i],
                                                     I["η"][i], I["t"][i]):
                    坏槽.append((mi, i))
            if I["稠密摘要"] != self._状态摘要(I["稠密"]):
                稠密一致 = False
        本拍一致 = (not 坏槽) and 稠密一致
        if not 本拍一致 and not self._幽灵锁存:
            self._幽灵锁存 = True
            self._首捕获坏槽 = [f"M{mi}#{i}" for mi, i in 坏槽] if 坏槽 \
                else (["稠密基"] if not 稠密一致 else [])
            if self.账本 is not None:
                self.账本.登记("结构事件",
                               payload={"事件": "幽灵写入捕获", "维护序": self.维护数,
                                        "坏槽": self._首捕获坏槽, "拍数": self.拍数},
                               理由="#122 N3 写-账对账检出绕路径直写（问责留痕，粘性锁存不洗白）")
        return dict(一致=not self._幽灵锁存, 当前一致=本拍一致, 坏槽=坏槽,
                    稠密一致=稠密一致, 锁存=self._幽灵锁存, 首捕获坏槽=self._首捕获坏槽,
                    槽数=[I["n"] for I in self._M实例], 写事件数=self._写事件数)

    _G3 = None  # #208 G3 驻留后端钩子（缺省 None=逐位不变；接线见 verify_all #208）

    def _M读(self, 键矩阵: np.ndarray) -> np.ndarray:
        """M @ 键矩阵（键矩阵 (D,Q)）：稠密物化基（衰减至当前拍）+ 秩1缓冲 GEMM 求和。
        #132：读写均作用于激活实例。"""
        I = self._M活
        if self._G3 is not None:  # #208 stage A：M 读驻留接管（缺省 None 走 numpy 原径）
            return self._G3.M读驻留对象(I, self.拍数, 键矩阵)
        out = I["稠密"] @ 键矩阵
        折 = LAMBDA_M ** (self.拍数 - I["物化拍"])
        if 折 != 1.0:
            out = out * 折
        n = I["n"]
        if n:
            w = I["η"][:n] * LAMBDA_M ** (self.拍数 - I["t"][:n])     # (n,)
            c = I["K"][:n] @ 键矩阵                                        # (n,Q)
            out += I["X"][:n].T @ (w[:, None] * c)                         # (D,n)@(n,Q) 小侧加权
        return out

    def _M物化(self) -> None:
        """缓冲项并入稠密基（GEMM 一次完成；物化后基衰减基准=当前拍）。#132：作用于激活实例。"""
        I = self._M活
        n = I["n"]
        折 = LAMBDA_M ** (self.拍数 - I["物化拍"])
        if 折 != 1.0:
            I["稠密"] *= 折
        if n:
            w = I["η"][:n] * LAMBDA_M ** (self.拍数 - I["t"][:n])
            I["稠密"] += (I["X"][:n].T * w) @ I["K"][:n]             # (D,n)@(n,D)
            I["n"] = 0
        I["物化拍"] = self.拍数
        I["稠密摘要"] = self._状态摘要(I["稠密"])      # #122 N3：物化沿稠密摘要经路径刷新

    # —— steps 1–6（R1 侧，逐拍） ——
    def 编码u(self, u_t: np.ndarray) -> np.ndarray:
        """#130 肌肉升级 v1：u→码。HDC_LEVEL=1（缺省）⇒ sign(Φ_u·u) 逐位不变；
        HDC_LEVEL=4 ⇒ 四分块多级编码 x̃=concat_ℓ sign(Φ^(ℓ)·u−c_ℓ·‖u‖₂)（幅值信息进码，
        修 sign 反正弦收缩+量化吞幅值的 10× 主导项）。"""
        if HDC_LEVEL == 1:
            return np.sign(self.Φ_u @ u_t.astype(np.float32))
        return np.sign(self._Φ_L @ u_t.astype(np.float32)
                       - self._c_vec * float(np.linalg.norm(u_t)))

    def 步进R1(self, u_t: np.ndarray, a_prev: np.ndarray, 帧新鲜: bool = True,
               内生偏置=None, 意图先验=None, 意图强度: float = 1.0,
               efe_mode: bool = False, 学习门: float = None, 私有码=None) -> dict:
        """R1 逐拍：steps 1–6。帧新鲜=False ⇒ stale：H 只衰减、预测用 H_{t-1}、e 不更新。
        #105/#106 M2：内生偏置=每候选动作的标量偏置（长度=|动作候选|），argmin G 前逐项减去；
        None（缺省）⇒ 与 #104 前行为逐位不变（既有消费方零影响）。
        #111/#112 M4：意图先验=1024 维自模型向量 S_int（想象终态），step 6 内
        G(a)−=意图强度·cos(x̂_a,S_int)（κ=300 无量纲标准化，#112 预注册；缺省 1.0=#111 口径）；
        意图先验=None⇒逐位不变。时间因果：S_int 由调用方保证产生于本拍之前（维护拍）。
        #114 P-N N1：efe_mode=True ⇒ a*=argmin EFE(a)=z(ê)−z(u_a)（u_a=mean|Z[:,i+1]| 符号前
        裕量，跨候选逐拍 z 标准化、无状态零常数）；step1–5/M 写入原样；False⇒逐位不变。
        #118 N2：学习门=标量∈[0.2,1]（元模型 M₂ 对本拍 M 写的乘子，见 m2_intrinsic.学习门）；
        None（缺省）⇒ η 原样逐位不变。
        #127 C*′：私有码=1024 维双极 P_{t−1}=R_priv⊛r_{t−1}（引擎私有熵绑定），并入 M 读写键
        键=a⊛sign(H)⊛P——引擎预测可条件于私有状态（world 转移核含 κ_w·Φ_w·r_{t−1}）而外部
        观察者不可；None（缺省）⇒ 键与原式逐位不变。"""
        self.拍数 += 1
        assert len(self.IM) <= IM上限, "|IM| 超标 ⇒ cleanup 应降 R2（E4 §2#10）"
        H_prev = self.H.copy()
        x̂键 = a_prev.astype(np.float32) if a_prev is not None else np.zeros(self.动作维度, np.float32)
        键向量 = self._动作向量(x̂键) * np.sign(H_prev)     # a_{t-1} ⊛ H_{t-1}（双极乘）
        if 私有码 is not None:                             # #127：⊛P_{t-1}（全 1 恒等=缺省不变）
            键向量 = 键向量 * np.asarray(私有码, dtype=np.float32).ravel()
        if 帧新鲜:
            x_t = self.编码u(u_t)                           # step 2（#130：编码经 编码u，L=1 逐位不变）
            self.H = GAMMA_H * self._ρ(self.H) + x_t        # step 2（连续累加；绑定侧取 sign）
        else:
            x_t = None                                       # stale：无新观测
            self.H = GAMMA_H * self._ρ(self.H)
        # step 3–4：预测与惊讶（M 读取经秩1列表+稠密基，单次 GEMM 覆盖全部候选键）
        候选键 = np.stack([(self._动作向量(c) if c is not None else self._动作向量(x̂键))
                           * np.sign(H_prev) for c in ([None] + self.动作候选)])   # (1+K, D)；首行=主预测键
        if 私有码 is not None:                             # #127：候选键同样并入私有码（读侧条件化）
            候选键 = 候选键 * np.asarray(私有码, dtype=np.float32).ravel()[None, :]
        Z = self._M读(候选键.T)                                                    # (D, 1+K)
        x̂ = np.sign(Z[:, 0])
        if x_t is None:
            e_t = None                                       # 丢帧期不产出惊讶（step 5 不写）
        else:
            e_t = 1.0 - self._cos(x_t, x̂)                    # M 零时 cos=0 ⇒ e=1（初始惊讶驱动学习）
            # step 5：惊讶门控写入（预测准不写）——秩1追加 O(1)；缓冲满 ⇒ 就地物化兜底
            η = ETA0 * min(e_t, 1.0)
            if 学习门 is not None:                         # #118 N2：元模型学习门（None⇒逐位不变）
                η = η * float(学习门)
            if η > 0.0:
                I = self._M活
                if I["n"] >= self._M项物化阈值:
                    self._M物化()
                i = I["n"]
                I["X"][i] = x_t
                I["K"][i] = 键向量
                I["η"][i] = η
                I["t"][i] = self.拍数
                I["n"] = i + 1
                # #122 N3：经路径写入侧账（槽摘要+写事件计数；只增侧账不改语义数组）
                I["槽摘要"][i] = self._状态摘要(I["X"][i], I["K"][i],
                                               I["η"][i], I["t"][i])
                I["写事件数"] += 1
                self._写事件数 += 1
                if 学习门 is not None:
                    g = float(学习门)
                    if g < 门控满门阈:                     # #122：仅异常调制入账（满门零调制）
                        self._门控桶[min(INT4桶数 - 1, max(0, int((g - 0.2) / 0.8 * INT4桶数)))] += 1
                        self._门控窗调制数 += 1
        # step 6：反事实行动选择 G(a)=ê(a)−γ_e·(1−cos(x̂_a,x̂_default))（复用 Z，无额外 matvec）
        z_default = Z[:, 1]                                                        # 零动作=候选0
        x̂_default = np.sign(z_default)
        最优i, 最优G, 反事实表, ê表, 意图cos表, u_a表, EFE表 = 0, None, [], [], [], [], []
        S_int = None if 意图先验 is None else np.asarray(意图先验, dtype=np.float32).ravel()
        for i in range(len(self.动作候选)):
            x̂_a = np.sign(Z[:, i + 1])
            ê = 1.0 - self._cos(x_t, x̂_a) if x_t is not None else 1.0
            G = ê - GAMMA_E * (1.0 - self._cos(x̂_a, x̂_default))
            cos意图 = float(self._cos(S_int, x̂_a)) if S_int is not None else 0.0
            if 内生偏置 is not None:                       # #105/#106 M2：内生偏置（None⇒逐位不变；
                G = G - float(内生偏置[i])                 #    偏置计算见 meeting17/m2_intrinsic.py）
            if S_int is not None:                          # #111/#112 M4：意图先验回注（κ=300 标准化）
                G = G - 意图强度 * cos意图
            反事实表.append(G)
            ê表.append(ê)
            意图cos表.append(cos意图)
            u_a表.append(float(np.mean(np.abs(Z[:, i + 1]))))   # #114：符号前裕量（epistemic 原料）
            if 最优G is None or G < 最优G:
                最优i, 最优G = i, G
        efe选i = 最优i
        if efe_mode and x_t is not None:                   # #114 P-N N1：EFE 选择层（False⇒逐位不变）
            Gv = np.asarray(反事实表, dtype=np.float64)    # #115：G′=G+novelty 偏置（时间驱动项并入）
            uv = np.asarray(u_a表, dtype=np.float64)
            zG = (Gv - Gv.mean()) / (Gv.std() + 1e-12)
            zu = (uv - uv.mean()) / (uv.std() + 1e-12)
            EFE表 = list(zG - zu)                          # 实用(含 novelty 时间驱动)+认识价值
            # #119：EFE 选择域排除零动作（i=0）——平局零动作死锁防护（M5 病态的 EFE 形态；
            # 与 M2 novelty 的 i≠0 同律）。零动作仍可经 G 路径被选（efe_mode=False 时照旧）。
            非零 = [i for i in range(1, len(EFE表))]
            efe选i = 非零[int(np.argmin([EFE表[i] for i in 非零]))]
        return dict(x=x_t, e=e_t, a星=self.动作候选[efe选i], a星序号=efe选i,
                    G表=反事实表, ê表=ê表, 意图cos表=意图cos表, u_a表=u_a表, EFE表=EFE表,
                    H双极=np.sign(self.H), 键向量=键向量)

    # —— steps 7–9（R2 侧，每 20 R1 拍） ——
    def 维护R2(self, q: np.ndarray, e_last: Optional[float], a星: np.ndarray,
               已注册模块数: int, 价值标量: float = 0.0,
               consolidation_permit: bool = True, 压平标记窗内: bool = False) -> dict:
        """R2 维护（每 20 R1 拍）：step 7 S 更新+查询 → step 8 弛豫+k* → step 9 W-HOME 巩固。"""
        self.维护数 += 1
        # #122 N3：INT-4 门控调制聚合入账（仅窗内有异常调制才登记——20000 拍最坏 1000 条 ≪
        # 账本环形 4096；学习门=None 缺省路径计数恒 0 ⇒ 账本零新增条目零扰动）
        if self._门控窗调制数 > 0:
            if self.账本 is not None:
                self.账本.登记("门控调制聚合",
                               payload={"事件": "学习门调制聚合", "维护序": self.维护数,
                                        "桶": list(self._门控桶),
                                        "调制写数": self._门控窗调制数},
                               理由="#122 N3 INT-4 门控调制入账（预注册聚合口径：仅 gate<0.995 "
                                    "异常调制，每维护拍一条防环形溢出）")
            self._门控桶 = [0] * INT4桶数
            self._门控窗调制数 = 0
        if self._M活["n"] >= self._M项物化阈值:
            self._M物化()                                    # 摊销物化（R2 窗口内，非 R1 热路径）
        # 身体图式投影：已注册模块多数投票（空 ⇒ 零向量）
        if 已注册模块数 <= 0:
            Φ_body = np.zeros(D_VSA, dtype=np.float32)
        else:
            Φ_body = np.sign(self.R_mod[:min(已注册模块数, R_MOD_上限)].sum(axis=0))
        qv = np.sign(self.Φ_q @ q.astype(np.float32))
        av = self._动作向量(a星)
        e标 = 1.0 if e_last is None else float(np.clip(e_last, 0.0, 2.0))
        # step 7：S 更新（双极）+ 查询（未发育 ⇒ R_act 分量前置零）
        S_raw = (GAMMA_S * self._ρ(self.S) + qv * self.R_q + av * self.R_act
                 + Φ_body * self.R_body + e标 * self.R_err)
        self.S = np.sign(S_raw)
        查询键 = self.S * (self.R_act if self.D1通过 else np.zeros(D_VSA, dtype=np.float32))
        if not self.D1通过 and not self._退化已登记:
            self._退化已登记 = True
            if self.账本 is not None:
                self.账本.登记("结构事件", payload={"事件": "未发育查询退化首触发", "维护序": self.维护数},
                               理由="D1 门通过前 unbind(S,R_act) 权重前置零（E3 §4-L5 预注册；结构事件非调参）")
        查询结果 = self._cleanup(查询键)
        # step 8：弛豫 C≈5 次 + 占据链
        S̃ = self.S.copy()
        for _ in range(C_弛豫):
            S̃ = np.sign(self.W_selfHop @ S̃)
        k星 = -1
        if self.IM:
            sims = [self._cos(S̃, it["井"]) for it in self.IM]
            k星 = int(np.argmax(np.abs(sims)))
        self.占据链.append(k星)
        self.井dwell[k星] = self.井dwell.get(k星, 0) + 1
        # 快照缓冲准入（P-D①等价选择律：按累积 tag 降序、容量顶格逐出尾部）
        self.快照缓冲.append((self.S.copy(), float(价值标量)))
        self.快照缓冲.sort(key=lambda p: p[1], reverse=True)
        if len(self.快照缓冲) > 快照缓冲容量:
            self.快照缓冲 = self.快照缓冲[:快照缓冲容量]
        # step 9：W-HOME 巩固（前置=consolidation_permit）
        巩固 = dict(规则1跳过=0, 规则1合并=0, 规则2压平=0, 规则3驱逐=0)
        if consolidation_permit and not 压平标记窗内:
            巩固 = self._W_HOME()
        return dict(S=self.S, S弛豫=S̃, k星=k星, 查询结果=查询结果,
                    巩固跳过=巩固["规则1跳过"], 压平数=巩固["规则2压平"], 驱逐数=巩固["规则3驱逐"],
                    饱和=bool(len(self.IM) >= 井容量),        # #84 E3 会签 A-修订版：IM_SATURATED 前向标记
                    饱和合并数=巩固["规则1合并"])             #   （|IM|≥井容量=满载⇒井集合非平稳段，下游测量前向剔除；
                                                              #    原有键值非饱和路径逐位不变——本两键为 #84 新增读出）

    def _W_HOME(self) -> dict:
        """step 9 巩固：W-HOME 三规则（max 消歧终稿；95_架构定稿 行 274–281 一字照抄）。

        #84 E3 会签 A-修订版（G0b IM 风暴缓解）冷却/合并语义：
          · 调用开头冷却余量减一：self._冷却余 = max(0, self._冷却余 − 1)（per-巩固调用=per-维护拍）；
          · 规则3 驱逐沿置位冷却余量 = C_cool = 20 维护拍（驱逐滞回）；
          · 冷却期内（余量>0）规则1 新颖性门通过的新颖 S 不写新井，改就近合并：
            k*=argmax_k|cos(S,IM[k])|（复用本循环已算 sims），IM[k*]←sign(IM[k*]+λ·S)
            （S 原快照，E3 文本字面；不用 S_perp）、W_selfHop+=λ·S⊗S（零对角保持同 _井写入 强制）；
            冷却期不做规则3 驱逐——|IM|≥井容量 的写入×驱逐 churn 压成 ~21 维护拍一次的慢渗驱逐；
          · 冷却到期仍满 ⇒ 现行规则3 驱逐 1 井并重置冷却。
        满载段井集合非平稳（合并漂移）⇒ 下游 IM_SATURATED 前向标记剔除测量（k_engine 测量窗_v21，
        #84(b)(ii)；只前向标记不回填改史）。冷却余量恒 0（未饱和）时全部分支与原实现逐位等价。

        #86 G0b-2：本方法=dispatcher（冷却余量递减两实现共用）——W_HOME_向量化=True 走
        _W_HOME_GEMM（向量化+阈带精确重算兜底+写入限流），False 走 _W_HOME_循环（循环参考实现，
        同语义含限流；G0B2_bitcheck 对照）。两实现判定语义逐位一致（自检见 #86）。"""
        self._冷却余 = max(0, self._冷却余 - 1)   # #84 E3 会签 A-修订版：冷却余量逐维护拍递减
        if W_HOME_向量化:
            return self._W_HOME_GEMM()
        return self._W_HOME_循环()

    def _限流事件登记(self) -> None:
        """#86 G0b-2 ⑤：写入限流首触发结构事件（沿 _合并已登记 风格；两实现共用）。"""
        if not self._限流已登记:
            self._限流已登记 = True
            if self.账本 is not None:
                self.账本.登记("结构事件",
                               payload={"事件": "W-HOME 写入限流首触发", "维护序": self.维护数,
                                        "井数": len(self.IM)},
                               理由="#86 G0b-2 ⑤（IM 上限补漏，#84 附带发现 1）：_W_HOME 写入前 clip 待写数 "
                                    "≤ IM上限−|IM|（调用起点计）——同拍先写后驱的中间态 |IM| 峰值不受 "
                                    "步进R1 assert 约束（理论峰值 井容量+快照缓冲容量=252）；限流候选按快照"
                                    "顺序（tag 降序即缓冲序），溢出新颖 S 本调用不写入不合并、留置快照缓冲"
                                    "（容量顶格逐出照旧）；机制首触发留痕（结构事件非调参）")

    def _W_HOME_循环(self) -> dict:
        """循环参考实现（#86 G0b-2：与 #85 版 _W_HOME 逐字同语义 + 写入限流；W_HOME_向量化=False 走本路径）。

        #84 E3 会签 A-修订版 冷却/合并语义（见 _W_HOME docstring）逐字保持。"""
        跳过 = 压平 = 驱逐 = 合并 = 限流 = 0
        写入额 = max(0, IM上限 - len(self.IM))     # #86 ⑤：本调用写入限流额（IM上限−|IM|，调用起点计）
        if len(self.IM) >= 井容量:                  # #92 G0f-1：饱和态写入额钳 1（恢复 #84 慢渗语义——
            写入额 = min(写入额, 1)                 #    #91 定谳冷却到期拍一次写 48 井=写爆管；未饱和分支不触发）
        # 规则1 新颖性门 + S_perp 写入（冷却期内新颖 S 改就近合并——#84 E3 会签 A-修订版）
        for S, _tag in self.快照缓冲:
            if not self.IM:
                self._井写入(S, _tag)
                写入额 -= 1
                continue
            sims = [abs(self._cos(S, it["井"])) for it in self.IM]
            if max(sims) >= THETA_W:                        # 规则1：防井合并
                跳过 += 1
                continue
            if self._冷却余 > 0:                            # #84 E3 会签 A-修订版：冷却期就近合并
                k星 = int(np.argmax(sims))                  # 就近井（复用规则1 已算 sims）
                self.IM[k星]["井"] = np.sign(
                    self.IM[k星]["井"] + W_HOME_合并步长 * S).astype(np.float32)
                self.W_selfHop += W_HOME_合并步长 * np.outer(S, S)
                np.fill_diagonal(self.W_selfHop, 0.0)       # 零对角强制（同 _井写入）
                合并 += 1
                if not self._合并已登记:                    # #84 E3 会签 A-修订版：首次合并=结构事件（沿 _退化已登记 风格）
                    self._合并已登记 = True
                    if self.账本 is not None:
                        self.账本.登记("结构事件",
                                       payload={"事件": "W-HOME 冷却期就近合并首触发", "维护序": self.维护数,
                                                "井数": len(self.IM)},
                                       理由="#84 E3 会签 A-修订版：满载冷却期内规则1 新颖 S 改就近合并"
                                            "（IM[k*]←sign(IM[k*]+λ·S)、W_selfHop+=λ·S⊗S，λ=0.5，零对角保持）"
                                            "——机制首触发留痕（结构事件非调参）；下游以维护R2 返回 "
                                            "饱和/饱和合并数 作 IM_SATURATED 前向标记剔除测量（不回填改史）")
                continue
            if 写入额 <= 0:                                 # #86 ⑤：写入限流（溢出新颖 S 本调用不写入不合并）
                限流 += 1
                self._限流事件登记()
                continue
            S_perp = np.sign(S - sum(self._cos(S, it["井"]) * it["井"] for it in self.IM))
            self._井写入(S_perp, _tag)
            写入额 -= 1
        # 规则2 反主导压平（价态排序：tag < median ⇒ 压）
        if self.IM:
            tags = np.array([it["tag"] for it in self.IM], dtype=np.float64)
            med = float(np.median(tags))
            T_obs = T_OBS_五窗[-1]
            for k, it in enumerate(self.IM):
                if self.井dwell.get(k, 0) > THETA_SHARE * T_obs and it["tag"] < med:
                    self.W_selfHop -= EPS_F * (np.outer(it["井"], it["井"]) - np.eye(D_VSA, dtype=np.float32))
                    压平 += 1
            if 压平:
                self.压平窗尾 = self.维护数
        # 规则3 容量驱逐：|井|>102 ⇒ 删 重叠最小优先、同分 |tag| 最小
        if self._冷却余 <= 0:                               # #84 E3 会签 A-修订版：冷却期不驱逐
            while len(self.IM) > 井容量:
                重叠 = []
                for i, it in enumerate(self.IM):
                    others = max((abs(self._cos(it["井"], o["井"]))
                                  for j, o in enumerate(self.IM) if j != i), default=0.0)
                    重叠.append((others, abs(it["tag"]), i))
                重叠.sort()                                 # 重叠最小优先；同分取 |tag| 最小
                驱逐i = 重叠[0][2]
                self.IM.pop(驱逐i)
                驱逐 += 1
            if 驱逐:
                self._冷却余 = W_HOME_冷却   # #84 E3 会签 A-修订版：规则3 驱逐沿置位冷却（慢渗 ~21 拍一次）
        return dict(规则1跳过=跳过, 规则1合并=合并, 规则1限流=限流, 规则2压平=压平, 规则3驱逐=驱逐)

    def _W_HOME_GEMM(self) -> dict:
        """#86 G0b-2 向量化实现：语义与 _W_HOME_循环 逐位一致（判定边界精确重算兜底）。

        向量化点（对应 #86 ①②③④⑤）：
          规则1 门：快照缓冲 (n_s,1024) 归一 × 井 (n_w,1024) 归一.T → f64 GEMM 一次全表 |cos|；
            井写入⇒追加列（GEMV）、合并⇒替换列（GEMV）——增量维护；阈值判定/就近 argmax 对
            W_HOME_阈带=1e-6 相对带内候选用原 _cos 逐对精确重算再判定（④兜底）；
            S_perp 投影求和照旧逐井 _cos（现有公式不变）。
          冷却期合并：井向量更新照旧逐次（后续快照的门判定依赖新井——时序语义）；W 累积收集合并 S
            批量，循环后一次 W += λ·(S_batch.T@S_batch) + fill_diagonal 一次（③；λ·±1 外积累积全为
            0.5 倍距精确值，f32 GEMM 无舍入——与逐次 np.outer 逐位同）。
          S_perp 写入：井登记照旧逐次（IM 追加即时——时序语义）；W 外积累积收集 S_perp 批量，
            循环后一次 W += (S_batchw.T@S_batchw)（③扩展，#87 如实登记：±1 外积累积=1 倍距精确值，
            同 λ 批量论证——f32 无舍入、与逐次 np.outer 逐位同；空 IM 首写仍走 _井写入）。
          规则3 驱逐链：进 while 前井×井重叠全表一次 f64 GEMM（①），while 内增量删行列；
            驱逐选择=重叠最小值及其 ±阈带内候选精确重算（④）后排序（同分 |tag| 最小照旧）。
          写入限流（⑤）：同 _W_HOME_循环。"""
        跳过 = 压平 = 驱逐 = 合并 = 限流 = 0
        带相对 = W_HOME_阈带
        写入额 = max(0, IM上限 - len(self.IM))     # #86 ⑤：本调用写入限流额（调用起点计）
        if len(self.IM) >= 井容量:                  # #92 G0f-1：饱和态写入额钳 1（与 _W_HOME_循环 同步改）
            写入额 = min(写入额, 1)                 #    恢复 #84 慢渗语义（#91 写爆管击穿修复；未饱和不触发）
        # —— 规则1 门 GEMM 预筛矩阵（快照行归一 × 井行归一；f64；预分配缓冲免拼接拷贝）——
        Ss = None
        if self.快照缓冲:
            Sb = np.stack([s for s, _t in self.快照缓冲]).astype(np.float64)
            sn = np.linalg.norm(Sb, axis=1)
            sn[sn == 0.0] = 1.0
            Ss = Sb / sn[:, None]
        C = None                                     # (n_s, 列数) |cos| 预筛视图；列数恒 == len(self.IM)
        Cbuf = None                                  # 预分配底层缓冲 (n_s, IM上限)——追加列零拷贝
        if self.IM and Ss is not None:
            Cbuf = np.empty((Ss.shape[0], IM上限), dtype=np.float64)
            X = np.stack([it["井"] for it in self.IM]).astype(np.float64)
            xn = np.linalg.norm(X, axis=1)
            xn[xn == 0.0] = 1.0
            Cbuf[:, :len(self.IM)] = np.abs(Ss @ (X / xn[:, None]).T)
            C = Cbuf[:, :len(self.IM)]

        def _列(w: np.ndarray) -> np.ndarray:
            wf = w.astype(np.float64)
            wn = float(np.linalg.norm(wf))
            return np.abs(Ss @ (wf / (wn if wn > 0.0 else 1.0)))

        合并批量: list = []
        写入批量: list = []                          # ③扩展：S_perp 写入外积批量（W 累积循环后一次 GEMM）
        # 规则1 新颖性门 + S_perp 写入（冷却期内新颖 S 改就近合并——#84 E3 会签 A-修订版）
        for 行i, (S, _tag) in enumerate(self.快照缓冲):
            if not self.IM:
                self._井写入(S, _tag)
                写入额 -= 1
                if Ss is not None:
                    Cbuf = np.empty((Ss.shape[0], IM上限), dtype=np.float64)
                    Cbuf[:, 0] = _列(self.IM[-1]["井"])
                    C = Cbuf[:, :1]
                continue
            row = C[行i]
            mg = float(row.max())
            if abs(mg - THETA_W) <= 带相对 * THETA_W:   # ④兜底：阈带内 ⇒ 逐对精确重算再判定
                sims = [abs(self._cos(S, it["井"])) for it in self.IM]
                新颖 = max(sims) < THETA_W
            else:
                新颖 = mg < THETA_W
            if not 新颖:                                   # 规则1：防井合并
                跳过 += 1
                continue
            if self._冷却余 > 0:                            # #84 E3 会签 A-修订版：冷却期就近合并
                k星 = int(np.argmax(row))                   # ④兜底：GEMM argmax（阈带内近同分序的语义边界见 #87——
                self.IM[k星]["井"] = np.sign(                #    非退化表与循环版逐位一致；退化均匀域 f32 噪声级如实登记）
                    self.IM[k星]["井"] + W_HOME_合并步长 * S).astype(np.float32)
                合并批量.append(S)
                合并 += 1
                if Ss is not None:
                    C[:, k星] = _列(self.IM[k星]["井"])      # 合并⇒替换列（井向量已变）
                if not self._合并已登记:                    # #84 E3 会签 A-修订版：首次合并=结构事件
                    self._合并已登记 = True
                    if self.账本 is not None:
                        self.账本.登记("结构事件",
                                       payload={"事件": "W-HOME 冷却期就近合并首触发", "维护序": self.维护数,
                                                "井数": len(self.IM)},
                                       理由="#84 E3 会签 A-修订版：满载冷却期内规则1 新颖 S 改就近合并"
                                            "（IM[k*]←sign(IM[k*]+λ·S)、W_selfHop+=λ·S⊗S，λ=0.5，零对角保持）"
                                            "——机制首触发留痕（结构事件非调参）；下游以维护R2 返回 "
                                            "饱和/饱和合并数 作 IM_SATURATED 前向标记剔除测量（不回填改史）")
                continue
            if 写入额 <= 0:                                 # #86 ⑤：写入限流（溢出新颖 S 本调用不写入不合并）
                限流 += 1
                self._限流事件登记()
                continue
            S_perp = np.sign(S - sum(self._cos(S, it["井"]) * it["井"] for it in self.IM))
            # 井登记照旧逐次（IM 追加即时——时序语义；同 _井写入 的登记字段）；W 外积入批量（③扩展）
            self.IM.append(dict(井=S_perp.astype(np.float32), tag=float(_tag), 写入拍=self.拍数))
            写入批量.append(S_perp.astype(np.float32))
            写入额 -= 1
            if Ss is not None:
                Cbuf[:, C.shape[1]] = _列(self.IM[-1]["井"])   # 写入⇒追加列（预分配缓冲，零拷贝）
                C = Cbuf[:, :C.shape[1] + 1]
        # ③扩展：S_perp 写入外积批量 GEMM（±1 精确累加——与逐次 np.outer 逐位同）
        if 写入批量:
            Bw = np.stack(写入批量)
            self.W_selfHop += Bw.T @ Bw
        # ③冷却期合并外积批量 GEMM（一次替代逐次 np.outer；0.5 倍距精确——f32 无舍入）
        if 合并批量:
            B = np.stack(合并批量)
            self.W_selfHop += W_HOME_合并步长 * (B.T @ B)
        if 写入批量 or 合并批量:
            np.fill_diagonal(self.W_selfHop, 0.0)           # 零对角强制（同 _井写入，一次）
        # 规则2 反主导压平（价态排序：tag < median ⇒ 压）——与循环版逐字同
        if self.IM:
            tags = np.array([it["tag"] for it in self.IM], dtype=np.float64)
            med = float(np.median(tags))
            T_obs = T_OBS_五窗[-1]
            for k, it in enumerate(self.IM):
                if self.井dwell.get(k, 0) > THETA_SHARE * T_obs and it["tag"] < med:
                    self.W_selfHop -= EPS_F * (np.outer(it["井"], it["井"]) - np.eye(D_VSA, dtype=np.float32))
                    压平 += 1
            if 压平:
                self.压平窗尾 = self.维护数
        # 规则1/②③① 容量驱逐：重叠全表一次 f64 GEMM + while 内增量删行列（#86 ①）；
        # 驱逐选择=重叠最小值 ±阈带内候选精确重算（④）后排序（同分 |tag| 最小照旧）
        if self._冷却余 <= 0 and len(self.IM) > 井容量:      # #84 E3 会签 A-修订版：冷却期不驱逐
            X = np.stack([it["井"] for it in self.IM]).astype(np.float64)
            xn = np.linalg.norm(X, axis=1)
            xn[xn == 0.0] = 1.0
            Xh = X / xn[:, None]
            G = np.abs(Xh @ Xh.T)
            np.fill_diagonal(G, -1.0)                    # 对角线常置 −1（行最大/argmax 天然避开自身；免逐驱逐全表拷贝）
            标签 = np.abs(np.array([it["tag"] for it in self.IM], dtype=np.float64))
            while len(self.IM) > 井容量:
                n = G.shape[0]
                行最大 = G.max(axis=1)
                m星 = float(行最大.min())
                候选井 = np.flatnonzero(行最大 <= m星 * (1.0 + 带相对))   # ④兜底：最小值阈带内候选
                others = 行最大.copy()
                for i in 候选井.tolist():
                    if 行最大[i] <= 0.0:
                        others[i] = 0.0                     # 与循环版 default=0.0 对齐
                        continue
                    j1 = int(np.argmax(G[i]))               # 行最大伙伴=GEMM argmax（阈带内近同分序的
                    others[i] = abs(self._cos(self.IM[i]["井"], self.IM[j1]["井"]))   # 语义边界见 #87）
                重叠 = sorted((float(others[i]), float(标签[i]), i) for i in range(n))
                驱逐i = 重叠[0][2]
                self.IM.pop(驱逐i)
                G = np.delete(np.delete(G, 驱逐i, axis=0), 驱逐i, axis=1)   # ①增量删行列（不重算全表）
                np.fill_diagonal(G, -1.0)
                标签 = np.delete(标签, 驱逐i)
                驱逐 += 1
            if 驱逐:
                self._冷却余 = W_HOME_冷却   # #84 E3 会签 A-修订版：规则3 驱逐沿置位冷却（慢渗 ~21 拍一次）
        return dict(规则1跳过=跳过, 规则1合并=合并, 规则1限流=限流, 规则2压平=压平, 规则3驱逐=驱逐)

    def _井写入(self, S: np.ndarray, tag: float) -> None:
        """W_selfHop += S_perp⊗S_perp（对称外积、零对角强制）+ IM 登记。"""
        self.W_selfHop += np.outer(S, S)
        np.fill_diagonal(self.W_selfHop, 0.0)               # 零对角强制
        self.IM.append(dict(井=S.astype(np.float32), tag=float(tag), 写入拍=self.拍数))

    def 离线压实(self) -> int:
        """#92 G0f-2：离线相 W-HOME 压实（睡眠依赖记忆巩固的架构语义，95_架构定稿 SHY 叙事）。

        每离线相末由 v2_engine 调用一次（ecae 离线步循环退出后、引擎状态回 wake 前）：
        贪心取 |cos| 最大井对，|cos|≥θ_w（复用=零新参数）⇒ 合并：
          w_a ← sign(w_a + w_b)（tag/写入拍 随 a 保留）、pop 两索引中较大者 b
          （井dwell 键按位移对齐：b 以上索引 −1，被并井 b 的 dwell 随井删除），
          重复至最大 |cos|<θ_w。W_selfHop 不触碰（预登记口径：井-井合并不改 hop 结构）。
        干净 seed 井近正交（|cos|<0.30）⇒ 零触发零改动（非饱和路径逐位不变——#92 可证明性）。
        返回合并次数。诚实残留（#92 如实登记）：dwell 按索引记账，b 以上索引位移 −1
        为有界归属漂移（仅喂规则2 压平门；离线相内 hdc 冻结无并发）。"""
        合并数 = 0
        while len(self.IM) >= 2:
            X = np.stack([it["井"] for it in self.IM]).astype(np.float64)
            xn = np.linalg.norm(X, axis=1)
            xn[xn == 0.0] = 1.0
            Xh = X / xn[:, None]
            G = np.abs(Xh @ Xh.T)
            np.fill_diagonal(G, 0.0)
            ij = int(np.argmax(G))
            i, j = divmod(ij, G.shape[1])
            if float(G[i, j]) < THETA_W:
                break
            a, b = (i, j) if i < j else (j, i)
            wa = self.IM[a]["井"].astype(np.float64)
            wb = self.IM[b]["井"].astype(np.float64)
            self.IM[a]["井"] = np.sign(wa + wb).astype(np.float32)
            self.IM.pop(b)
            self.井dwell = {k if k < b else k - 1: v
                            for k, v in self.井dwell.items() if k != b}
            合并数 += 1
            if not self._压实已登记:
                self._压实已登记 = True
                if self.账本 is not None:
                    self.账本.登记("结构事件",
                                   payload={"事件": "IM 离线压实首触发", "维护序": self.维护数,
                                            "当前井数": len(self.IM), "本轮合并": 合并数},
                                   理由="#92 G0f-2：离线相 W-HOME 压实首触发（贪心 |cos|≥θ_w 井对合并，"
                                        "w_a←sign(w_a+w_b)、pop 较大索引、dwell 键位移对齐；零新参数——"
                                        "θ_w 复用 #65 预注册值；睡眠依赖记忆巩固架构语义，95_架构定稿 SHY 叙事）")
        return 合并数

    # —— 判据导出（C1–C4 消费面） ——
    def 导出占据链(self) -> list[int]:
        """k*(t) 占据链（低频，供 k_battery.c2b_readout 五窗判读）；S_t 低频差分由台架侧逐维护累计。"""
        return list(self.占据链)

    def 反事实归因(self, x_t: np.ndarray, a星: np.ndarray, H_prev双极: np.ndarray) -> float:
        """sim(x_t, x̂|a*) − max_{a≠a*} sim(x_t, x̂|a)（接口签名语义对齐 selfprobe，M8 消费）。"""
        候选键 = np.stack([self._动作向量(c) * H_prev双极 for c in self.动作候选])
        Z = self._M读(候选键.T)
        a星i = self._最近候选(a星)
        s星 = self._cos(x_t, np.sign(Z[:, a星i]))
        其他 = [self._cos(x_t, np.sign(Z[:, i])) for i in range(len(self.动作候选)) if i != a星i]
        return s星 - (max(其他) if 其他 else 0.0)

    def 井弛豫能量斜率(self, 探针: np.ndarray) -> float:
        """C2 仪器：井弛豫能量斜率均值。E0=−cos(连续探针, 井)（迭代前初值），
        E(c)=−cos(连续激活 z_c=W@S̃_c 归一, 井)；斜率<−c₀ ⇒ 盆内回落（探针→井底）。"""
        if not self.IM:
            return 0.0
        k = int(np.argmax(np.abs([self._cos(探针, it["井"]) for it in self.IM])))
        能量 = [-self._cos(探针 / max(float(np.linalg.norm(探针)), 1e-12), self.IM[k]["井"])]
        S̃ = 探针.copy()
        for _ in range(C_弛豫):
            z = self.W_selfHop @ S̃
            S̃ = np.sign(z)
            nz = float(np.linalg.norm(z))
            z方向 = z / nz if nz > 0 else z
            能量.append(-self._cos(z方向, self.IM[k]["井"]))
        if len(能量) < 2:
            return 0.0
        return float(np.polyfit(np.arange(len(能量)), 能量, 1)[0])

    def 结构事件(self, kind_payload: dict, 理由: str) -> None:
        """超预算降环等结构事件入口（K:8→4 等第一动作，不是调 η）。"""
        if self.账本 is not None:
            self.账本.登记("结构事件", payload=dict(kind_payload), 理由=理由)
