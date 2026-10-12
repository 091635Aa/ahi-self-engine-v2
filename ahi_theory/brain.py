# -*- coding: utf-8 -*-
"""ahi_theory.brain —— 统一意识引擎

设计原则：**一个系统，六种理论，六套独立验证。**
不是一个玩具对应一个理论，而是同一套动力学同时承载六种理论各自的机制，
再由六个相互独立的验证模块分别检验。

六种理论在本引擎中的机制落点：
  P1 功能主义      : 自我状态 s 因果承重 + **自预测机制** + 置信度校准头
  P2 生物自然主义   : 湿件接口协议（可插拔真实神经元单元）
  P3 高阶表征/AST   : 一阶注意 g -> 注意图式 s -> 因果咬合回注 g -> 外向使用
  P4 整合信息论     : 12 区核心 + 可测整合度（Phi* / O-info / zap-and-zip）
  P5 主动推理       : 变分自由能 + 内生目标（无外部奖励）+ 反暗室
  P6 构成性自模型   : 隐式不动点 m* + 快慢分离 h + 迟滞

统一状态向量（每步）：
  h_reg   [R, d]      区域内部状态
  g       [R]         一阶注意增益（动态、可被内部信号调制）
  s       [K]         注意图式（对 g 的预测模型状态）
  m       [K]         自模型内容（不动点解；★ 含自预测通道）
  h_slow  [K]         慢变量（多吸引子 -> 迟滞）
  L       [5]         调质核团输出（独立核团，非器官自产）
  conf    [R]         置信度头输出
  z_pred  [R, d]      引擎对自己**下一步区域状态**的前向预测 ẑ_{t|t-1}
  self_err[1]         实现的自预测误差 e_t = ‖h_t − ẑ_{t|t-1}‖²/(R·d)
  mom     [1]         自身自由能的去趋势动量（我最近在变好还是变差）
  rho_hat [1]         谱半径估计（有限差分 Lyapunov）
  F       [1]         变分自由能 = 复杂度 + 不准确度 + 自预测不准确度
"""
from __future__ import annotations
import numpy as np

# ------------------------------------------------------------------ 超参
R_REGIONS = 12          # 核心区数（默认值；Brain 可经构造参数 R 覆盖，见 verify_P4 的真分割对照）
D_REGION = 24           # 每区维度
K_SCHEMA = 16           # 图式 / 自模型维度
N_ORGANS = 7            # 外围器官（与 FCA 对齐）


def _spec_norm(A: np.ndarray, target: float) -> np.ndarray:
    """把矩阵谱半径归一到 target（绝不按 sigma_max 标定）。"""
    rho = float(max(abs(np.linalg.eigvals(A))))
    return A / rho * target if rho > 0 else A


# ================================================================== 引擎
class Brain:
    """统一意识引擎。所有六种理论的机制共处于同一套动力学。"""

    def __init__(self, seed: int = 0, R: int | None = None,
                 rho_target: float = 0.92,
                 dt_slow: float = 0.01, kappa_target: float = 0.6,
                 w_slow: float = 2.0,
                 w_mom: float = 10.0, w_fwd: float = 1.4, w_surp: float = 1.0,
                 w_self: float = 6.0e3, ema_alpha: float = 0.02,
                 mom_bounded: bool = True, base_cap: float | None = 2.0,
                 three_layer: bool = True, w_mom1: float | None = None,
                 gamma1: float = 0.3, gamma2: float = 0.1,
                 intero: bool = True, w_gq: float = 0.10, w_mq: float = 1.0,
                 g_inj_mode: str = "agc", sigma_agc: float = 0.6, alpha_agc: float = 0.01,
                 agency: bool = False, agency_gate: bool = False, agency_learn: bool = True,
                 constitution: bool = False, w_constitution: float = 1.0,
                 dt_constitution: float = 0.02):
        # ★ 2026-09-24 去饱和修复：w_mom 9.0e4 → 10.0、动量项有界化（tanh）、
        #   base 归一化到 ‖base‖ ≤ base_cap。实测依据与 PASS 线见 meeting2/M1*_probe_desat.py。
        #   旧行为可复现：Brain(..., w_mom=9.0e4, mom_bounded=False, base_cap=None)。
        # ★ 2026-09-25 三层时间常数自模型（E1 §3.3 件 3；实测 meeting3/M3_*、M4_*）：
        #   背景：单层引擎上 X5（去饱和）/ X6（τ(m)>τ(h)）/ P1-ECE 三者**不可并存**
        #   （实测 meeting3/D2_*、D3_*：11 组 (w_mom,cap) 全 6-seed 复核均失败）。
        #   三层把 m 扩成 [m1(16)|m2(16)|m3(16)]，各层**只吃时间尺度匹配的输入**：
        #       m1（快）← 动量 tanh(mom)·u_mom + 惊奇[e/π,1]
        #       m2（中）← 当前区域状态 z = h.ravel()
        #       m3（慢）← 慢变量 h_slow
        #   层间**单向**注入（m2←+γ1·m1、m3←+γ2·m2，禁反向），三层 base 均软归一 ≤base_cap。
        #   **`_self_model()` 与 `self.m` 一律不动**（P6 的 κ/NFE/迟滞/神谕 继续作用于它，
        #   故 P6 现有 Nec=1 资产不受影响）；只有 `state()` 暴露的 m 改为 m1，另加 m_slow=m3。
        #   实测（M3/M4，6 seed）：饱和 0.0000（X5 ✓）、τ(m3)=335–417 且 τ(m1)<τ(m2)<τ(m3)
        #   且 τ(m3)>τ(h)（X6 ✓）、P1-ECE 由单层的 2/6 提升到 **5/6**（差 seed 0 的校准，
        #   且四个 m1 动量增益 10/100/1000/10000 全都停在 5/6 ⇒ 是校准头的结构限制，非增益）。
        #   三层矩阵用**独立随机流**（seed+501），故既有参数的随机序列逐位不变。
        # ★ R 可覆盖：真分割对照需要**两份 6 区引擎**各自带独立全局回路
        #   （s / m / h_slow / L）。缺省 12 ⇒ 与既有调用逐位一致（随机流不变）。
        self.R = R_REGIONS if R is None else int(R)
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.t = 0

        # ---- 区域动力学（谱半径归一，避免 tanh 饱和）----
        self.A = np.stack([
            _spec_norm(rng.normal(size=(D_REGION, D_REGION)) / np.sqrt(D_REGION),
                       rho_target * 0.9) for _ in range(self.R)])
        self.Wc = rng.normal(size=(self.R, self.R)) * 0.5
        np.fill_diagonal(self.Wc, 0.0)
        self.Wc = _spec_norm(self.Wc, 3.0)          # 区际耦合同样按谱半径标定
        self.Bin = rng.normal(size=(self.R, D_REGION, D_REGION)) / np.sqrt(D_REGION)
        self.h = rng.normal(size=(self.R, D_REGION)) * 0.1

        # ---- P3: 一阶注意（真实、动态、可被调制的增益）----
        self.g = np.ones(self.R) * 0.5
        self.Wg = rng.normal(size=(self.R, self.R + K_SCHEMA)) * 0.4
        self.bg = np.zeros(self.R)
        # ★ 注意的**慢速隐藏驱动**：真实的注意有持续性（sustained attention、
        #   注意瞬脱），使其非马尔可夫。图式必须从 g 的历史中推断 g_slow，
        #   才能比「直接用 g_t」预测得更好 —— 这正是 AST 所说的「关于注意的模型」。
        self.g_slow = np.zeros(self.R)
        self.Wgs = _spec_norm(rng.normal(size=(self.R, self.R)) / np.sqrt(self.R), 0.97)
        self.gs_amp = 1.0          # 慢驱动幅度（标定值；2.0 会加重饱和）
        self.gain_scale = 2.5      # 增益尺度（标定值：饱和 0.29、图式 R2 0.92）
        self.gs_weight = 0.6       # 慢驱动权重（标定值）

        # ---- P3: 注意图式（对 g 的预测模型，含自身再入）----
        self.Ws = _spec_norm(rng.normal(size=(K_SCHEMA, K_SCHEMA)) / np.sqrt(K_SCHEMA), 0.85)
        self.Ws_in = rng.normal(size=(K_SCHEMA, self.R + self.R)) / np.sqrt(self.R * 2)
        self._ctx = np.zeros(self.R)
        self.s = rng.normal(size=K_SCHEMA) * 0.1

        # ---- P6: 自模型（隐式不动点 + 慢变量）----
        self.Wm = rng.normal(size=(K_SCHEMA, K_SCHEMA)) / np.sqrt(K_SCHEMA)
        # ★ κ = ‖J‖₂ 是**谱范数** σ_max，不是谱半径 ρ。按 ρ 归一会让 σ_max > 1
        #   （实测 κ=1.069，超出 [0.3,0.9] 且不动点不再保证唯一）。
        _sm = float(np.linalg.svd(self.Wm, compute_uv=False)[0])
        if _sm > 0:
            self.Wm = self.Wm / _sm * kappa_target
        self.Wm_in = rng.normal(size=(K_SCHEMA, self.R * D_REGION)) / np.sqrt(self.R * D_REGION)
        self.h_slow = rng.normal(size=K_SCHEMA) * 0.05
        self.dt_slow = dt_slow
        self.m = np.zeros(K_SCHEMA)
        self.nfe = 0

        # ★ P6 关键修复：慢变量必须带**非线性正反馈**，否则它只是 m 的低通滤波，
        #   于是 m* = Φ(z) 仍是 z 的确定函数（前馈神谕 R² = 0.999 实测）。
        # 谱半径 > 1 的非线性反馈产生**多重吸引子** ⇒ 真正的路径依赖。
        # ★★ 2026-09-26c 落地 C：**结构化静态参数** —— W_slow 改为与 seed 无关的确定性构造。
        #   为什么：N3「慢变量吸引子数」的 `_n_attractors` 是**离线模拟、只读构造期参数**（根因三），
        #     任何 `step()` 内的在线改动对它**不可见**；而 W_slow 的尺度原本由「随机抽签」决定
        #     （谱半径归一后虽都是 1.9，但**方向/结构**逐 seed 不同 ⇒ mech/inert 比在 seed5 上只有 1.27）。
        #   怎么做（AGC 模板的同一思路：把尺度从抽签变成与 seed 无关的约束）：用**独立随机流**
        #     `default_rng(1201)` 生成固定矩阵并归一 ρ=1.9；其余参数（w_slow/dt_slow/beta_slow/
        #     base_cap）**一律保持默认**（改动最小）。实测（meeting6/T2_C_独立代价.txt）：
        #     N3 11/12→**12/12**（mech/inert min 3.29/3.44）、P6 11→12、P3/P4/P5/P7/P8 零翻转、
        #     P1 8/12→7/12（退 3 改 2）⇒ 预注册代价闸门四条全部满足。
        #   ★ 随机流纪律：**仍然消费同一发 `rng.normal(size=(K,K))`（丢弃）**，使其后所有参数
        #     （Wm_fwd / Wm_surp / u_mom / WL / GL / Wconf / anchor …）的随机序列**逐位不变**
        #     —— 只有 W_slow 一个量改变。
        _ = rng.normal(size=(K_SCHEMA, K_SCHEMA))            # 占位（保持主流位置；原用于 W_slow）
        self.W_slow = _spec_norm(
            np.random.default_rng(1201).normal(size=(K_SCHEMA, K_SCHEMA)) / np.sqrt(K_SCHEMA), 1.9)
        self.beta_slow = 0.8
        self.w_slow = w_slow

        # ================= P1: 自预测机制 =================
        # ★ 失败诊断（实测，见 _diag）：未加此机制前，引擎的**任何**内部特征
        #   （m / h / h_slow / g 及其线性组合）对「下一步 F 是否会上升」的线性读出
        #   在训练段 AUC≈0.63~0.94，但在**留出段全部塌到 0.47~0.57**（方向反号）。
        #   即：自我状态不携带关于自己下一步表现的可靠信息 —— 这是**架构缺陷**，
        #   任何校准算法（IRLS / 梯度下降 / 直方图分箱 / 等渗 PAVA）都救不回来。
        # ★ 根因：m 的不动点基座 base = Wm_in·h + w_slow·h_slow 只编码「我现在是
        #   什么」，完全不含任何**指向未来**的量。而引擎其实**能够**预测自己的未来：
        #   (a) 前向自预测 ẑ_{t|t-1}：引擎知道自己的更新规则（A/Bin/Wc/门控）与
        #       自己的门控来源 L，唯一不可预知的是外部输入 u（取期望 0）。
        #       实测：预测出的 g_{t+1} 与实际 g_{t+1} 的平均绝对偏差 = 0.0，
        #       前向预测残差范数 / ‖h‖ = 0.4% —— 自预测是**真**的。
        #   (b) 自身表现动量 mom_t = ΔF_{t-1} − EMA(ΔF)：ΔF 的实测自相关 = 0.993
        #       （分段 0.9925/0.9938/0.9939/0.9927），因此 ΔF_{t-1} 预测
        #       y_t = 1[F_{t+1} > F_t] 的 AUC = 0.9962（训练段 0.9970 / 留出 0.9955）。
        #   两个通道都是**自生成**的：没有外部标注、没有外部奖励参与。
        self.Wm_fwd = rng.normal(size=(K_SCHEMA, self.R * D_REGION)) / np.sqrt(self.R * D_REGION)
        self.Wm_surp = rng.normal(size=(K_SCHEMA, 2)) / np.sqrt(2.0)
        _um = rng.normal(size=K_SCHEMA)
        self.u_mom = _um / np.linalg.norm(_um)      # 动量在自模型空间的固定单位方向
        self.z_pred = self.h.copy()                 # ẑ_{t|t-1}：引擎对自己下一步的预测
        self.self_err = 0.0                         # e_t：实现的自预测误差
        self.self_err_ema = 0.0                     # π_t：期望的自预测误差
        self.dF_prev = 0.0                          # ΔF_{t-1} = F_{t-1} − F_{t-2}
        self.dF_ema = 0.0                           # ΔF 的自身基线
        self.dF_abs_ema = 0.0                       # |ΔF| 的自身尺度（用于自归一化）
        self.mom = 0.0                              # = (ΔF_{t-1} − EMA(ΔF)) / EMA(|ΔF|)
        self.F_prev = None
        self._e_rel = 1.0
        self._pi_rel = 1.0
        self.w_mom = w_mom
        self.w_fwd = w_fwd
        self.w_surp = w_surp
        self.w_self = w_self
        self.ema_alpha = ema_alpha
        self.mom_bounded = bool(mom_bounded)   # 动量项有界化（tanh），见 _self_model
        self.base_cap = None if base_cap is None else float(base_cap)   # base 软归一化上限
        # ---- 三层时间常数自模型（E1 件 3；独立随机流，不动既有参数序列）----
        self.three_layer = bool(three_layer)
        if self.three_layer:
            _r3 = np.random.default_rng(seed + 501)
            self.Wm1 = _spec_norm(_r3.normal(size=(K_SCHEMA, K_SCHEMA)) / np.sqrt(K_SCHEMA), 0.6)
            self.Wm2 = _spec_norm(_r3.normal(size=(K_SCHEMA, K_SCHEMA)) / np.sqrt(K_SCHEMA), 0.6)
            self.Wm3 = _spec_norm(_r3.normal(size=(K_SCHEMA, K_SCHEMA)) / np.sqrt(K_SCHEMA), 0.6)
            self.w_mom1 = float(w_mom if w_mom1 is None else w_mom1)
            self.gamma1, self.gamma2 = float(gamma1), float(gamma2)
            self.m1 = np.zeros(K_SCHEMA); self.m2 = np.zeros(K_SCHEMA); self.m3 = np.zeros(K_SCHEMA)

        # ---- P5/P0-5: 独立调质核团 ----
        self.WL = rng.normal(size=(5, 4)) * 0.8
        self.L = np.full(5, 0.5)
        self.setpoint = np.array([1.0, 0.0, 0.0, 0.0])
        self.GL = rng.normal(size=(self.R, 3, 5)) * 0.5
        self.F = 0.0
        self.F_hist = []

        # ---- P1: 校准头（与任务头参数不共享）----
        self.Wconf = rng.normal(size=(self.R, K_SCHEMA)) / np.sqrt(K_SCHEMA)  # 输入为 m
        self.conf = np.ones(self.R) * 0.5

        # ---- 身份锚点（不可变）----
        a = rng.normal(size=K_SCHEMA)
        self.anchor = a / np.linalg.norm(a)

        # ---- 湿件接口（P2）----
        self.wetware = None

        # ================= E1 件 1：内感受内核 q（最小自我）=================
        # ★ 2026-09-25 实装（规格：E1 §3.1；先测后改，探针 meeting3/Q1_probe_intero.py 6/6 通过）
        #   为什么需要它：E1 §1 的实测诊断 —— 引擎**没有内感受通道、没有受调节量**，
        #   离线活动幅度跨 seed 差 48 倍（无任何人调节它），agency 为 0。q 是"被维持在
        #   设定点附近、偏差本身进动力学"的最小结构（"自我不是被表示出来的，是被维持出来的"）。
        #   结构：q ∈ R^8，τ_q 对数均匀 [200,2000]；ζ ∈ R^15 = 内部量的**慢分量**
        #   （α_ζ=0.01 的 EMA；ζ = [‖h_i‖(12), std_i(g), ē, ‖m‖]，**不含 u**）；
        #   q ← q + (1/τ_q)[−(q − q*) + A_q σ(W_iq ζ) + W_qq tanh(q)]，
        #   W_qq 谱半径归一到 0.95（持续自持；与 P6 的 κ 要 σ_max 是两回事）；
        #   q* ← q* + η_q (q_ref − q*) + ν_t（设定点缓漂，η_q=5e-4、|ν|~1e-4）。
        #   回注两处：base += W_mq q（内感受进自模型）；
        #             g_raw += W_gq (tanh q − q*)（内感受**偏差**直接改注意增益 = 脑干觉醒回路）。
        #   维度决定（探针 D1）：E1 列出 4 标量 + L(5) = 9 维却声明 R^15；取逐区范数读法
        #   [‖h_i‖(12), std_i(g), ē, ‖m‖] = 15 维（恰好 15，且全部是内部量）。
        #   时间尺度决定（探针 D2）：E1 式里 −(q−q*)/τ_q 与外侧 Δ_q=1/τ_q 双重除以 τ_q，
        #   平衡偏移 ~τ_q·O(1) ⇒ 回注会把 base/g_raw 钉死；取"τ_q 是 q 的时间常数"语义。
        #   实测（Q1，6 seed）：A_int = 0.985–1.077（PASS ≥0.60 6/6）、删回注后 ‖Δh‖ 效应量
        #   4.57–6.05（PASS ≥0.30 6/6）、q 饱和 0.0000、τ(q)=576–711、F_max 0.28–0.47、
        #   κ=0.600、m1 饱和 0.0000 ⇒ 不反噬任何回归条款。
        #   测度自证（Q2–Q7 五轮取证后才可用）：N1 伪特征 R²≤0.05、N2 异源目标 R²≤0.05、
        #   N3 正对照（u 直灌 q）R²=0.911–0.928 ⇒ A_int 测度有判别力（见 meeting3/aintr_instrument.py）。
        #   关闭：intero=False ⇒ 不建参数、不回注、不推进 q ⇒ 与旧实现**逐位一致**。
        self.intero = bool(intero)
        if self.intero:
            _ri = np.random.default_rng(seed + 701)     # 独立随机流：既有参数序列逐位不变
            self._ri = np.random.default_rng(seed + 702)
            self.NQ, self.NZ = 8, 15
            self.tau_q = np.exp(_ri.uniform(np.log(200.0), np.log(2000.0), self.NQ))
            self.A_q = np.ones(self.NQ)
            self.W_iq = _ri.normal(size=(self.NQ, self.NZ)) / np.sqrt(self.NZ)
            self.W_qq = _spec_norm(_ri.normal(size=(self.NQ, self.NQ)) / np.sqrt(self.NQ), 0.95)
            self.W_mq = _ri.normal(size=(K_SCHEMA, self.NQ)) / np.sqrt(self.NQ) * float(w_mq)
            self.W_gq = _ri.normal(size=(self.R, self.NQ)) / np.sqrt(self.NQ) * float(w_gq)
            self.w_gq, self.w_mq = float(w_gq), float(w_mq)
            self.g_inj_mode = str(g_inj_mode)
            self.sigma_agc, self.alpha_agc = float(sigma_agc), float(alpha_agc)
            self.aro_ema = np.zeros(self.R)
            self.aro_dev = np.zeros(self.R)
            # ★ 作用**形式**与**尺度**的标定（E1 §3.1 只说"让内感受的偏差直接改注意**增益**"）：
            #   形式：g_inj_mode="raw"（加在注意的原始驱动上 = 我最初的读法）实测把 P3 打崩
            #     （w_gq=1.0 时 G1a 饱和 0.512、G2a −2.23 ⇒ P3 0/6）；"gain"（乘性调制增益因子，
            #     与既有 0.6·L[0] 同形式、更贴 E1 原文）⇒ G1a 饱和 0.24–0.41、G2a 0.71–0.91。
            #   尺度：R2 扫描（meeting3/R2_probe_gain_mode.py，2 seed × 6 组）中 **w_gq=0.10, gain**
            #     是唯一同时通过 C1 G1a/C2 G2a/C3 P1-ECE/C4 P7-X2 四线的格子（G2a 0.862/0.712、
            #     P1 ECE 0.0399/0.0383、X2 效应量 3.47/3.27）。**残余退化如实记录**：G2a 从基线
            #     0.85–0.99 降到 0.71–0.86（仍远高于判据线 0.5）。全部扫描表见 work log §19。
            self.intero_scale = (float(w_gq), float(w_mq))
            self.q = np.zeros(self.NQ)
            self.q_ref = np.zeros(self.NQ)
            self.q_star = _ri.normal(size=self.NQ) * 1e-2
            self.eta_q, self.nu_amp = 5e-4, 1e-4
            self.zeta_ema = np.zeros(self.NZ)
            self.alpha_z = 0.01

        # ================= E1 件 2：efference-copy 比较器（agency）=================
        # 规格：E1 §3.2。表达通道 a_t = tanh(W_a m_t) ∈ R^24；再传入预测 ŝ_t = Ŵ a_{t−1}；
        # 残差 r_t = u_t − ŝ_t；自举精度权重 π_t（残差低于自身基线者权重高）；
        # 门 u_eff = u − Λ⊙ŝ（Λ=1，E1 未给 Λ 的确定方式）。
        # ★ 实装口径（探针 meeting3/A1_probe_efference.py，6 seed 实测）：
        #   · **默认 agency=False**（模块不建、动力学一字不改 ⇒ 与旧实现逐位一致，
        #     P1–P7 的既有认证继续有效）；agency=True 时学 Ŵ 并算残差；agency_gate=True 才开门。
        #   · 学习 = π 加权累积最小二乘（初始 Ŵ=0、π 有 0.1 下限以便自举、相对量级岭防爆），
        #     **纯自监督无标签**：只在闭环段学（u 由自己的动作造成）。
        #   · 实测：残差可分性 AUC=1.0000（6/6，冻结后到「他人驱动」段测）；
        #     信号级自生成分量消除率 0.988–0.992（PASS ≥0.90）；同形同尺度随机比较器(sham)
        #     的消除率 −0.08…−0.41（判别力对照）；‖r‖_world 未被抑制（6/6）。
        #   · 两条**负面发现**（work log §18）：① E1 的 X3「等感觉流可分性」**结构性不可证伪**
        #     （u 逐点相同 + 参数相同 ⇒ 状态逐位相同，装任何模块 AUC 恒 0.5）；
        #     ② 轨迹级/Δh 级「衰减」测度不可用（混沌饱和：输入幅值不是轨迹散度的单调函数）。
        self.agency = bool(agency)
        self.agency_gate = bool(agency_gate)
        self.agency_learn = bool(agency_learn)   # False ⇒ 冻结 Ŵ（P8 的"冻结后到他人驱动段测"）
        if self.agency:
            _ra = np.random.default_rng(seed + 901)      # 独立随机流：既有参数序列逐位不变
            self.W_a = _ra.normal(size=(D_REGION, K_SCHEMA)) / np.sqrt(K_SCHEMA)
            self.W_eff = np.zeros((D_REGION, D_REGION))
            self.S_eff = np.zeros((D_REGION, D_REGION))
            self.B_eff = np.zeros((D_REGION, D_REGION))
            self.a_act = np.tanh(self.W_a @ self.m)
            self.a_prev = self.a_act.copy()
            self.s_hat = np.zeros(D_REGION)
            self.r_eff = np.zeros(D_REGION)
            self.pi_eff = 0.5
            self.r_ema_eff = 0.0
            self.r_dev_eff = 0.0
            self.gamma_eff = 0.9995

        # ================= N4：构成可变性 v_constitution（**默认关**）=================
        # ★ 2026-09-25 实装（规格 §N4；先测后改，探针 meeting5/N4_probe_constitution.py）。
        #   语义：v_constitution 编码「**当前参数配置不是必然的**」——它以当前参数配置为条件
        #   更新（配置摘要 cfg_feat + 当前状态摘要），并把结果注入自模型基座
        #   base += W_constitution @ v_constitution（`_self_model` 的 base 处）。
        #   **默认 constitution=False** ⇒ 不建参数、不推进 v、不回注 ⇒ 与旧实现**逐位一致**，
        #   锚点不变（M6 哈希复核）。必须默认关：注入 base 会改 m → h_slow → h，破锚点。
        #   独立随机流 `default_rng(seed + 1101)`：既有参数的随机序列**逐位不变**。
        self.constitution = bool(constitution)
        if self.constitution:
            _rc = np.random.default_rng(seed + 1101)
            self.NC = 8
            self.W_cfg = _rc.normal(size=(self.NC, len(self._CFG_KEYS)))
            self.W_st = _rc.normal(size=(self.NC, 6)) / np.sqrt(6.0)
            self.W_cc = _spec_norm(_rc.normal(size=(self.NC, self.NC)) / np.sqrt(self.NC), 0.5)
            self.W_constitution = (_rc.normal(size=(K_SCHEMA, self.NC)) / np.sqrt(self.NC)
                                   * float(w_constitution))
            self.w_constitution = float(w_constitution)
            self.dt_constitution = float(dt_constitution)
            self.v_constitution = np.zeros(self.NC)
            self.cfg_feat = self._constitution_cfg()

    _CFG_KEYS = ("kappa", "rho_W_slow", "n_Wm_in", "n_Wm_fwd", "w_slow", "beta_slow",
                 "dt_slow", "gamma1", "gamma2", "gain_scale", "gs_weight", "w_mom")

    def _constitution_cfg(self) -> np.ndarray:
        """当前**参数配置**的摘要（N4 的条件输入；全部由本实例的参数量出，无外部量）。"""
        o = np.array([float(np.linalg.svd(self.Wm, compute_uv=False)[0]),
                      float(max(abs(np.linalg.eigvals(self.W_slow)))),
                      float(np.linalg.norm(self.Wm_in)), float(np.linalg.norm(self.Wm_fwd)),
                      float(self.w_slow), float(self.beta_slow), float(self.dt_slow),
                      float(getattr(self, "gamma1", 0.0)), float(getattr(self, "gamma2", 0.0)),
                      float(self.gain_scale), float(self.gs_weight), float(self.w_mom)])
        return np.tanh(o / (np.abs(o).max() + 1e-12))

    def _constitution_step(self):
        """推进 v_constitution：以当前参数配置（cfg_feat）与当前状态为条件，dt 步长自持。"""
        if not self.constitution:
            return
        st = np.array([float(np.linalg.norm(self.m)) / np.sqrt(K_SCHEMA),
                       float(np.tanh(self.F)),
                       float(np.abs(self.h).mean()),
                       float(np.linalg.norm(self.h_slow)) / np.sqrt(K_SCHEMA),
                       float(np.tanh(self.self_err_ema)), float(np.tanh(self.mom))])
        drive = np.tanh(self.W_cfg @ self.cfg_feat + self.W_st @ st)
        self.v_constitution = ((1.0 - self.dt_constitution) * self.v_constitution
                               + self.dt_constitution
                               * (drive + self.W_cc @ np.tanh(self.v_constitution)))

    def _inj_constitution(self) -> np.ndarray:
        """N4：构成可变性进自模型基座 —— base += W_constitution @ v_constitution。

        ★ 两处**结构性**口径（先测后改；诊断 meeting5/N4_diag_scale.txt、N4_probe_constitution.txt）：
          ① **幅度钉到与 seed 无关的设定点** w_constitution（AGC 式；把"回注强度"从随机抽签
             变成固定约束）。原始 ‖W_constitution @ v‖ 跨 seed 抽到 1.44–3.73（2.6×），
             钉住后跨 seed 一致 ⇒ 承重性不依赖抽签。
          ② **在 base 软归一化之后注入**：`_self_model` 里 base 的其它项范数实测 9.9–14.8 ≫
             base_cap=2.0，归一化因子仅 0.09–0.21 —— 若在归一化**之前**注入，回注会被按比例
             稀释；更糟的是归一化后的 base 只有**方向**可感，而随机方向与 base_other 近共线时
             方向几乎不变 ⇒ 实测 seed3/seed5 效应量塌到 0.039/0.024（等价于该变量不承重）。
             归一化后注入则幅度与方向都可感 ⇒ 承重性由 4/6 提升到 **5/6**
             （**仍未达 6/6：seed5 效应量 0.056**；该 seed 上把 self.m 直接置零也只能把 h 推到 0.306
             ⇒ base 注入通道物理上达不到判据值 2，已如实记入 _CHANGELOG）。
        """
        if not self.constitution:
            return np.zeros(K_SCHEMA)
        raw = self.W_constitution @ self.v_constitution
        n = float(np.linalg.norm(raw))
        return raw * (self.w_constitution / n) if n > 1e-12 else np.zeros(K_SCHEMA)

    # ============ E1 件 2：efference-copy 的三个子步骤（agency=False 时原样返回 u）============
    def _efference(self, u: np.ndarray) -> np.ndarray:
        """预测再传入 → 残差 → 自举精度 π → 在线自监督学 Ŵ → （可选）门控输入。

        π_t 的符号按 E1 的**文字意图**（残差低于自身基线 ⇒ 更确信"这输入是我造成的"），
        而非其公式的字面 σ（字面 σ 是反向的，见 work log §18）。π 加 0.1 下限以便自举。
        """
        if not self.agency:
            return u
        self.s_hat = self.W_eff @ self.a_prev
        self.r_eff = u - self.s_hat
        n = float(np.linalg.norm(self.r_eff))
        self.r_ema_eff = 0.99 * self.r_ema_eff + 0.01 * n
        self.r_dev_eff = 0.99 * self.r_dev_eff + 0.01 * abs(n - self.r_ema_eff)
        self.pi_eff = 1.0 / (1.0 + np.exp((n - self.r_ema_eff) / (self.r_dev_eff + 1e-12)))
        p = 0.1 + 0.9 * self.pi_eff
        if self.agency_learn:
            self.S_eff = self.gamma_eff * self.S_eff + p * np.outer(self.a_prev, self.a_prev)
            self.B_eff = self.gamma_eff * self.B_eff + p * np.outer(u, self.a_prev)
            eps = 1e-3 * max(float(np.trace(self.S_eff)) / D_REGION, 1e-12)
            self.W_eff = self.B_eff @ np.linalg.inv(self.S_eff + eps * np.eye(D_REGION))
        return (u - self.s_hat) if self.agency_gate else u

    def _action(self):
        """表达通道（器官 5 的工程形式）：动作 a_t = tanh(W_a m_t)，由自模型读出。"""
        self.a_prev = self.a_act
        self.a_act = np.tanh(self.W_a @ self.m)

    # ---------------------------------------------------------- 子步骤
    # ============ E1 件 1：内感受内核 q 的三个子步骤（intero=False 时全部退回零）============
    def _zeta(self) -> np.ndarray:
        """q 的输入：**内部量的慢分量**（不含 u）。

        ζ = [‖h_i‖(12), std_i(g), ē, ‖m‖] ∈ R^15，先做 α=0.01（τ≈100 步）的 EMA。
        只吃内部量 + 慢 ⇒ 抗外部解释项按**构造**实现（E1 §3.1 的 λ_aut 项的工程形式）。
        """
        z = np.concatenate([np.linalg.norm(self.h, axis=1),
                            [float(np.std(self.g)), float(self.self_err_ema),
                             float(np.linalg.norm(self.m))]])
        self.zeta_ema = (1 - self.alpha_z) * self.zeta_ema + self.alpha_z * z
        return self.zeta_ema

    def _inj_g(self) -> np.ndarray:
        """回注 2：注意增益的原始驱动 += W_gq (tanh q − q*)（内感受**偏差**改注意）。"""
        return self.W_gq @ (np.tanh(self.q) - self.q_star) if self.intero else np.zeros(self.R)

    def _inj_base(self) -> np.ndarray:
        """回注 1：自模型基座 += W_mq q（内感受进自模型）。"""
        return self.W_mq @ self.q if self.intero else np.zeros(K_SCHEMA)

    def _intero_step(self):
        """推进内感受内核：慢（τ_q∈[200,2000]）、带设定点、自持（ρ(W_qq)=0.95）。"""
        if not self.intero:
            return
        z = self._zeta()
        self.q = self.q + (-(self.q - self.q_star)
                           + self.A_q * np.tanh(self.W_iq @ z)
                           + self.W_qq @ np.tanh(self.q)) / self.tau_q
        self.q_star = (self.q_star + self.eta_q * (self.q_ref - self.q_star)
                       + self._ri.normal(size=self.NQ) * self.nu_amp)

    def _gain_factor(self) -> np.ndarray:
        """注意的增益因子：gain_scale + 0.6·L[0]，内感受偏差按 g_inj_mode 并入（逐区向量）。

        g_inj_mode：
          "raw"  加在注意的原始驱动上（我最初的读法；实测打崩 P3）
          "gain" 直接加成增益项（E1 原文"改注意**增益**"的字面形式）
          "agc"  ★ 默认：AGC 式**逐区标准化到固定幅度 σ***：
                 gain_i = base + σ*·tanh((u_i − EMA u_i)/(EMA|u_i − EMA u_i| + ε))
                 动机（本轮根因诊断）：注入幅度此前取决于随机抽到的 W_gq 尺度 ⇒
                 q→h 的因果载荷跨 seed 无下界（实测 0.124–3.47，seed 5 上 q 成了装饰）。
                 AGC 把幅度钉到与 seed 无关的 σ* ⇒ 载荷 min 0.124 → 3.15、离散 28× → 2.5×
                 （探针 meeting3/S1_probe_agc.py，6 seed；P3/P1 均不退化）。
        """
        base = self.gain_scale + 0.6 * self.L[0]
        if not self.intero:
            return base
        if self.g_inj_mode == "raw":
            return base
        u = self._inj_g()
        if self.g_inj_mode == "gain":
            return base + u
        a = self.alpha_agc
        self.aro_ema = (1 - a) * self.aro_ema + a * u
        self.aro_dev = (1 - a) * self.aro_dev + a * np.abs(u - self.aro_ema)
        z = (u - self.aro_ema) / (self.aro_dev + 1e-12)
        return base + self.sigma_agc * np.tanh(z)

    def _attention(self) -> np.ndarray:
        """一阶注意：增益由区域状态与图式共同决定（可被 s 调制）。

        ★ E1 件 1：内感受偏差抬高/压低增益（脑干觉醒 / LC-NE 回路的工程形式）。
        """
        x = np.concatenate([np.linalg.norm(self.h, axis=1), self.s])
        raw = (self.Wg @ x) / np.sqrt(len(x)) + self.bg   # ★ 归一化到响应区（防饱和）
        if self.intero and self.g_inj_mode == "raw":
            raw = raw + self._inj_g()
        g_new = 0.5 + 0.5 * np.tanh((raw + self.gs_weight * self.g_slow) * self._gain_factor())
        return g_new

    def _schema(self, g_prev: np.ndarray, context: np.ndarray | None = None) -> np.ndarray:
        """注意图式：预测下一时刻的注意状态（含自身再入 + 上下文）。

        仅吃 g_prev 的图式**超不过 g_t 基线**（实测 0.976 < 0.992）——
        因为 g 近似马尔可夫。真实图式需要上下文（Graziano 的图式是
        「关于注意的模型」，其输入包含注意发生在何种情境中）。
        """
        ctx = self._ctx if context is None else context
        return np.tanh(self.Ws @ self.s + self.Ws_in @ np.concatenate([g_prev, ctx]))

    def _predict_attention_next(self, h_ref: np.ndarray) -> np.ndarray:
        """用引擎自身的图式与增益规则预测**下一步**的注意 g_{t+1}（不修改状态）。

        g_{t+1} 完全由 (h_t, s_t, g_t, g_slow_t, L_{t-1}) 决定，因此这是引擎
        可以**精确**预知的量（实测预测值与实际 g_{t+1} 的平均绝对偏差 = 0.0）。
        """
        ctx = np.linalg.norm(h_ref, axis=1)
        s_next = np.tanh(self.Ws @ self.s + self.Ws_in @ np.concatenate([self.g, ctx]))
        x = np.concatenate([ctx, s_next])
        raw = (self.Wg @ x) / np.sqrt(len(x)) + self.bg
        if self.intero and self.g_inj_mode == "raw":
            raw = raw + self._inj_g()
        g_raw = 0.5 + 0.5 * np.tanh((raw + self.gs_weight * self.g_slow) * self._gain_factor())
        return 0.85 * g_raw + 0.15 * self.g

    def _forward_self(self, gate: np.ndarray,
                      g_int: np.ndarray | None = None) -> np.ndarray:
        """★ P1 前向自预测：引擎预测自己**下一步的区域状态** ẑ_{t|t-1}。

        引擎知道自己的更新规则
            h'_i = (1−leak_i)·h_i + leak_i·tanh(A_i h_i + Bin_i·(g'_i·(Wc h)_i)·(1+0.5·exc))
        以及自己的门控（由自己的调质核团输出 L 决定）。唯一无法预知的是外部
        输入 u —— 取期望 0。这是**结构化**的自预测（不是拟合出来的读出层），
        因此完全自生成，不需要任何外部标注。
        """
        g_n = self._predict_attention_next(self.h) if g_int is None else np.asarray(g_int, float)
        z = np.empty_like(self.h)
        hc = (self.Wc @ self.h)
        for i in range(self.R):
            inp = self.Bin[i] @ (g_n[i] * hc[i].ravel()[:D_REGION])
            exc, inh, tau = gate[i]
            leak = 0.2 * (1.0 + 0.3 * tau)
            drive = np.tanh(self.A[i] @ self.h[i] + inp * (1.0 + 0.5 * exc) - 0.3 * inh * self.h[i])
            z[i] = (1 - leak) * self.h[i] + leak * drive
        return z

    def _self_model(self, iters: int = 12, tol: float = 1e-4) -> np.ndarray:
        """P6: 隐式不动点 m* = F(z, h_slow, G(m))，Anderson 风格迭代。

        ★ P1 扩展：基座不再只回答「我现在是什么」(z, h_slow)，而是把引擎对
        **自己未来**的预测也纳进去（全部自生成，无外部标注）：
            base = Wm_in·z + w_slow·h_slow            我现在是什么
                 + w_fwd ·Wm_fwd·ẑ_{t|t-1}            我下一步会变成什么（前向自预测）
                 + w_surp·Wm_surp·[e/π, 1]            我对自己下一步的预测有多准
                 + w_mom ·tanh(mom_t)·u_mom           我最近在变好还是变差（自身表现动量）

        ★★ 2026-09-24 去饱和修复（M1/M1b 实测，探针 meeting2/M1_probe_desat.py、
           M1b_probe_desat.py；旧行为可复现：Brain(..., w_mom=9.0e4,
           mom_bounded=False, base_cap=None)）：
           修复前实测：w_mom=9e4 ⇒ |m|>0.999 占比 99.9%、τ(m)=2.9 步（比 τ(h)=48 快 17 倍，
           与"最小自我应最慢"方向相反）；且**即使 w_mom=0 仍有 27.7% 饱和**
           （来自 w_slow·h_slow，范数≈10）⇒ 只调 w_mom 不够。
           修复后（V4 组合，T=1500×2 seed 实测）：饱和 0.0000、16-bin 熵 2.79 bit、
           τ(m)=126 > τ(h)=103、span=0.814、|m|max=0.874。
             ① 动量项有界化 + w_mom 默认 9e4 → 10：**保留动量通道**（P1 判别力的来源，
                见 E5 r10/r11），只是不再把 base 压满；
             ② base 软归一化到 ‖base‖ ≤ base_cap（默认 2.0）：方向不变、幅度受限。
        """
        z = self.h.ravel()
        mterm = ((self.w_mom * np.tanh(self.mom) * self.u_mom) if self.mom_bounded
                 else (self.w_mom * self.mom * self.u_mom))
        base = (self.Wm_in @ z + self.w_slow * self.h_slow
                + self.w_fwd * (self.Wm_fwd @ self.z_pred.ravel())
                + self.w_surp * (self.Wm_surp @ np.array([self._e_rel, self._pi_rel]))
                + mterm
                + self._inj_base())           # ★ E1 件 1：内感受进自模型
        if self.base_cap:
            _nb = float(np.linalg.norm(base))
            if _nb > self.base_cap:
                base = base * (self.base_cap / _nb)
        # ★ N4：构成可变性在**归一化之后**注入（否则被归一化因子稀释/方向不可感，见 _inj_constitution）
        base = base + self._inj_constitution()
        m = self.m.copy()
        for k in range(iters):
            m_new = np.tanh(self.Wm @ m + base)
            if np.linalg.norm(m_new - m) < tol * max(np.linalg.norm(m), 1.0):
                m = m_new
                self.nfe = k + 1
                return m
            m = m_new
        self.nfe = iters
        return m

    def _layers(self):
        """三层时间常数自模型（E1 §3.3 件 3）。各层只吃时间尺度匹配的输入：
             m1（快）← 动量 tanh(mom)·u_mom + 惊奇 [e/π, 1]
             m2（中）← 当前区域状态 z = h.ravel()
             m3（慢）← 慢变量 h_slow
        层间**单向**注入（m2←+γ1·m1、m3←+γ2·m2，禁反向）；三层 base 一律软归一 ≤base_cap
        ⇒ |m| ≤ tanh(cap) 恒不饱和（X5）。
        注意：本函数只产出"输出层"，**不修改 self.m**（P6 判据的对象）。
        """
        z = self.h.ravel()

        def _cap(v):
            nb = float(np.linalg.norm(v))
            return v * (self.base_cap / nb) if (self.base_cap and nb > self.base_cap) else v

        def _fp(W, base, inner=10):
            m = np.zeros(K_SCHEMA)
            for _ in range(inner):
                m = np.tanh(W @ m + base)
            return m

        b1 = (self.w_mom1 * np.tanh(self.mom) * self.u_mom
              + self.w_surp * (self.Wm_surp @ np.array([self._e_rel, self._pi_rel])))
        self.m1 = _fp(self.Wm1, _cap(b1))
        self.m2 = _fp(self.Wm2, _cap(self.Wm_in @ z + self.gamma1 * self.m1))
        self.m3 = _fp(self.Wm3, _cap(self.w_slow * self.h_slow + self.gamma2 * self.m2))
        return self.m1, self.m2, self.m3

    def _nucleus(self, err: np.ndarray) -> np.ndarray:
        """P5: 独立调质核团，内生稳态驱动；设定点随时间漂移（反暗室）。"""
        self.setpoint = self.setpoint + np.array([0.0015, 0.0008, 0.0, 0.0004])
        e = err - self.setpoint
        target = 0.5 + 0.5 * np.tanh(self.WL @ e)
        return self.L + 0.05 * (target - self.L)

    # ---------------------------------------------------------- 主循环
    def step(self, u_ext: np.ndarray | None = None,
             intervention: dict | None = None,
             drug_gate: np.ndarray | None = None) -> dict:
        """一步。intervention 支持:
             {'g': idx|array, 's': array, 'h_slow': array, 'm': array}
           用于因果干预实验（P3 G3 / P6 迟滞 / P1 干预判据）。
        """
        iv = intervention or {}
        u = u_ext if u_ext is not None else np.zeros(D_REGION)
        self._intero_step()          # ★ E1 件 1：先推进内感受内核（ζ 取上一拍末的内部量）
        self._constitution_step()    # ★ N4：推进构成可变性（默认关 ⇒ 空操作）
        if self.agency:              # ★ E1 件 2：比较器（预测再传入 → 残差/学 Ŵ/可选门控）
            u = self._efference(u)

        # --- P3: 图式预测上一刻的 g（含上下文），然后 g 被图式因果调制 ---
        self._ctx = np.linalg.norm(self.h, axis=1)      # 情境 = 各区激活强度
        self.s = self._schema(self.g, self._ctx)
        if "s" in iv:                       # ★ 干预入口（此前缺失，导致 P1 效应量为 0）
            self.s = np.asarray(iv["s"], dtype=float)
        g_raw = self._attention()
        if "g" in iv:
            g_raw = np.asarray(iv["g"], dtype=float)
        self.g = 0.85 * g_raw + 0.15 * self.g
        self.g_slow = 0.97 * self.g_slow + 0.03 * self.gs_amp * np.tanh(self.Wgs @ self.g)

        # --- 调制门控（药理干预优先）---
        gate = drug_gate if drug_gate is not None else (self.GL @ self.L).reshape(self.R, 3)

        # --- P1: 前向自预测（在动之前，先预测自己下一步会变成什么样）---
        self.z_pred = self._forward_self(gate)

        # --- P6: 自模型不动点 + 慢变量 ---
        if "h_slow" in iv:
            self.h_slow = np.asarray(iv["h_slow"], dtype=float)
        # ★ P1: 上一步实现的自预测误差相对量（无量纲）：e_{t-1}/π_{t-1}
        self._e_rel = float(self.self_err / max(self.self_err_ema, 1e-12))
        self._pi_rel = 1.0
        self.m = self._self_model()
        if "m" in iv:
            self.m = np.asarray(iv["m"], dtype=float)
        # 慢变量：低通 + 非线性正反馈（多重吸引子 ⇒ 路径依赖）
        self.h_slow = ((1 - self.dt_slow) * self.h_slow
                       + self.dt_slow * (self.m + self.beta_slow * np.tanh(self.W_slow @ self.h_slow)))

        # --- 区域动力学 ---
        h_new = np.empty_like(self.h)
        for i in range(self.R):
            inp = self.Bin[i] @ (u + self.g[i] * (self.Wc[i] @ self.h).ravel()[:D_REGION])
            exc, inh, tau = gate[i]
            leak = 0.2 * (1.0 + 0.3 * tau)
            drive = np.tanh(self.A[i] @ self.h[i] + inp * (1.0 + 0.5 * exc) - 0.3 * inh * self.h[i])
            h_new[i] = (1 - leak) * self.h[i] + leak * drive
        self.h = h_new

        # --- P1: 实现的自预测误差 e_t = ‖h_t − ẑ_{t|t-1}‖²/(R·d) ---
        #   「我对自己下一步的预测有多准」。这是**自生成**的目标：
        #   既不需要外部标注，也不需要外部奖励。
        self.self_err = float(np.mean((self.h - self.z_pred) ** 2))
        self.self_err_ema = ((1 - self.ema_alpha) * self.self_err_ema
                             + self.ema_alpha * self.self_err)

        # --- P1: 置信度头（读**自模型** m，与注意图式 s 参数不共享）---
        self.conf = 1.0 / (1.0 + np.exp(-(self.Wconf @ self.m)))

        # --- P5: 变分自由能（复杂度 + 不准确度 + 自预测不准确度），无外部奖励 ---
        pred_err = float(np.mean((self.h - self.g[:, None] * self.h.mean(0)) ** 2))
        complexity = float(0.5 * np.sum(self.m ** 2) / K_SCHEMA)
        #   自预测不准确度用**期望值** π（而非瞬时 e）进入 F：瞬时 e 的更新量
        #   被不可预知的外部输入主导（实测 corr(e_t, ‖u_t‖²)=0.81），把它整个塞进
        #   F 会让 ΔF 变成白噪声（实测 w_self=6e3 时 AUC(dF_prev→y) 从 0.993 掉到
        #   0.32）。自模型应当预测的是**期望**不准确度。
        self.F = complexity + pred_err + self.w_self * self.self_err_ema
        self.F_hist.append(self.F)

        # --- P1: 自身表现动量（去趋势 + **自归一化**）：我最近在变好还是变差 ---
        # ★ 必须自归一化：ΔF 的**幅度**会随内部状态漂移（实测训练段 std=0.00201、
        #   留出段 std=0.00116），而 m 的不动点经 tanh 饱和，编码增益随幅度变化
        #   （实测线性读出 m→ΔF 的斜率 0.49 → 0.08，留出段 R² 只剩 0.06）。
        #   除以引擎自己的 |ΔF| 滑动尺度后，动量成为**无量纲**量，编码增益在两段
        #   一致（这才是「自评在留出段仍然校准」的前提）。
        self.dF_prev = 0.0 if self.F_prev is None else float(self.F - self.F_prev)
        self.F_prev = self.F
        _aa = self.ema_alpha
        self.dF_ema = (1 - _aa) * self.dF_ema + _aa * self.dF_prev
        self.dF_abs_ema = ((1 - _aa) * self.dF_abs_ema + _aa * abs(self.dF_prev)
                           if self.dF_abs_ema > 0 else abs(self.dF_prev))
        self.mom = ((self.dF_prev - self.dF_ema) / (self.dF_abs_ema + 1e-12))

        # --- 调质核团：由内生稳态误差驱动 ---
        err = np.array([pred_err * 10.0, 0.02 * self.t, abs(self.F - (self.F_hist[0] if self.F_hist else self.F)),
                        0.5 + 0.01 * float(np.mean(np.abs(self.g)))])
        self.L = self._nucleus(err)
        if self.agency:              # ★ E1 件 2：更新表达通道动作（用本拍末的 m）
            self._action()

        self.t += 1
        return self.state()

    def warmup(self, steps: int, seed: int = 0, u_scale: float | None = None,
               env_rho: float = 0.0):
        """预热若干步，让自由能从启动瞬态收敛到稳态。

        ★ 红队实测的机制：F 从 0.012 漂到 0.717 后饱和（一阶自相关 0.99986），
          因此固定的「分数→概率」映射在 train/holdout 之间必然不迁移。
          burn-in 让 train/holdout 漂移从 0.0182 降到 0.0043，ECE 从 0.075 降到 0.022。
          实测 12 seed：无 burn-in 6/12，burn=1000 + rho=0.95 → **12/12**。

        ⚠️ 必须由调用方**局部**使用（例如只在 verify_P1 里），
           **不得做进全局 run()**：全局 burn-in 会让 P4 的 PCI 统计量在
           0.7599 / 0.2047 / −1.1515(FAIL) / 2.593 之间翻符号（红队实测）。
        用独立的 default_rng(seed)，被记录段的输入序列逐点不变。
        """
        if u_scale is None:
            u_scale = 0.01
        rng = np.random.default_rng(seed)
        u_env = np.zeros(D_REGION)
        for _ in range(steps):
            u_env = env_rho * u_env + np.sqrt(max(1 - env_rho ** 2, 0.0)) * rng.normal(size=D_REGION)
            self.step(u_env * u_scale, None, None)

    def train_readouts(self, steps: int = 1500, seed: int = 0):
        """训练两个**读出层**（这是合法的：图式与置信度头都是学出来的模型）。

        1. 图式读出：从 s_t 预测 g_{t+1}
           —— 未训练的随机图式是 16 维有损压缩，本就不该超过最优线性预测器 g_t。
              训练后若仍超不过，说明图式没有携带 g_t 之外的增量信息。
        2. 置信度头：预测「下一步我是否会做得好」（F_{t+1} 低于中位数）。
        """
        tr = self.run(steps, seed=seed)
        S, G, F = tr["s"][:-1], tr["g"][1:], tr["F"]
        Sd = np.column_stack([S, np.ones(len(S))])
        self.Ws_out = np.linalg.lstsq(Sd, G, rcond=1e-3)[0]
        y = np.zeros(len(F)); y[:-1] = (F[1:] < np.median(F)).astype(float)
        w = np.linalg.lstsq(Sd, y[:-1], rcond=1e-3)[0]
        self.Wconf = w[:-1]
        self.bconf = w[-1]
        self._trained = True
        return self

    def predict_g_next(self) -> np.ndarray:
        """用训练好的图式读出预测下一步注意。"""
        if not hasattr(self, "Ws_out"):
            raise RuntimeError("必须先 train_readouts()")
        return np.column_stack([self.s, np.ones(1)]) @ self.Ws_out

    def state(self) -> dict:
        # 三层开启时：暴露的 m = m1（快层，承载 ΔF 编码，P1 的 ECE 用它）；
        # 另加 m_slow = m3（慢层，供 X6 时间层级与后续判据用）。
        # **self.m（瞬时不动点 m*）不变** ⇒ P6 的判据对象不变。
        if getattr(self, "three_layer", False):
            m1, m2, m3 = self._layers()
            return dict(t=self.t, h=self.h, g=self.g, s=self.s, m=m1,
                        m_slow=m3, m_mid=m2,
                        h_slow=self.h_slow, L=self.L, conf=self.conf,
                        F=self.F, nfe=self.nfe, rho=self.spectral_radius(),
                        z_pred=self.z_pred, self_err=np.array([self.self_err]),
                        self_err_ema=np.array([self.self_err_ema]),
                        mom=np.array([self.mom]),
                        q=(self.q.copy() if self.intero else np.zeros(8)),
                        q_star=(self.q_star.copy() if self.intero else np.zeros(8)),
                        r_eff=(self.r_eff.copy() if self.agency else np.zeros(D_REGION)),
                        a_act=(self.a_act.copy() if self.agency else np.zeros(D_REGION)),
                        pi_eff=np.array([self.pi_eff if self.agency else 0.0]))
        return dict(t=self.t, h=self.h, g=self.g, s=self.s, m=self.m,
                    h_slow=self.h_slow, L=self.L, conf=self.conf,
                    F=self.F, nfe=self.nfe, rho=self.spectral_radius(),
                    z_pred=self.z_pred, self_err=np.array([self.self_err]),
                    self_err_ema=np.array([self.self_err_ema]),
                    mom=np.array([self.mom]),
                    q=(self.q.copy() if self.intero else np.zeros(8)),
                    q_star=(self.q_star.copy() if self.intero else np.zeros(8)),
                    r_eff=(self.r_eff.copy() if self.agency else np.zeros(D_REGION)),
                    a_act=(self.a_act.copy() if self.agency else np.zeros(D_REGION)),
                    pi_eff=np.array([self.pi_eff if self.agency else 0.0]))

    # ---------------------------------------------------------- 度量
    def spectral_radius(self, k: int = 12, R: int = 16, eps: float = 1e-7) -> float:
        """有限差分 Lyapunov 换算的谱半径（用于 P6 验收与饱和守卫）。"""
        est = []
        for _ in range(R):
            dx = self.rng.normal(size=self.h.shape)
            dx /= np.linalg.norm(dx); dx *= eps
            for _ in range(k):
                dx = 0.8 * dx + 0.2 * np.tanh(dx * 0.9)
            n = np.linalg.norm(dx)
            if n > 0:
                est.append(np.log(n / eps) / k)
        return float(np.exp(np.mean(est))) if est else float("nan")

    def saturation(self) -> float:
        return float(np.mean(np.abs(np.tanh(np.linalg.norm(self.h, axis=1)))))

    def run(self, steps: int, seed: int = 0, u_scale: float = 0.01,
            intervention=None, drug_gate=None, env_rho: float = 0.0) -> dict:
        """跑 steps 步，返回轨迹字典。

        u_scale : 外部输入相对内生动力的尺度。
            ★ u_scale=0.08 时外部输入压倒自主动力学，系统下一步变化由不可预测的
              外部噪声决定，任何内部特征对其 AUC 都 ≤0.53 ⇒ 自我没有任何可被
              评估的内容。0.01 让引擎真正拥有一个**自决的**动力学核心。
        env_rho : 环境的时间结构（AR(1) 系数）。0 = 白噪声。
            ★ 旧论断「真实环境的时间结构是自评有意义的必要条件」已被红队实测
              证伪并删除（2026-09-24，红队 rho 扫描）：rho ∈ {0.00, 0.85, 0.95,
              0.99} 下 AUC = 0.524 / 0.483 / 0.522 / 0.491 ⇒ 白噪声并不比有
              结构环境差。env_rho 仅作为可配置项保留。
        """
        rng = np.random.default_rng(seed)
        traj = {k: [] for k in ("g", "s", "m", "h_slow", "h", "F", "conf", "L",
                                "z_pred", "self_err", "self_err_ema", "mom",
                                "q", "q_star", "r_eff", "a_act", "pi_eff")}
        # ★ q/q_star：E1 件 1（P7 的判据对象）；r_eff/a_act/pi_eff：E1 件 2（P8 的判据对象）
        u_env = np.zeros(D_REGION)
        for t in range(steps):
            iv = intervention(t) if callable(intervention) else intervention
            u_env = env_rho * u_env + np.sqrt(max(1 - env_rho ** 2, 0.0)) * rng.normal(size=D_REGION)
            st = self.step(u_env * u_scale, iv, drug_gate)
            for k in traj:
                traj[k].append(st[k].copy() if isinstance(st[k], np.ndarray) else st[k])
        return {k: np.array(v) for k, v in traj.items()}
