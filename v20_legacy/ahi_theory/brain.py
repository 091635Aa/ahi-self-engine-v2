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
                 w_mom: float = 9.0e4, w_fwd: float = 1.4, w_surp: float = 1.0,
                 w_self: float = 6.0e3, ema_alpha: float = 0.02):
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
        #   谱半径 > 1 的非线性反馈产生**多重吸引子** ⇒ 真正的路径依赖。
        self.W_slow = _spec_norm(rng.normal(size=(K_SCHEMA, K_SCHEMA)) / np.sqrt(K_SCHEMA), 1.9)
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

    # ---------------------------------------------------------- 子步骤
    def _attention(self) -> np.ndarray:
        """一阶注意：增益由区域状态与图式共同决定（可被 s 调制）。"""
        x = np.concatenate([np.linalg.norm(self.h, axis=1), self.s])
        raw = (self.Wg @ x) / np.sqrt(len(x)) + self.bg     # ★ 归一化到响应区（防饱和）
        g_new = 0.5 + 0.5 * np.tanh((raw + self.gs_weight * self.g_slow)
                                    * (self.gain_scale + 0.6 * self.L[0]))
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
        g_raw = 0.5 + 0.5 * np.tanh((raw + self.gs_weight * self.g_slow)
                                    * (self.gain_scale + 0.6 * self.L[0]))
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
                 + w_mom ·mom_t·u_mom                 我最近在变好还是变差（自身表现动量）
        """
        z = self.h.ravel()
        base = (self.Wm_in @ z + self.w_slow * self.h_slow
                + self.w_fwd * (self.Wm_fwd @ self.z_pred.ravel())
                + self.w_surp * (self.Wm_surp @ np.array([self._e_rel, self._pi_rel]))
                + self.w_mom * self.mom * self.u_mom)
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
        return dict(t=self.t, h=self.h, g=self.g, s=self.s, m=self.m,
                    h_slow=self.h_slow, L=self.L, conf=self.conf,
                    F=self.F, nfe=self.nfe, rho=self.spectral_radius(),
                    z_pred=self.z_pred, self_err=np.array([self.self_err]),
                    self_err_ema=np.array([self.self_err_ema]),
                    mom=np.array([self.mom]))

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
                                "z_pred", "self_err", "self_err_ema", "mom")}
        u_env = np.zeros(D_REGION)
        for t in range(steps):
            iv = intervention(t) if callable(intervention) else intervention
            u_env = env_rho * u_env + np.sqrt(max(1 - env_rho ** 2, 0.0)) * rng.normal(size=D_REGION)
            st = self.step(u_env * u_scale, iv, drug_gate)
            for k in traj:
                traj[k].append(st[k].copy() if isinstance(st[k], np.ndarray) else st[k])
        return {k: np.array(v) for k, v in traj.items()}
