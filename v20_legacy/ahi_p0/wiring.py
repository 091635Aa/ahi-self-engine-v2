# -*- coding: utf-8 -*-
"""ahi_p0.wiring —— P0 接线层：调质核团 / 耦合总线 / 器官（含引力接线）

修复的三项：
  P0-5  独立调质核团（器官自产调质必然共线，实测有效秩 = 1.0）
  P0-1  耦合总线有接收方（现状是只写广播，TE 与独立随机信号不可区分）
  P0-2  身份引力真正作用于持久状态（现状：引力() 全目录零调用）
"""
from __future__ import annotations
import numpy as np

D_LATENT = 32          # 器官私有维度
D_MOTHER = 64          # 母语维度
N_ORGANS = 7
ORGAN_NAMES = ["感知", "感受", "推理", "记忆", "表达", "元认知", "身份锚定"]


# ======================================================================
# P0-5  独立调质核团
# ======================================================================
class ModulatoryNucleus:
    """独立核团 —— L(t) = [ACh, NE, DA, 5HT] + 唤醒

    **不由器官自产**。理由（实测）：7 个器官的 E/sigma 组合有效秩 = 1.0，
    自产调质必然共线，无法承载 4 个独立通道。真实的蓝斑 / 中缝核 /
    基底前脑 / VTA 都是独立解剖核团，弥散投射全脑。

    驱动源：**内生稳态偏差**（非外部旋钮）。
    """

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_ch = 5
        self.W = rng.normal(size=(self.n_ch, 4)) * 0.8
        self.b = np.zeros(self.n_ch)
        self.state = np.full(self.n_ch, 0.5)
        # 每个器官 × 每层一个增益读出矩阵（E1: 全局标量会丢掉 80% 表达力）
        self.G = rng.normal(size=(N_ORGANS, 3, self.n_ch)) * 0.5
        # 稳态设定点（会漂移 —— 这是反退化的结构条件）
        self.setpoint = np.array([1.0, 0.0, 0.0, 0.0])
        self.t = 0

    def step(self, homeostatic_err: np.ndarray, dt: float = 0.05) -> np.ndarray:
        """homeostatic_err = [资源赤字, 边界磨损, 预测误差累积, 负载]"""
        self.t += 1
        # 设定点随时间漂移 ⇒ "不动"必然漂出安全区（封堵暗室解）
        self.setpoint = self.setpoint + np.array([0.002, 0.001, 0.0, 0.0005])
        e = np.asarray(homeostatic_err, float) - self.setpoint
        target = np.tanh(self.W @ e + self.b) * 0.5 + 0.5
        self.state = self.state + dt * (target - self.state)
        return self.state

    def gain(self, organ_idx: int) -> np.ndarray:
        """返回该器官的 3 维增益 [兴奋增益, 抑制增益, 时间常数调制]。"""
        return self.G[organ_idx] @ self.state

    @property
    def channels(self) -> dict:
        n = ["ACh", "NE", "DA", "5HT", "唤醒"]
        return {k: float(v) for k, v in zip(n, self.state)}


# ======================================================================
# P0-1  耦合总线（有接收方）
# ======================================================================
class Packet:
    """广播包 —— 字段与 核心/接口协议.py 对齐。"""
    __slots__ = ("v", "E", "t", "i", "sigma", "tau")

    def __init__(self, v, E, t, i, sigma, tau):
        self.v, self.E, self.t, self.i = v, float(E), int(t), int(i)
        self.sigma, self.tau = float(sigma), float(tau)


class CouplingBus:
    """广播 + **接收**。

    现状缺陷：器官只写广播，从没有任何器官读取别的器官的 v/sigma/tau
    （AST 复核：sigma 与 tau 在包外无人读取）。后果是成对 TE 与独立
    随机信号不可区分（实测 0.00275 vs 零假设 0.00269）。

    本实现给出明确的三条接收路径：
      R1  v      -> 输入耦合（加权求和进对方动力学）
      R2  sigma  -> 输入增益缩放（置信度加权）
      R3  tau    -> 时间常数调制（对方的泄漏率被对方的时间尺度影响）
    """

    def __init__(self, seed: int = 0, coupling_scale: float = 3.0):
        rng = np.random.default_rng(seed + 1)
        self.W = rng.normal(size=(N_ORGANS, N_ORGANS, D_MOTHER, D_MOTHER)) / np.sqrt(D_MOTHER)
        for _i in range(N_ORGANS):          # 无自环 + 块范数归一
            for _j in range(N_ORGANS):
                if _i == _j:
                    self.W[_i, _j] = 0.0
                else:
                    _n = np.linalg.norm(self.W[_i, _j])
                    if _n > 0:
                        self.W[_i, _j] /= _n      # 每个块 Frobenius 范数 = 1
        self.coupling_scale = coupling_scale
        self.packets: dict[int, Packet] = {}
        self.mask_v = self.mask_sigma = self.mask_tau = False   # 消融开关

    def broadcast(self, pkt: Packet) -> None:
        self.packets[pkt.i] = pkt

    def receive(self, i: int) -> tuple[np.ndarray, float, float]:
        """返回 (耦合输入, sigma 增益, tau 调制)。"""
        v_in = np.zeros(D_MOTHER)
        sig_acc, tau_acc, wsum = 0.0, 0.0, 0.0
        for j, p in self.packets.items():
            if j == i:
                continue
            if not self.mask_v:
                v_in += self.W[i, j] @ p.v
            if not self.mask_sigma:
                sig_acc += p.sigma
            if not self.mask_tau:
                tau_acc += p.tau
            wsum += 1.0
        if wsum > 0:
            sig_acc /= wsum
            tau_acc /= wsum
        else:
            sig_acc, tau_acc = 1.0, 1.0
        # 接收方真的使用了 sigma 与 tau
        gain = 0.5 + sig_acc                       # sigma 调制输入增益
        tau_mod = 1.0 / (1.0 + np.exp(-(tau_acc - 1.0)))  # tau 调制泄漏率
        return self.coupling_scale * v_in, float(gain), float(tau_mod)


# ======================================================================
# 器官（含 P0-2 引力接线）
# ======================================================================
class IdentityAnchor:
    """身份锚点 —— 固定方向（与原项目一致：由人设卡哈希得到）。"""

    def __init__(self, dim: int = D_MOTHER, theta: float = 0.5, seed: int = 7):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=dim)
        self.a = a / np.linalg.norm(a)
        self.theta = theta
        self.kappa, self.eta = 1.0, 0.5

    def deviation(self, v: np.ndarray) -> float:
        return float(1.0 - np.dot(v, self.a) / (np.linalg.norm(v) + 1e-12))

    def gravity(self, v: np.ndarray) -> tuple[np.ndarray, bool]:
        """正交拉回：只削垂直分量，沿 a 分量严格不变。

        **这是 P0-2 的修复点**：在现项目中此函数全目录零调用，
        因此能量恒为 0.1505 而非 0.25。
        """
        d = self.deviation(v)
        if d <= self.theta:
            return v, False
        par = np.dot(v, self.a) * self.a
        perp = v - par
        n = np.linalg.norm(perp)
        if n < 1e-12:
            return v, True
        strength = (d - self.theta) / max(1.0 - self.theta, 1e-12)
        step = min(self.eta * self.kappa * strength, n)
        return v - step * (perp / n), True


class Organ:
    """一个器官：私有潜状态 -> 对齐到母语空间 -> 广播。"""

    def __init__(self, idx: int, seed: int = 0):
        rng = np.random.default_rng(seed + 100 + idx)
        self.idx = idx
        self.A = rng.normal(size=(D_LATENT, D_LATENT)) * 0.3
        # ★ 关键修复：把递归矩阵的**谱半径**归一到 0.9。
        #   未归一化时 rho(A) ~= 3.4（不稳定）⇒ h 爆到 tanh 饱和区
        #   ⇒ 器官输出退化为 sign(h)，任何耦合指标都是饱和产物。
        _rho = float(max(abs(np.linalg.eigvals(self.A))))
        if _rho > 0:
            self.A = self.A / _rho * 0.9
        self.B = rng.normal(size=(D_LATENT, D_MOTHER)) / np.sqrt(D_MOTHER)
        self.phi = rng.normal(size=(D_MOTHER, D_LATENT)) / np.sqrt(D_LATENT)
        self.h = rng.normal(size=D_LATENT) * 0.1
        self.tau = 1.0
        self.sigma = 1.0
        self.gate_nuc = np.zeros(3)

    def step(self, u_ext: np.ndarray, v_couple: np.ndarray, gain: float,
             tau_mod: float, gravity: IdentityAnchor, use_gravity: bool,
             fixed_modulation: bool = False) -> Packet:
        # --- 时间常数被接收到的 tau 调制（P0-1 的 R3 通路）---
        alpha = float(np.clip(0.2 * tau_mod * (1.0 + 0.3 * self.gate_nuc[2]), 0.01, 0.9))
        inp = self.B @ (u_ext + v_couple) * gain * (1.0 + 0.5 * self.gate_nuc[0])
        h_new = (1 - alpha) * self.h + alpha * np.tanh(self.A @ self.h + inp)
        self.h = h_new

        v = self.phi @ self.h
        # --- P0-2：引力真正作用于持久向量 ---
        if use_gravity:
            v, _ = gravity.gravity(v)
        n = np.linalg.norm(v)
        if fixed_modulation:
            # 孪生体：调质通路固定为常数 ⇒ 屏蔽 sigma/tau 必然是无操作
            self.sigma, self.tau = 1.0, 1.0
        else:
            self.sigma = float(1.0 / (1.0 + np.linalg.norm(h_new)))
            self.tau = float(1.0 + 0.5 * np.linalg.norm(h_new))
        self._v = v / (n + 1e-12)
        return Packet(self._v, float(n), 0, self.idx, self.sigma, self.tau)


# ======================================================================
# 系统
# ======================================================================
class System:
    """七器官 + 调质核团 + 耦合总线 + 修正能量度量。"""

    def __init__(self, seed: int = 0, use_gravity: bool = True,
                 use_nucleus: bool = True, fixed_modulation: bool = False):
        self.bus = CouplingBus(seed)
        self.nucleus = ModulatoryNucleus(seed)
        self.gravity = IdentityAnchor()
        self.organs = [Organ(i, seed) for i in range(N_ORGANS)]
        self.use_gravity = use_gravity
        self.use_nucleus = use_nucleus
        self.fixed_modulation = fixed_modulation
        self.drug_gate: np.ndarray | None = None   # 药理类似物的门控覆盖
        self.t = 0
        self.homeo = np.array([1.0, 0.0, 0.0, 0.0])

    def step(self, u_ext: np.ndarray | None = None):
        u = u_ext if u_ext is not None else np.zeros(D_MOTHER)
        mods = self.nucleus.step(self.homeo) if self.use_nucleus else np.full(5, 0.5)
        pkts = []
        for o in self.organs:
            v_c, g, tm = self.bus.receive(o.idx)
            if self.fixed_modulation:
                g, tm = 1.0, 0.5          # 调质通路固定为常数（阴性对照）
            if self.drug_gate is not None:
                o.gate_nuc = self.drug_gate          # 药理干预优先
            elif self.use_nucleus:
                o.gate_nuc = self.nucleus.gain(o.idx)
            p = o.step(u, v_c, g, tm, self.gravity, self.use_gravity,
                       self.fixed_modulation)
            pkts.append(p)
        for p in pkts:
            self.bus.broadcast(p)
        # 内生稳态误差由器官活动驱动（非外部奖励）
        self.homeo = np.array([
            1.0 + 0.05 * np.mean([p.E for p in pkts]),
            0.02 * self.t,
            0.1 * np.std([p.E for p in pkts]),
            0.5 + 0.01 * np.mean([o.sigma for o in self.organs]),
        ])
        self.t += 1
        return pkts

    def signals(self, n: int, seed: int = 0):
        """跑 n 步，返回 [n, 7] 的信号矩阵（取 v 的第一维）。"""
        rng = np.random.default_rng(seed)
        out = np.empty((n, N_ORGANS))
        for t in range(n):
            u = rng.normal(size=D_MOTHER) * 0.08
            pkts = self.step(u)
            out[t] = [p.v[0] for p in pkts]
        return out

