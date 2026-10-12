# -*- coding: utf-8 -*-
"""ahi_theory.modality —— R4 单模态插件（文本 · 感知管线验证）原型（AHI 2.2 / Task 0）。

规格出处：`.trae/specs/add-ahi22-modality-plugin/spec.md`（M1–M5）；
任务书 `03_AHI/AHI2.1/完整` §5.4（模块描述符）/ §5.5（三正交维度+时间特性）/
§5.6（四类驱动）/ §5.9（未知模态在线学投影器）。

★★ 复用纪律（Task 0，逐条对齐规格）
  · `brain.py` 一行未改；`mbp.py` 仅**纯追加** `MBP总线.挂投影器()`（既有接口行为与签名不变）；
    `selfprobe.py` / `world.py` 未改（只复用）。
  · **真值（主题 s_t、切换步）只存在于设备实例内部**，仅供评测探针 / M1 事件计数读取；
    投影器学习的全部输入 = 原始流 x_t（结构自证：`观测(x)` / `__call__(x)` 签名只收 x，
    无任何标签入参；学习规则读不到 `真值` / `原型`）。描述符能力只写 ("文本流",)，
    不泄露主题结构（§5.4「内核不知道这是电压」的文本版）。

★★ φ_txt 在线学习规则（预注册；§5.9「发测试信号→观察响应→更新φ」在感知类的退化形式：
    感知类无 CMD 通道 ⇒ 探测循环 = 被动观察原始流，「更新 φ」= 下述规则）
  目标（任务书建议的「投影空间内的时序预测/对比」口径）：最大化每个投影通道输出的
  **时间可预测性**（输出 lag-1 自相关 r）——等价于时序对比目标（正对=相邻帧），
  trivial 常数解由「输入方差归一」（Rayleigh 商分母）排除（SFA 口径）。
  实现（全部在线、标签无关、增量累计 + 周期重构）：
    1. 架构：z = [ W_rand·x (64 维，§5.9「初始化随机φ」块，**永不学习**)；
                  慢特征库 7 尺度 × 64 维 = 448 维 ]，共 512 维（§5.4 投影维度）。
       慢特征库尺度 α_j = 2^{-j}（j=0..6，时间常数 1..64 步）——未知模态 ⇒ 未知时间尺度
       ⇒ 尺度库为架构先验；**哪些尺度/方向/增益可用完全由数据决定**。
    2. 学习前：库通道增益 d ≡ 0（未收敛 ⇒ 无可用投影，§5.9 第 4 步的反面）
       ⇒ sham（φ 冻结于随机映射 = 冻结于该初始状态）的投影流 = [W_rand·x; 0] ⇒ 线性探针 = chance。
    3. 在线统计（对每尺度 j 的 EMA 状态 s_j 直接累计**中心化二阶矩**）：
       C0_j = Σ_t ŝ_t ŝ_t^T（输出 lag-0，Gram ⇒ 半正定）、C1_j = Σ_t ŝ_{t−1} ŝ_t^T（lag-1 交叉矩）、
       其中 ŝ_t = s_j,t − m̂（m̂ = 原始流均值，200 步后冻结；去均值去 DC 伪方向）。
    4. 周期重构（每 200 步，t≥2000）：去中心（减 n·ŝ̄ŝ̄^T）后解广义特征问题
           C1_j v = r · C0_j v
       ⇒ r = 通道(α_j, v) 输出的 lag-1 自相关（Rayleigh 商；解析校验：白噪声输入 ⇒ r = 1−α）。
    5. 通道增益：超额慢度 ϵ = r − (1−α_j)（超出「同尺度白噪声基线」的部分）；
       d = [ max(ϵ, 0) / (1 − r + 0.02) ]²（平方整定：主题通道 d≈1–2、噪声通道 d≲0.05，
       实测线性增益对慢尺度噪声通道抑制不足——见 meeting9/R4_diag3.py 整定记录）。
       瞬时通道（α=1）自动被压制（1−r≈0.94 ⇒ d≈0.02）。
  ⇒ 收敛后：主题子空间方向 × 慢尺度通道被放大（时间积分去噪），投影空间中主题结构
    **线性可读**；瞬时/噪声通道被抑制。整个规则只用 x_t 的二阶时间统计，无标签。

★★ M5 闭环兼容：`闭环感知兼容(自探测循环)` —— 复用 R3 状态机/挂起/重归因，
    仅对 感知类 分支改证据通道：感知类无 CMD 通道（bus.写 结构性拒绝，RESP 置信=0）
    ⇒ 无 self 证据；SENSE 有效到达 ⇒ ρ≡1（指令可解释方差=0）入 world 带，
    同参（窗 20 / 稳定 10 / 阈 0.75）迁移 world。正向对照（执行类→self）同场验证。

★★ 定稿（Task 1.1，2026-09-26f）—— **可选启用**（默认关闭，零侵入）：
  · 本模块**不被任何既有判定路径自动加载**：`verify_all.py` 的 P1–P8 / N / W / S 组均不 import
    本文件；M 组（verify_M1/M2/M4/M5）沿用 R2/R3 的**惰性 import** 模式，仅在本组被显式调用时
    才加载本模块；`run_verify.py` 默认输出与计数零变化（M 组为 `--modality` 纯追加段）。
  · 启用方式（三选一）：
      ① `python run_verify.py --modality`（独立 M 组计分段，与 P1–P8/N/W/S 分开计分）；
      ② 直接调用 `ahi_theory.verify_all.verify_M1(seed) / verify_M2 / verify_M4 / verify_M5`；
      ③ 业务侧显式接入：`from ahi_theory.modality import 文本流描述符, 文本源模块, 慢特征投影器`，
         `bus.注册(文本流描述符(...), 文本源模块(...))` + `bus.挂投影器(mid, 慢特征投影器(...))`
         （`挂投影器` 为 mbp.py 的 R4 纯追加方法；不调用则总线行为与 Task 0 之前逐位一致）。
  · **定稿不改 Task 0 已验证的行为**：双集验证（meeting9/R4_task0_双集.txt）所用的学习规则与
    参数一字未改——本定稿只追加本段注释；预注册学习步数 40000 由调用方显式传入
    （`在线学文本投影(..., 步数=40000)`，与 R4_task0_eval.py 同口径）。
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

try:                     # 与 world.py / selfprobe.py 相同的导入容错
    from .mbp import MBP总线, MBP模块, 模块描述符
except ImportError:      # pragma: no cover
    from ahi_theory.mbp import MBP总线, MBP模块, 模块描述符

try:
    from .selfprobe import 自探测循环, 窗长, 稳定步, 阈world, 带中点
except ImportError:      # pragma: no cover
    from ahi_theory.selfprobe import 自探测循环, 窗长, 稳定步, 阈world, 带中点


# ------------------------------------------------------------ 预注册常量（Task 0 原型，勿按数据回调）
K默认 = 5               # 主题数（≥4）
N默认 = 64              # 原始维度 n
子空间默认 = 4          # 质心子空间维 q = K−1（正则单纯形，散度各向同性）
幅值默认 = 0.45         # a：质心幅度（散度特征值 = a²；原始单帧线性探针 ≈ chance，
                        #   慢尺度时间积分后线性可读——原型整定见 meeting9/R4_sweep*.py）
噪声默认 = 1.0          # σ
切换概率默认 = 0.0025   # 主题马尔可夫切换概率（平均 epoch 400 步；整定理由同上）
尺度数默认 = 7          # 慢特征库尺度数（×64 = 448；+64 随机块 = 512）
最长滞后默认 = 256
预热默认 = 200
重构周期默认 = 200
首次重构默认 = 2000
学习步数默认 = 16000
重放预热 = 256          # 评测重放时丢弃的 EMA 预热帧


def _随机正交(n: int, rng: np.random.Generator) -> np.ndarray:
    """随机正交矩阵（QR，按 R 对角符号归一；与 world.py 同法，本地复制以免交叉依赖）。"""
    Q, R = np.linalg.qr(rng.normal(size=(int(n), int(n))))
    d = np.sign(np.diag(R))
    d[d == 0.0] = 1.0
    return Q * d


def _sha16(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()[:16]


# ================================================================== 文本源（感知类设备）
class 文本源模块(MBP模块):
    """合成主题流文本源（感知类，§5.6「内核读，模块写」）。

    x_t = μ_{s_t} + σ·ε_t；s_t 马尔可夫切换（P(切)=切换概率，均匀切到其它主题）。
    μ_k = a·(随机 q 维子空间中的高斯向量)。

    ★ 真值（`真值` / `切换步` / `原型`）只在设备实例内部，供评测读取；
      内核、描述符、投影器学习规则均不可见。
    ★ `感知(ts)` 幂等：同一 ts 只推进一次（总线读 / 闭环读共享同一采样）。
    """

    def __init__(self, 描述符: 模块描述符, K: int = K默认, 幅值: float = 幅值默认,
                 噪声: float = 噪声默认, 切换概率: float = 切换概率默认,
                 子空间: int = 子空间默认, seed: int = 0, 事件模式: bool = False):
        super().__init__(描述符)
        self.K = int(K)
        self.幅值 = float(幅值)
        self.噪声 = float(噪声)
        self.切换概率 = float(切换概率)
        self.子空间 = int(子空间)
        self.事件模式 = bool(事件模式)
        self.rng = np.random.default_rng(int(seed) * 101 + 7)
        n = self.描述符.原始维度
        # 质心 = **正则单纯形顶点**（修订 2026-09-26f：原为随机高斯 g_k，其散度迹随 seed
        # 波动 ~2×（χ² 类涨落），导致判据考察"源信号运气"而非投影器学习；单纯形使质心
        # 散度**各向同性且强度恒定**（特征值 = a² × q 维），seed 间仅方向/实现不同。
        # 单纯形结构是模块内部设计，描述符与内核均不可见（§5.4）。
        q = min(self.子空间, self.K - 1)
        C = np.eye(self.K) - 1.0 / self.K
        Qf, _ = np.linalg.qr(C)
        Vtx = (C @ Qf[:, :q]) * (self.幅值 * np.sqrt(self.K))   # 散度特征值 = a²·I_q
        Q = _随机正交(n, self.rng)[:, :q]
        self.原型 = Vtx @ Q.T                                   # (K, n)——仅设备内部
        self.主题 = int(self.rng.integers(self.K))
        self.最后帧: Optional[np.ndarray] = None
        self._上次ts: Optional[int] = None
        self._本步切换 = False
        self.真值 = {}                               # {t: s}——仅评测
        self.切换步 = []                             # [t, ...]——仅评测（事件模式的事件时刻）

    def _推进(self, ts: int) -> np.ndarray:
        s旧 = self.主题
        if self.rng.random() < self.切换概率:
            self.主题 = (self.主题 + 1 + int(self.rng.integers(self.K - 1))) % self.K
        self._本步切换 = (self.主题 != s旧)
        x = self.原型[self.主题] + self.噪声 * self.rng.normal(size=self.描述符.原始维度)
        self.最后帧 = x
        self.真值[int(ts)] = self.主题
        if self._本步切换:
            self.切换步.append(int(ts))
        return x

    def 感知(self, 时间戳: int):
        ts = int(时间戳)
        if ts != self._上次ts:
            self._上次ts = ts
            self._推进(ts)
        if self.事件模式 and not self._本步切换:
            return None
        return self.最后帧


def 文本流描述符(模块ID: str = "txt_0", 时间特性: str = "流式", 缓冲深度: int = 1,
                n: int = N默认, 投影维度: int = 512) -> 模块描述符:
    """文本源描述符（§5.4）：感知类 / 时间特性 / 原始维度 n / 投影维度 512 / φ_txt。

    能力只声明 ("文本流",) —— 不泄露主题结构（§5.4「内核不知道这是电压」）。
    `φ_txt` ∉ mbp.已知投影器 ⇒ 总线按未知模态给**可学**投影器（§5.9）。
    """
    return 模块描述符(模块ID, "感知类", 时间特性, int(n), int(投影维度), "φ_txt",
                      采样率="1kHz", 缓冲深度=int(缓冲深度), 代谢常数=0.01, 延迟="1ms",
                      能力=("文本流",), 协议版本="MBP-1.0")


# ================================================================== φ_txt：慢特征在线学习投影器
class 慢特征投影器:
    """φ_txt：R^64 → R^512 的**状态化线性**投影器（多尺度 leaky 积分库 + 在线慢特征重构）。

    结构与学习规则见模块 docstring（预注册）。接口与 `mbp.通用投影器` 兼容的部分：
    `__call__(x)`（总线 读() 逐新样本调用，EMA 状态每调用推进一次）、`指纹()`、`已收敛`。
    学习输入只有 `观测(x)` 收到的原始帧——**无任何标签入参**（结构自证）。
    """

    def __init__(self, 原始维度: int = N默认, 尺度数: int = 尺度数默认, seed: int = 0,
                 最长滞后: int = 最长滞后默认, 预热: int = 预热默认,
                 重构周期: int = 重构周期默认, 首次重构: int = 首次重构默认):
        self.n = int(原始维度)
        self.J = int(尺度数)
        self.输出维度 = self.n + self.J * self.n     # 64 随机块 + 7×64 库 = 512
        self.α = 2.0 ** (-np.arange(self.J))         # [1, 1/2, ..., 1/64]
        self.最长滞后 = int(最长滞后)
        self.预热 = int(预热)
        self.重构周期 = int(重构周期)
        self.首次重构 = int(首次重构)
        rng = np.random.default_rng(int(seed) * 211 + 13)
        # ---- §5.9「初始化随机φ」块：W_rand（64×64，永不学习）----
        self.W随机 = rng.normal(size=(self.n, self.n)) / np.sqrt(self.n)
        # ---- 慢特征库：学习前增益 d≡0（未收敛 ⇒ 无可用投影）----
        self.d = [np.zeros(self.n) for _ in range(self.J)]
        self._M = [np.zeros((self.n, self.n)) for _ in range(self.J)]   # M_j = diag(d)·V^T
        self.s = [np.zeros(self.n) for _ in range(self.J)]              # EMA 状态（原始流）
        # ---- 在线统计：每尺度 EMA 状态的中心化二阶矩 ----
        self._C0 = [np.zeros((self.n, self.n)) for _ in range(self.J)]
        self._C1 = [np.zeros((self.n, self.n)) for _ in range(self.J)]
        self._S1 = [np.zeros(self.n) for _ in range(self.J)]            # Σŝ（去中心用）
        self._nj = [0 for _ in range(self.J)]
        self._prev = [None for _ in range(self.J)]                      # 上一步的 ŝ
        self._m: Optional[np.ndarray] = None
        self._均值累计 = np.zeros(self.n)
        self._均值计数 = 0
        self.已收敛 = False
        self.n重构 = 0

    # -------------------------------------------------- 在线统计（标签无关：只收 x）
    def 观测(self, x) -> None:
        """累计 EMA 状态的中心化二阶矩。`x` = 本步原始帧（状态已在 `__call__` 中推进）。"""
        x = np.asarray(x, dtype=float).ravel()
        if self._m is None:
            self._均值累计 += x
            self._均值计数 += 1
            if self._均值计数 >= self.预热:
                self._m = self._均值累计 / self._均值计数
            return
        for j in range(self.J):
            ŝ = self.s[j] - self._m
            if self._prev[j] is not None:
                self._C1[j] += np.outer(self._prev[j], ŝ)
            self._C0[j] += np.outer(ŝ, ŝ)
            self._S1[j] += ŝ
            self._nj[j] += 1
            self._prev[j] = ŝ

    # -------------------------------------------------- 前向（总线 读() 逐新样本调用）
    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        out = np.empty(self.输出维度)
        out[:self.n] = self.W随机 @ x
        for j in range(self.J):
            self.s[j] += self.α[j] * (x - self.s[j])
            ŝ = self.s[j] if self._m is None else self.s[j] - self._m
            base = self.n + j * self.n
            out[base:base + self.n] = self._M[j] @ ŝ
        return out

    # -------------------------------------------------- 周期重构（慢特征广义特征问题）
    def 重构(self) -> bool:
        if self._m is None or self._nj[0] < self.首次重构 - self.预热:
            return False
        for j in range(self.J):
            n = float(self._nj[j])
            ŝ̄ = self._S1[j] / n
            C0 = self._C0[j] - n * np.outer(ŝ̄, ŝ̄)
            C1 = self._C1[j] - n * np.outer(ŝ̄, ŝ̄)
            ridge = 1e-6 * max(float(np.trace(C0)), 1e-12) / self.n
            w0, U0 = np.linalg.eigh(C0 + ridge * np.eye(self.n))
            w0 = np.clip(w0, 1e-12, None)
            G0m = (U0 * (1.0 / np.sqrt(w0))) @ U0.T         # C0^{-1/2}
            Mm = G0m @ C1 @ G0m
            λ, W = np.linalg.eigh(0.5 * (Mm + Mm.T))
            序 = np.argsort(λ)[::-1]
            r = np.clip(λ[序], 0.0, 0.9999)
            W = W[:, 序]
            Vj = np.empty((self.n, self.n))
            for i in range(self.n):
                v = G0m @ W[:, i]
                nv = float(np.linalg.norm(v))
                Vj[:, i] = v / (nv if nv > 1e-12 else 1.0)
            ϵ = r - (1.0 - self.α[j])
            dj = (np.maximum(ϵ, 0.0) / (1.0 - r + 0.02)) ** 2
            self.d[j] = dj
            self._M[j] = dj[:, None] * Vj.T
        self.n重构 += 1
        return True

    # -------------------------------------------------- 指纹 / 快照 / 冻结重放
    def 指纹(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.W随机, dtype=np.float64).tobytes())
        for j in range(self.J):
            h.update(np.ascontiguousarray(self._M[j], dtype=np.float64).tobytes())
        if self._m is not None:
            h.update(np.ascontiguousarray(self._m, dtype=np.float64).tobytes())
        return _sha16(h.digest())

    def 参数快照(self):
        """当前 φ 参数的深拷贝（W随机 + 各尺度 M_j + 均值 m̂）——供 sham 冻结与评测离线重放。"""
        return (self.W随机.copy(), [self._M[j].copy() for j in range(self.J)],
                None if self._m is None else self._m.copy())

    @staticmethod
    def 冻结重放(参数, α表, 帧阵: np.ndarray, 预热: int = 重放预热) -> np.ndarray:
        """用（冻结的）φ 参数重放帧序列：EMA 状态从零起，丢 `预热` 帧，返回 (T−预热, 512)。"""
        W随机, M表, m̂ = 参数
        J = len(α表)
        n = 帧阵.shape[1]
        s = [np.zeros(n) for _ in range(J)]
        Z = []
        for t in range(帧阵.shape[0]):
            x = 帧阵[t]
            row = np.empty(W随机.shape[0] + J * n)
            row[:W随机.shape[0]] = W随机 @ x
            for j in range(J):
                s[j] += α表[j] * (x - s[j])
                ŝ = s[j] if m̂ is None else s[j] - m̂
                base = W随机.shape[0] + j * n
                row[base:base + n] = M表[j] @ ŝ
            if t >= 预热:
                Z.append(row)
        return np.asarray(Z)


# ================================================================== §5.9 在线学习驱动
def 在线学文本投影(总线: MBP总线, 模块ID: str, 步数: int = 学习步数默认,
                  快照步=()) -> tuple:
    """§5.9 探测循环（感知类退化形式）：被动观察原始流 → 更新统计 → 周期重构 φ。

    每步：总线步进 → 读()（设备推进一次 + φ 前向推进一次 = 内核可见投影流）
          → φ.观测(原始帧) → 到点重构。返回 (帧阵, {步: 参数快照})。
    """
    设备 = 总线.取设备(模块ID)
    φ = 总线.取投影器(模块ID)
    快照 = {0: φ.参数快照()}                     # 步 0 = 随机初始化状态（sham 冻结点）
    想要 = set(int(s) for s in 快照步)
    帧 = np.empty((int(步数), 设备.描述符.原始维度))
    for i in range(1, int(步数) + 1):
        总线.步进(1)
        总线.读(模块ID)
        x = 设备.最后帧
        φ.观测(x)
        if i >= φ.首次重构 and (i % φ.重构周期 == 0):
            φ.重构()
        帧[i - 1] = x
        if i in 想要:
            快照[i] = φ.参数快照()
    return 帧, 快照


# ================================================================== 评测用线性探针（真值只进这里）
#: 预注册（评测窗 = 学习流**最后 12000 帧**，重放丢 256 预热后 11744 行）：
#: **分层采样**——测试块 = 窗内 [8000:11700] 每类均匀取 24 行（完全留出）；
#: 训练块 = 测试块**以外**的全部非过渡帧，每类均匀取 30 行（保证 K 类覆盖：
#: 若按"隔 1000 步的训练块"取，实测 P(某类在训练块缺席)≈3.5%/块（seed1：类2 训练 0 行
#: ⇒ 该类测试行全错 ⇒ 0.800 恰为 4/5），扩为"补集"后 P≈0.8%）。
#: **切换后过渡帧剔除**（`排除` 掩码，评测侧真值使用、仅行剔除、对真值/sham 同口径）：
#: 慢尺度 EMA 在主题切换后 ~τ 步内物理上是前后主题的混合，其帧标签本身有歧义；
#: 剔除切换后 ≤64 步（=最大尺度时间常数）的帧后，探针量的是稳态可读性。
#: 修订记录：初版"训练/测试相隔 ≥1000 步无 epoch 共享"因类覆盖问题放宽为上述补集口径，
#: 在此如实登记。真值标签 y **只在此处**使用（评测探针的行选择与训练），从不进入 φ 的
#: 任何更新路径。
测试范围 = (8000, 11700)
每类训练 = 30
每类测试 = 24
切换剔除步数 = 64


def _分层行(y: np.ndarray, K: int, 起止, 每类: int, 排除=None) -> np.ndarray:
    行 = []
    for k in range(K):
        idx = np.asarray([i for i in range(*起止)
                          if y[i] == k and (排除 is None or not 排除[i])], dtype=int)
        if len(idx) == 0:
            continue
        take = min(每类, len(idx))
        sel = np.linspace(0, len(idx) - 1, take).astype(int)
        行.extend(idx[sel].tolist())
    return np.asarray(sorted(行), dtype=int)


def 切换剔除掩码(y窗: np.ndarray, 剔除步数: int = 切换剔除步数) -> np.ndarray:
    """按标签跳变生成「切换后剔除」掩码（评测侧真值使用；每窗口行数 = len(y窗)）。"""
    y窗 = np.asarray(y窗, dtype=int)
    排 = np.zeros(len(y窗), dtype=bool)
    for s in np.flatnonzero(np.diff(y窗) != 0) + 1:      # 新主题首帧
        排[s:s + 剔除步数 + 1] = True
    return 排


def 线性探针准确率(Z: np.ndarray, y: np.ndarray, K: int,
                  λ: float = 1.0, 通道数: int = 36,
                  测试范围=测试范围, 每类训练=每类训练, 每类测试=每类测试,
                  排除=None) -> float:
    """线性判别探针（两阶段，预注册）：

    阶段 1（通道选择）：在**分层训练块**上按 ANOVA 类间/类内方差比排名，取前 `通道数`=36
    通道（512 维下裸岭回归受维数惩罚，实测 0.815；选通后 ≈ oracle 上界 0.94——
    见 meeting9/R4_diag4.py 口径对照。m=36 兼顾准确率与检查点间稳定，见 R4_diag7.py）。
    阶段 2：对选中通道做岭回归 one-hot（训练集逐特征标准化，λ=1），argmax 计准确率。
    """
    y = np.asarray(y, dtype=int)
    n = len(y)
    t0, t1 = 测试范围
    掩 = np.ones(n, dtype=bool)                      # 训练候选 = 测试块以外的全部行
    掩[t0:t1] = False
    if 排除 is not None:
        掩 &= ~np.asarray(排除, dtype=bool)[:n]
    训练 = _分层行(y, K, (0, n), 每类训练, ~掩)
    测试 = _分层行(y, K, 测试范围, 每类测试, 排除)
    Ztr, ytr = Z[训练], y[训练]
    mu = Ztr.mean(0)
    sd = np.maximum(Ztr.std(0), 1e-12)
    A = (Ztr - mu) / sd
    ntr = A.shape[0]
    列 = np.arange(A.shape[1])
    if 通道数 is not None and 通道数 < A.shape[1]:
        B = np.zeros(A.shape[1])
        for k in range(K):
            m = ytr == k
            if m.any():
                B += int(m.sum()) * (A[m].mean(0) - A.mean(0)) ** 2
        F = B / (A.var(0) + 1e-12)
        列 = np.argsort(F)[::-1][:通道数]
    As = A[:, 列]
    Y = np.eye(K)[ytr]
    W = np.linalg.solve(As.T @ As + λ * ntr * np.eye(As.shape[1]), As.T @ Y)
    Zte = ((Z[测试] - mu) / sd)[:, 列]
    pred = np.argmax(Zte @ W, axis=1)
    return float(np.mean(pred == y[测试]))


# ================================================================== M5：感知类兼容的 R3 闭环
class 闭环感知兼容(自探测循环):
    """R3 闭环的感知类兼容扩展（`selfprobe.py` 未改，纯子类）。

    感知类分支（预注册）：
      · CMD 试探：bus.探测 → RESP 为**降级包（置信=0）**——总线对感知类的结构性拒绝
        （§5.6：感知类无写通道）⇒ **无 self 证据**（不产生任何 ρ 样本进 self 带）；
      · SENSE 读：包有效（置信=1 且校验通过）⇒ ρ≡1（指令可解释方差=0，无 CMD 通道）
        入窗；窗满后按 R3 同参（窗 20 / 稳定 10 / 阈world 0.75）迁移 world；
      · 拔出挂起 / 插回重归因 / 异常隔离：全部继承基类 `步进`，不改。
    CMD 类模块走基类路径（`_探测一次` 原样），指令流 `rng指令` 不被感知分支消耗
    ⇒ 与"无文本源"对照run的指令序列逐位一致（共存互不干扰可逐位核对）。
    """

    def __init__(self, 总线: MBP总线, 学习: bool = True,
                 冻结方式: Optional[str] = None, seed: int = 0):
        super().__init__(总线, 学习=学习, 冻结方式=冻结方式, seed=seed)
        self.rng感知指令 = np.random.default_rng(int(seed) * 57 + 11)

    def _探测一次(self, mid: str, ev, t: int) -> None:
        d = self.bus.查询(mid)
        if d is not None and d.模块类 == "感知类":
            a = self.rng感知指令.normal(size=d.投影维度)
            self.bus.探测(mid, a, 学习=False)          # RESP = 降级包（置信 0）：无 self 证据
            p = self.bus.读(mid)                        # SENSE：世界侧证据
            if p.置信 == 1.0 and p.验证():
                ev.探测数 += 1
                ev.窗.append(1.0)                       # ρ ≡ 1
            if len(ev.窗) >= 窗长 and ev.状态 == "unknown":
                ρ̄ = float(np.mean(ev.窗))
                if ρ̄ >= 阈world:
                    ev.world连击 += 1
                    ev.self连击 = 0
                else:
                    ev.self连击 = 0
                    ev.world连击 = 0
                if ev.world连击 >= 稳定步:
                    ev.状态 = "world"
                    ev.收敛步 = ev.探测数
                    self.迁移日志.append((t, mid, "world", ev.收敛步, ρ̄,
                                          min(1.0, abs(ρ̄ - 带中点) / 带中点)))
            if ev.状态 in ("self", "world") and len(ev.窗):
                ev.置信 = min(1.0, abs(float(np.mean(ev.窗)) - 带中点) / 带中点)
            ev.事件流.append((t, float("nan"), 1.0, float(1.0 / (1.0 + np.exp(1.0))),
                              float("nan"), ev.状态))
            return
        super()._探测一次(mid, ev, t)                   # 执行类/双向类：R3 原路径
