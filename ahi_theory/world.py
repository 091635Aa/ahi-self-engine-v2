# -*- coding: utf-8 -*-
"""ahi_theory.world —— R2 世界框架的**最小世界**（AHI 2.2）。

规格出处：`03_AHI/AHI2.1/完整` §6（6.1 最小定义 / 6.2 最小实现 / 6.3 接入 / 6.4 判据 W1–W7）；
`.trae/specs/add-ahi22-world-framework/spec.md` 与 `tasks.md` 的 Task 0。

★ 本文件只提供两样东西：
  1. `世界` —— §6.2 的最小世界（稳定线性动力学 `A=正交×0.95` + 动作影响 `B` + 小噪声）。
  2. `世界模块` —— 把 `世界` 包成 **MBP 双向类** 设备（`感知`/`响应`/`探针响应`/`标定`），
     供 `mbp.MBP总线.注册()` 挂载；**不新增内核接口**（内核只用总线既有 `交换()`/`读()`/`写()`）。

★ 设计纪律（本项目通用）
  · **不 import `brain.py`**（内核一行不改），也不被 `brain.py` import ⇒ 零锚点代价。
  · 判据可失败、非恒真：本文件只承载"世界"结构；W1–W7 的可失败判据与 sham 对照在 `meeting7/` 探针里。
  · 未实测数字一律标注；失败如实写。

三条必要 + 两条排除（§6.1）的**结构保证**：
  1 独立   —— 状态在无动作时按自身 `A@状态` 演化（`感知` 为只读，不改状态）
  2 可作用 —— `响应(a)` 的即时效果 `Δ = B@a (+ 非线性) + ξ` 是 `a` 的可学映射
  3 可观测 —— `感知()` 返回状态 ⇒ 经总线 `读()` 成为 SENSE 包
  4 不完全可预测 —— 响应含噪声 `ξ` ⇒ 在线 `f` 的预测误差 `e > 0`
  5 不完全随机   —— 响应含结构 `B@a` ⇒ `e` 远小于"纯随机"上界
（以上五条由 `meeting7/R2_probe_world.py` 实测，逐条给数字。）
"""
from __future__ import annotations

import numpy as np

try:                     # 允许 `world.py` 单独被 import（不强制依赖包上下文）
    from .mbp import MBP模块, 模块描述符
except ImportError:       # pragma: no cover
    from ahi_theory.mbp import MBP模块, 模块描述符


稳定因子 = 0.95          # §6.2：A = 随机正交矩阵 × 0.95 ⇒ 谱半径 0.95 < 1（稳定、衰减、不发散）


def 随机正交矩阵(n: int, rng: np.random.Generator) -> np.ndarray:
    """随机正交矩阵：对高斯矩阵做 QR，并按 R 的对角符号归一，保证 `Q^T Q = I` 且**可复现**。"""
    Q, R = np.linalg.qr(rng.normal(size=(int(n), int(n))))
    d = np.sign(np.diag(R))
    d[d == 0.0] = 1.0
    return Q * d


class 世界:
    """§6.2 的最小世界。

    参数
    ----
    维度      : 状态维度（＝ MBP 原始维度）
    动作维度  : 动作维度（默认等于状态维度）
    噪声      : 过程/观测噪声标准差 σ（>0 ⇒ 不完全可预测）
    增益      : 动作通道增益尺度（`B` 的元素 ~ N(0, 增益²)）
    非线性    : 附加**非线性**动作通道系数 β，`Δ += β·tanh(W_nl @ a)`（β>0 ⇒ 更复杂、信号更强）
    种子      : 随机种子（A/B/初态/噪声全部由它派生 ⇒ 可复现）
    """

    def __init__(self, 维度: int = 16, 动作维度: int = None, 噪声: float = 0.01,
                 增益: float = 0.1, 非线性: float = 0.0, 种子: int = 0,
                 私有耦合: float = 0.0, 稳定因子: float = None):
        self.维度 = int(维度)
        self.动作维度 = int(动作维度 if 动作维度 is not None else 维度)
        self.噪声 = float(噪声)
        self.增益 = float(增益)
        self.非线性 = float(非线性)
        self.种子 = int(种子)
        self.私有耦合 = float(私有耦合)      # #127 C*′：κ_w=增益/2 预注册；0.0 缺省=逐位不变
        # #156 高响应世界：稳定因子=None⇒模块常量 0.95（现有消费方逐位不变）；0.8=重基线体制
        self.稳定因子 = float(稳定因子) if 稳定因子 is not None else globals()["稳定因子"]
        self.rng = np.random.default_rng(int(种子))

        self.A = 随机正交矩阵(self.维度, self.rng) * self.稳定因子   # 稳定动力学
        self.B = self.rng.normal(size=(self.维度, self.动作维度)) * self.增益
        # 非线性动作通道（β=0 时退化为纯线性，与 §6.2 一致）
        self.W_nl = self.rng.normal(size=(self.维度, self.动作维度)) * self.增益
        self.状态 = self.rng.normal(size=self.维度)
        self.t = 0

    # ------------------------------------------------------------ 自身动力学
    def 动作影响(self, 动作) -> np.ndarray:
        """动作对世界的即时效果 `Ga`（G 即动作增益；线性部分 B + 非线性部分）。**不含噪声。**"""
        a = np.asarray(动作, dtype=float).ravel()
        Δ = self.B @ a
        if self.非线性 != 0.0:
            Δ = Δ + self.非线性 * np.tanh(self.W_nl @ a)
        return Δ

    def _推进(self, 动作=None, s_priv=None) -> np.ndarray:
        """一步动力学：`x ← A x + Ga + ξ`；返回**动作的即时效果** `Ga + ξ`（作为 RESP 载荷）。
        #127 C*′：s_priv=引擎私有熵投影（16 维，引擎自产、公共流不可恢复）⇒ 私有状态成为
        联合系统隐变量（κ_w·s_priv 进转移核）；s_priv=None（缺省）⇒ 与原式逐位不变。"""
        影响 = np.zeros(self.维度) if 动作 is None else self.动作影响(动作)
        ξ = self.rng.normal(size=self.维度) * self.噪声
        私有项 = (self.私有耦合 * np.asarray(s_priv, dtype=float).ravel()
                  if (self.私有耦合 != 0.0 and s_priv is not None) else 0.0)
        self.状态 = self.A @ self.状态 + 影响 + ξ + 私有项   # 自身动力学 + 动作 + 噪声 + 私有调制
        self.t += 1
        return 影响 + ξ

    def 步(self, 动作=None) -> np.ndarray:
        """推进一步（无动作则纯自身动力学）；返回新状态（可观测）。"""
        self._推进(动作)
        return self.状态

    # ------------------------------------------------------------ MBP Device 侧接口
    def 感知(self, 时间戳: int) -> np.ndarray:
        """SENSE：**只读**返回当前状态（内核观测**不改**世界动力学 ⇒ 条件 1 独立）。"""
        return self.状态.copy()

    def 响应(self, 指令向量, s_priv=None) -> np.ndarray:
        """CMD→RESP：发动作 a，世界前进一步；RESP = 动作的即时效果 `Ga + ξ`（可被在线 `f` 预测）。
        #127 C*′：s_priv=私有熵投影（None 缺省 ⇒ 逐位不变）。"""
        return self._推进(指令向量, s_priv=s_priv)

    def 标定(self, 标定向量) -> np.ndarray:
        """CAL：把标定向量投影到状态维（用于 MBP B7 往返）。"""
        v = np.asarray(标定向量, dtype=float).ravel()
        out = np.zeros(self.维度)
        m = min(len(v), self.维度)
        out[:m] = v[:m]
        return out


class 世界模块(MBP模块):
    """把 `世界` 包成 **MBP 双向类** 设备（§6.3）。

    · `时间特性 = 流式`、`原始维度 = 世界维度`、`模块类 = 双向类`、`投影器 = φ_world`（未知模态 ⇒ 可在线学）。
    · `探针响应(x)`：世界自身的固定编码 `G_world @ x`，供总线 §5.9 在线学 `φ_world`（内核母语化）。
    · 世界**不新增任何内核接口**：内核只经总线 `读()`/`写()`/`交换()` 与之交互。
    """

    def __init__(self, 描述符: 模块描述符, 世界实例: 世界, 编码: np.ndarray = None, seed: int = 0):
        super().__init__(描述符)
        self.世界 = 世界实例
        if 编码 is None:
            rng = np.random.default_rng(int(seed))
            d = 描述符.投影维度
            n = 描述符.原始维度
            self.编码 = rng.normal(size=(d, n)) / np.sqrt(max(n, 1))
        else:
            self.编码 = np.asarray(编码, dtype=float)

    def 感知(self, 时间戳: int) -> np.ndarray:
        return self.世界.感知(时间戳)

    def 响应(self, 指令向量) -> np.ndarray:
        return self.世界.响应(指令向量)

    def 探针响应(self, 原始信号) -> np.ndarray:
        return self.编码 @ np.asarray(原始信号, dtype=float).ravel()

    def 标定(self, 标定向量) -> np.ndarray:
        return self.世界.标定(标定向量)


class 随机扰动模块(MBP模块):
    """★ 外部扰动源（双向类）：`响应(a) = η`，与指令**无关**、纯随机 ⇒ 任何 `f` 都学不出来。

    用于 W6 的对照：与"我的动作（世界）"**同形同尺度**，但**非我所造成**。
    """

    def __init__(self, 描述符: 模块描述符, 尺度: float = 0.4, seed: int = 0):
        super().__init__(描述符)
        self.尺度 = float(尺度)
        self.rng = np.random.default_rng(int(seed))

    def 响应(self, 指令向量) -> np.ndarray:
        return self.rng.normal(size=self.描述符.投影维度) * self.尺度

    def 探针响应(self, 原始信号) -> np.ndarray:
        return self.rng.normal(size=self.描述符.投影维度)

    def 标定(self, 标定向量) -> np.ndarray:
        return np.zeros(self.描述符.投影维度)


def 世界描述符(维度: int = 16, 投影维度: int = 16, 模块ID: str = "world",
               投影器: str = "φ_world") -> 模块描述符:
    """构造世界的 MBP 描述符（§6.3：双向类 / 流式 / 原始维度=世界维度 / φ_world 可在线学）。

    注意：`投影器="φ_world"` **不在** `mbp.已知投影器` 内 ⇒ 总线按"未知模态"给**可学**投影器（§5.9）。
    """
    return 模块描述符(模块ID, "双向类", "流式", int(维度), int(投影维度), 投影器,
                      采样率="1kHz", 缓冲深度=1, 代谢常数=0.01, 延迟="1ms",
                      能力=("世界状态",), 协议版本="MBP-1.0")
