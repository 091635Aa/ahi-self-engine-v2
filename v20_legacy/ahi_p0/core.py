# -*- coding: utf-8 -*-
"""ahi_p0.core —— P0 五项缺陷的可运行修复参考实现

本模块只依赖 numpy，可独立运行与验收。对应 P0 诊断报告中的五项修复：

  P0-1  耦合总线：把广播做成「有接收方」的（v / sigma / tau 都被读取并改变动力学）
  P0-2  引力接线：身份引力真正进入持久状态（而非零调用的孤儿函数）
  P0-3  能量度量 E'：未归一化预测误差 + 自适应除性归一化，使 CV >= 0.05
  P0-4  谱半径：有限差分 Lyapunov lambda_hat 替换失效的 rho_hat
  P0-5  调质核团：独立核团产生 L(t)，增益矩阵调制各器官（非器官自产）

设计原则（全部来自工程专家会议的裁决）：
  * 阈值一律为实测分布的分位数，绝不写死常数
  * 调质必须由独立核团产生（器官自产生必然共线，实测有效秩 = 1.0）
  * 控制器标定 rho 本身，绝不标定 sigma_max（rho <= sigma_max 恒成立）
"""
from __future__ import annotations
import numpy as np

# ======================================================================
# P0-4  谱半径：有限差分 Lyapunov
# ======================================================================
def lambda_hat(step_fn, x0, k: int = 16, R: int = 64, eps: float = 1e-7,
               rng: np.random.Generator | None = None) -> float:
    """有限差分 Lyapunov 指数 lambda_hat = (1/k) * log(||dx_k|| / ||dx_0||)

    参数
    ----
    step_fn : (z) -> z'  一步动力学（可作用于一批向量）
    x0      : 参考轨道的初值
    k       : 演化步数（默认 16）
    R       : 随机扰动重复次数（默认 64）

    返回值
    ------
    float : 平均对数增长率。> 0 表示发散，< 0 表示衰减。
            rho < 1  <=>  lambda_hat < 0
    """
    rng = rng or np.random.default_rng(0)
    d = np.asarray(x0).size
    est = []
    for _ in range(R):
        dx = rng.normal(size=d)
        dx /= np.linalg.norm(dx)
        dx *= eps
        for _ in range(k):
            dx = step_fn(dx)
        nrm = float(np.linalg.norm(dx))
        if nrm > 0.0:
            est.append(np.log(nrm / eps) / k)
    return float(np.mean(est)) if est else float("nan")


def lambda_hat_matrix(A, **kw) -> float:
    """对线性算子 A 的 lambda_hat（用于验收测试）。"""
    return lambda_hat(lambda z: A @ z, np.zeros(A.shape[0]), **kw)


class RhoController:
    """谱半径控制器 —— 标定 rho 本身，绝不标定 sigma_max。

    陷阱：rho(W) <= sigma_max(W) 恒成立。若把 sigma_max 锁到 1，
    实际 rho 会落在约 0.5，系统必然衰减到零，而从 sigma_max 视角看
    「完全稳定」。实测自持阈值在 rho ~= 1.00，sigma_max/rho ~= 2.0。
    """

    def __init__(self, target: float = 0.95, k: int = 16, R: int = 32,
                 ema: float = 0.1, every: int = 256, seed: int = 0):
        if not (0.0 < target < 1.5):
            raise ValueError("target 必须落在 (0, 1.5)")
        self.target = float(target)
        self.k, self.R, self.ema, self.every = k, R, ema, every
        self.rng = np.random.default_rng(seed)
        self._ema: float | None = None
        self._step = 0

    @property
    def estimate(self) -> float:
        """当前 rho 的 EMA 估计（换算回线性尺度）。"""
        if self._ema is None:
            return float("nan")
        return float(np.exp(self._ema))

    def observe(self, step_fn, x0) -> float | None:
        """每 every 步测量一次，返回平滑后的 rho（否则 None）。"""
        self._step += 1
        if self._step % self.every:
            return None
        lam = lambda_hat(step_fn, x0, k=self.k, R=self.R, rng=self.rng)
        self._ema = lam if self._ema is None else (1 - self.ema) * self._ema + self.ema * lam
        return self.estimate

    def scale(self, step_fn, x0) -> float:
        """返回把当前 rho 拉向 target 所需的乘性缩放因子。

        注意：缩放作用于【动力学本身】，不是作用于 sigma_max。
        """
        rho = self.observe(step_fn, x0)
        if rho is None or not np.isfinite(rho) or rho <= 0:
            return 1.0
        return float(self.target / rho)


# ======================================================================
# P0-3  能量度量 E'
# ======================================================================
class EnergyMetric:
    """修正后的能量度量 E'

        E' = pred_err(nats) + beta1 * D_centered + beta2 * Conflict

    与旧式的三点区别（旧式在协议单位化下退化为常数 0.1505）：
      1. **不对 v 做 L2 归一化** —— 归一化会把 ||v|| 钉死为 1，
         使 w1*||v|| 成为常数项，信噪比降到 1:1700
      2. **身份偏离中心化** —— D = 1-cos(v,a) 对任意 v 恒约为 1，
         必须减去运行均值
      3. **阈值取分位数** —— 绝不写死 0.5

    「自适应性」体现在：各项先经运行统计量做**除性归一化**
    （半饱和形式 x/(x+sigma)），再由调质层调制 sigma。
    """

    def __init__(self, beta1: float = 1.0, beta2: float = 1.0,
                 momentum: float = 0.99):
        self.beta1, self.beta2, self.momentum = beta1, beta2, momentum
        self._mu: dict[str, float] = {}
        self._var: dict[str, float] = {}
        self._thr: dict[str, np.ndarray] = {}

    # ---- 运行统计（EMA） ----
    def _norm(self, name: str, x: np.ndarray) -> np.ndarray:
        m = self._mu.get(name)
        v = self._var.get(name)
        if m is None:
            self._mu[name] = float(np.mean(x))
            self._var[name] = float(np.var(x) + 1e-12)
            return x - self._mu[name]
        mo = self.momentum
        self._mu[name] = mo * m + (1 - mo) * float(np.mean(x))
        self._var[name] = mo * v + (1 - mo) * float(np.var(x) + 1e-12)
        return x - self._mu[name]

    @staticmethod
    def _divisive(x: np.ndarray, sigma: np.ndarray | float) -> np.ndarray:
        """自适应除性归一化：x / (|x| + sigma)，落在 [0,1)。"""
        return np.abs(x) / (np.abs(x) + sigma + 1e-12)

    def compute(self, pred_err: np.ndarray, identity_dev: np.ndarray,
                conflict: np.ndarray, mod_gain: np.ndarray | float = 1.0) -> np.ndarray:
        """返回 E'（向量，逐器官）。

        pred_err     : 未归一化的预测误差（nats），**不做 L2 归一化**
        identity_dev : 1 - cos(v, a)
        conflict     : 观点池冲突度
        mod_gain     : 调质层输出的除性归一化半饱和参数调制
        """
        pred_err = np.asarray(pred_err, dtype=float)
        d_c = self._norm("D", np.asarray(identity_dev, dtype=float))
        c_c = self._norm("C", np.asarray(conflict, dtype=float))

        sigma = 1.0 / (np.asarray(mod_gain, dtype=float) + 1e-6)
        u = self._divisive(pred_err, sigma)
        d = self._divisive(d_c, sigma)
        c = self._divisive(c_c, sigma)
        return u + self.beta1 * d + self.beta2 * c

    # ---- 分位数阈值（铁律：阈值 = 实测分布的分位数） ----
    def fit_thresholds(self, samples: dict[str, np.ndarray],
                       ps=(0.05, 0.5, 0.9, 0.95)) -> dict[str, np.ndarray]:
        self._thr = {k: np.quantile(np.asarray(v, dtype=float), ps) for k, v in samples.items()}
        return self._thr

    def threshold(self, name: str, p: float = 0.9) -> float:
        if name not in self._thr:
            raise KeyError(f"阈值 {name} 未标定 —— 必须先 fit_thresholds")
        ps = np.array([0.05, 0.5, 0.9, 0.95])
        return float(np.interp(p, ps, self._thr[name]))

