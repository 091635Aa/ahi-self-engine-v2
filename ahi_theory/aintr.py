# -*- coding: utf-8 -*-
"""ahi_theory.aintr —— A_int（内感受自主性）的**定稿测度**。

结论链（全部实测，探针 meeting3/Q2–Q7）：
  · E1 原口径（50 阶原始延时嵌入 → q 水平值）：零假设目标 R²≈−4.5（无泄漏）但
    u_only 正对照也给 −3.7 ⇒ **对 τ_q ≥ 200 的慢变量没有判别力**（窗口 ≪ 记忆）。
  · 多尺度窗口 / EMA 基 → q 水平值：**趋势外推伪影**——把特征换成与引擎无关的纯随机流
    仍得 R²=0.55（real）/0.85（null）（Q6 决定性取证）；去冷启动与去 3 阶趋势都救不了。
  · **定稿**：目标改 Δq_t = q_t − q_{t−1}（平稳、短记忆 ⇒ 训练段拟合不能外推到留出段），
    特征 = u_{t..t−7}（192 维），60/40 前后切分，岭回归 α∈{1,10,100,1000} 取最强。
    三重自证（Q7，6 条件 × 3 seed 实测）：
      N1 伪特征（纯随机 u）：R² = −0.016…0.020  ✓ 管道无伪影
      N2 异源目标（独立 u 流驱动的另一台引擎）：R² = −0.093…−0.035  ✓
      N3 正对照 u_only（u 直接灌进 q）：R² = 0.911 / 0.925 / 0.928  ✓ 判别力极强
    规格本体 real：R² = 0.015 / −0.004 / −0.077 ⇒ **A_int = 0.985 / 1.004 / 1.077**。
用法：`a_int_from(U, Q, burn=1100)` → dict(r2, alpha, A_int)；U 为 u 流 (T,24)，Q 为 q 轨迹 (T,8)。
"""
from __future__ import annotations
import numpy as np

LAGS = 8
SPLIT = 0.6
ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def feats_dq(U, lags=LAGS):
    """因果特征：u_{t..t−lags+1}（含 t 时刻，q_t 的更新发生在 u_t 进入 h 之前）。"""
    idx = np.arange(lags - 1, len(U))
    X = np.stack([U[idx - k] for k in range(lags)], 1).reshape(len(idx), -1)
    return X, idx


def r2_hold(X, Y, alphas=ALPHAS, split=SPLIT):
    """单次前后切分 R²（训练段在前、留出段在后）——对短记忆目标无未来泄漏。"""
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Y = Y - Y.mean(0)
    n = len(X)
    c = int(n * split)
    best, ba = -1e9, None
    for a in alphas:
        W = np.linalg.solve(X[:c].T @ X[:c] + a * np.eye(X.shape[1]), X[:c].T @ Y[:c])
        P = X[c:] @ W
        r2 = 1.0 - float(((Y[c:] - P) ** 2).sum()) / max(float((Y[c:] ** 2).sum()), 1e-30)
        if r2 > best:
            best, ba = r2, a
    return best, ba


def _prep(U, Q, burn):
    dQ = np.diff(Q, axis=0)
    X, idx = feats_dq(U)
    keep = idx >= max(burn, 1)
    i2 = idx[keep]
    return X[keep], dQ[i2 - 1]


def a_int_from(U, Q, burn=1100):
    """A_int = 1 − R²_hold(Δq ← u_{t..t−7})；返回自检所需的全部数字。"""
    X, Y = _prep(U, Q, burn)
    r2, a = r2_hold(X, Y)
    return dict(r2=r2, alpha=a, A_int=1.0 - r2, n=len(Y))


def r2_shift_null(U, Q, burn=1100, B=20):
    """零假设分布：把特征在时间上整体循环移位（保留边缘分布、破坏对齐）。返回 (μ, σ)。"""
    X, Y = _prep(U, Q, burn)
    vals = []
    for k in range(1, B + 1):
        sh = int(len(X) * k / (B + 1))
        Xs = np.roll(X, sh, axis=0)
        vals.append(r2_hold(Xs, Y)[0])
    v = np.asarray(vals, float)
    return float(v.mean()), float(v.std() + 1e-12)


def r2_rand_feats(Q, burn=1100, seed=0, d_u=24):
    """N1 自证：用纯随机 u（与引擎无关）做特征。"""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(len(Q), d_u)) * 0.01
    return a_int_from(U, Q, burn)["r2"]