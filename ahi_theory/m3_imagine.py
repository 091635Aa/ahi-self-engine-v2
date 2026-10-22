# -*- coding: utf-8 -*-
"""ahi_theory/m3_imagine.py —— M3 想象引擎（#108 预登记实装；meeting17 裁决 1 + FCA-S S1 对齐）。

影子只读纪律：零引擎改动、零控制流接管——只读消费 hdc 公开/既知状态（IM/W_selfHop/S/M 读/_动作向量/
_cos），episode 与 k̃ 自预测均不写任何引擎状态（J3 判据：W 哈希前后全同）。

机制：
  rollout（裁决 1 骨架）：v₀=IM[k*] → v_{c+1}=sign(W@sign(M@(a_c⊛v_c)))，a_c=G 表 top-2，L=2；
  库外组合：终态 max_k|cos(v_L,IM[k])| < θ_w（0.30 复用=零新参数）；
  k̃ 自预测（底层专家三参与点之读出面）：当前 S 作 C 弛豫 → argmax_k|cos(S̃,IM[k])|；
  近邻外推 sham：v₀ 单步 W 弛豫（无动作绑定）——库外率对照。
"""
from __future__ import annotations

import numpy as np

__all__ = ["想象引擎"]

THETA_W = 0.30          # 复用 hdc_core 预注册 θ_w（零新参数）
C_弛豫 = 5              # 复用 hdc_core C≈5
L_步 = 2                # #108 预注册：rollout 深度


class 想象引擎:
    """M3 影子想象探针（构造时注入 hdc 实例；自身无状态除统计计数）。"""

    def __init__(self, hdc):
        self.hdc = hdc
        self.episode数 = 0
        self.跳过数 = 0                      # IM 空或 k*=-1（无井可锚）

    def _cos表(self, v) -> np.ndarray:
        """v 与全部井的 |cos| 表（无井⇒空数组）。"""
        if not self.hdc.IM:
            return np.zeros(0)
        X = np.stack([it["井"] for it in self.hdc.IM]).astype(np.float64)
        xn = np.linalg.norm(X, axis=1)
        xn[xn == 0.0] = 1.0
        vf = np.asarray(v, dtype=np.float64)
        vn = np.linalg.norm(vf)
        if vn == 0.0:
            return np.zeros(len(X))
        return np.abs((X / xn[:, None]) @ (vf / vn))

    def rollout(self, a星表, v0: np.ndarray):
        """L 步 rollout：v_{c+1}=sign(W@sign(M@(a_c⊛v_c)))；返回（终态, 轨迹）。只读。"""
        v = v0.astype(np.float32)
        轨迹 = [v.copy()]
        for c in range(min(L_步, len(a星表))):
            key = self.hdc._动作向量(a星表[c]) * np.sign(v)      # a_c ⊛ v_c（双极乘）
            Z = self.hdc._M读(key.reshape(-1, 1).astype(np.float32))
            x̂ = np.sign(Z[:, 0])
            v = np.sign(self.hdc.W_selfHop @ x̂).astype(np.float32)
            轨迹.append(v.copy())
        return v, 轨迹

    def episode(self, a星表, k星: int):
        """单次想象 episode。返回 dict(库外=bool, 终态亲和=float, sham库外=bool, k̃=int|None)。
        IM 空或 k*<0 ⇒ 跳过（k̃=None）。只读，不写任何引擎状态。"""
        if not self.hdc.IM or k星 < 0 or k星 >= len(self.hdc.IM):
            self.跳过数 += 1
            return dict(库外=None, 终态亲和=None, sham库外=None, k̃=None, 跳过=True)
        self.episode数 += 1
        v0 = self.hdc.IM[k星]["井"]
        v_L, _轨迹 = self.rollout(a星表, v0)
        # 库外（real rollout）
        sims = self._cos表(v_L)
        库外 = bool(len(sims) == 0 or sims.max() < THETA_W)
        终态亲和 = float(sims.max()) if len(sims) else 0.0
        # 近邻外推 sham：v₀ 单步 W 弛豫（无动作绑定）
        v_sham = np.sign(self.hdc.W_selfHop @ v0).astype(np.float32)
        sims_sham = self._cos表(v_sham)
        sham库外 = bool(len(sims_sham) == 0 or sims_sham.max() < THETA_W)
        # k̃ 自预测（当前 S 弛豫 C 次）
        S̃ = self.hdc.S.copy()
        for _ in range(C_弛豫):
            S̃ = np.sign(self.hdc.W_selfHop @ S̃)
        sims_k = self._cos表(S̃)
        k̃ = int(np.argmax(sims_k)) if len(sims_k) else None
        return dict(库外=库外, 终态亲和=终态亲和, sham库外=sham库外, k̃=k̃, 跳过=False)

    def 自预测预报(self, q, a星, e_last) -> int | None:
        """#109 J2' 预报式 k̃：用维护前可得输入复算 step 7（S 更新）后弛豫 C 次。
        q=维护将用的内感受（u[:8]）、a星=将传入的动作、e_last=上一拍 e。只读。"""
        if not self.hdc.IM:
            return None
        from ahi_theory.hdc_core import GAMMA_S
        qv = np.sign(self.hdc.Φ_q @ np.asarray(q, dtype=np.float32))
        av = self.hdc._动作向量(np.asarray(a星, dtype=np.float32))
        e标 = 1.0 if e_last is None else float(np.clip(e_last, 0.0, 2.0))
        S_raw = (GAMMA_S * self.hdc._ρ(self.hdc.S) + qv * self.hdc.R_q
                 + av * self.hdc.R_act + e标 * self.hdc.R_err)   # Φ_body=0（已注册模块数=0）
        S̃ = np.sign(S_raw)
        for _ in range(C_弛豫):
            S̃ = np.sign(self.hdc.W_selfHop @ S̃)
        sims = self._cos表(S̃)
        return int(np.argmax(sims)) if len(sims) else None
