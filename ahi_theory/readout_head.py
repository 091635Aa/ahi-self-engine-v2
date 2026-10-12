# -*- coding: utf-8 -*-
"""ahi_theory.readout_head —— #134/#135/#136/#137 引擎预测读出头（float 线性 NLMS）。

★★ 台账链：#134 预登记（C*⁵ 设计）→ #135 特征修订（外族同特征+s16=149/133 维）→
   #136 C*⁵ PASS（J4 12/12+sham 12/12 双集全绿，状态不可入性纯净成立）→ #137 转正（本文件，
   WCET 实测 13.7µs/拍=R1 预算 0.171%）。
★ 角色：引擎自有 float 预测组件（模拟量读出）；HDC 核仍是状态/记忆基底。预测 û_{t+1}=W·f_t，
   NLMS 增量乘加学习——无 lstsq/solve/pinv/inv（C4 合规）；常量全预注册零拟合。
★ 缺省行为：本组件仅在调用方显式构建/调用时参与——不触碰任何既有路径（逐位不变）。
"""
from __future__ import annotations

import numpy as np

#: #134/#135 预注册常量（定值冻结不凑数；变更=账本结构事件）
ETA_R = 0.1          # NLMS 步长（动力学稳定界 η<2；#152 冻结不动）
F_偏置 = 1           # 特征块：偏置
F_窗 = 128           # 特征块：u 窗口 8×16（#135：外族同款特征面）
F_动 = 4             # 特征块：动作向量
F_私 = 16            # 特征块：s16（私有上下文模式；#135 保留）
F_交 = 64            # #152 特征块：交互 I=flatten(a_t⊗u窗[-1])（一阶交互全展开，零参数）
f_维 = F_偏置 + F_窗 + F_动 + F_私         # = 149（#152 前口径；交互开时 213/145）
范数保护 = 1e-6


class 预测算头:
    """引擎自有 float 预测算头（#134/#137 转正）：预测下一拍世界状态 u（float 域）。

    #152 T3‴：交互开=True ⇒ 追加交互块 I=flatten(a_t⊗u窗[-1])（64 维，结构性展开
    零拟合；动作开=False 时同步排除——对照纯净）。缺省 交互开=False 逐位不变。"""

    def __init__(self, 种子: int, u维: int = 16, 私有开: bool = True,
                 动作开: bool = True, 交互开: bool = False):
        d = f_维 - (0 if 私有开 else F_私) - (0 if 动作开 else F_动)
        if 交互开:
            d += F_交 - (0 if 动作开 else F_交)
        self.f维 = d
        self.私有开 = 私有开
        self.动作开 = 动作开
        self.交互开 = 交互开
        self.W = np.zeros((u维, d), dtype=np.float64)

    def 构建特征(self, u窗, a_t, s16_t=None) -> np.ndarray:
        """t 拍末构建特征（#135：u 窗口 8×16 + a_t + s16_t；信息截止 t）。
        #145：动作开=False 时去动作块（T3″ 动作增量对照头）。
        #152：交互开=True 时追加 I=flatten(a_t⊗u窗[-1])（u窗[-1]=最新拍 u）。"""
        u窗 = np.asarray(u窗, np.float64).ravel()
        u_last = np.asarray(u窗, np.float64).reshape(-1, 16)[-1]     # 8×16 展平后取末行
        块 = [np.array([1.0]), u窗]
        if self.动作开:
            a = np.asarray(a_t, np.float64)
            块.append(a)
            if self.交互开:
                块.append((a[:, None] * u_last[None, :]).ravel())    # 4×16=64
        if self.私有开:
            块.append(np.asarray(s16_t, np.float64))
        return np.concatenate(块)

    def 预测(self, f: np.ndarray) -> np.ndarray:
        """û = W·f（信息截止 f 构建拍）。"""
        return self.W @ f

    def 学习(self, f: np.ndarray, u: np.ndarray) -> None:
        """NLMS 增量：W←W+η_r·(u−û)·f^T/‖f‖²（纯乘加）。"""
        f64 = np.asarray(f, np.float64)
        u64 = np.asarray(u, np.float64)
        û = self.W @ f64
        n2 = float(f64 @ f64) + 范数保护
        self.W += (ETA_R / n2) * np.outer(u64 - û, f64)
