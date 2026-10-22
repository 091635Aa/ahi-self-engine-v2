# -*- coding: utf-8 -*-
"""ahi_theory/closed_world.py —— meeting17-M1 闭环世界（主方案 M1 行动闭环 MVP 的反应式世界域）

预登记：ahi_theory/verify_all.py `_CHANGELOG` #96（2026-10-02『M1 行动闭环 MVP 预登记』，先于改码）。
规格出处：meeting17/00_架构合成会议v2纪要.md 裁决 3（M1=a* 真驱动 world，u_next 回流，E0 逐位基线
+循环反应判据+sham）。五分身两轮 MoE 会签方案的一部分。

设计（零拟合红线）：
  · 全部矩阵为固定随机结构常数（单 rng 种子派生，顺序固定），无任何数据驱动标定；
  · 确定性：同 seed + 同动作序列 ⇒ 状态/观测序列逐位一致（E0 逐位基线的世界侧）；
  · 动作通路双通道：Bu（动作→观测签名，瞬时）+ Bs（动作→状态后果，持续）——
    闭环里 a* 既有"感觉签名"又有"世界后果"，循环反应（可重复自我刺激）在结构上可能；
  · tanh 饱和界：A/C ×0.5 缩放保持非饱和动力学（谱范数≈1）；
  · 离线相期间世界同步冻结（维护R2 内同步执行离线步循环，R1 拍轴无墙钟消耗——
    引擎五模块冻结语义的世界侧对偶）。

接口：
    步(a执行) -> u   # a执行=本拍执行动作（引擎上一拍 a星，1 拍执行延迟——文档化口径）；
                     # u = tanh(C@s + Bu@a)；s ← tanh(A@s + Bs@a)；返回 u
"""
from __future__ import annotations

import numpy as np

__all__ = ["闭环世界", "M1世界种子"]

#: M1 闭环世界预注册主种子（角色/矩阵全单 rng 派生；逐 seed 世界种子=世界种子+engine_seed）
M1世界种子 = 4096


class 闭环世界:
    """确定性反应式世界（M1 闭环臂专用；与 v3 开环世界代理流无关）。"""

    def __init__(self, 输入维度: int = 16, 动作维度: int = 4, 种子: int = M1世界种子):
        rng = np.random.default_rng(种子)      # 单 rng 顺序派生（预注册顺序，不挑）
        self.A = (rng.standard_normal((输入维度, 输入维度)) * 0.5).astype(np.float32)
        self.Bs = rng.standard_normal((输入维度, 动作维度)).astype(np.float32)
        self.C = (rng.standard_normal((输入维度, 输入维度)) * 0.5).astype(np.float32)
        self.Bu = rng.standard_normal((输入维度, 动作维度)).astype(np.float32)
        self.s = np.tanh(rng.standard_normal(输入维度)).astype(np.float32) * 0.5
        self.输入维度 = int(输入维度)
        self.动作维度 = int(动作维度)
        self.拍数 = 0

    def 步(self, a执行) -> np.ndarray:
        """单 R1 拍：u = tanh(C@s + Bu@a)；s ← tanh(A@s + Bs@a)。a执行=零向量或 ±单位轴。"""
        a = np.asarray(a执行, dtype=np.float32).ravel()
        u = np.tanh(self.C @ self.s + self.Bu @ a).astype(np.float32)
        self.s = np.tanh(self.A @ self.s + self.Bs @ a).astype(np.float32)
        self.拍数 += 1
        return u
