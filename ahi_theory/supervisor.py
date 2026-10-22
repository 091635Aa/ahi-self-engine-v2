# -*- coding: utf-8 -*-
"""ahi_theory.supervisor —— M3 监督者：R1 检查点托管 + 冷启动预注册 a₀。

★★【M3 实装 · 2026-09-28】本文件为 **纯新增**（预登记于 `ahi_theory/verify_all.py`
  `_CHANGELOG` #63 M3 实装预登记条目）。O7 脊髓态裁决（所有者 2026-09-28，同台账
  O7 条目）三条硬约束的本体落位：
  ① R0 只输出预注册安全默认动作集 —— 本模块**不属于** R0 代码路径；
  ② R1 检查点快照由本模块托管、**仅用于 R1 恢复重启**；
  ③ R0 永不消费失效层输出 —— `three_loop.R0反射环` 类体不 import / 不读取本模块
     （`three_loop` 仅在实时壳 `_建实时体` 与 `恢复动作` 处**惰性** import）；
  冷启动用预注册 a₀（8 维零向量，与 m1 动作维度一致、O7"静止标量"精神一致）。

规格出处：`meeting11/95_架构定稿.md` §5-M3（行 465）、越级抑制六条（行 92–95）、
L1 参数表（行 106–124）；O7 裁决见台账 #62 前置条目。

诚实边界（如实登记，不虚设）：
  · 检查点为**骨架级轻量摘要** {拍数, t_ns, 最后动作, 存档钟ns} —— 动力学级全量
    快照（h/m/g/W 矩阵）属 M11 持久层，规格未给内容，本模块不虚构；
  · 单检查点槽 = **最新覆盖式**（无历史版本链；"仅用于 R1 恢复重启" ⇒ 只需最新）；
  · `恢复重启` 只读不删（幂等可重复调用）；新鲜度/龄期策略由调用方（壳）决定；
  · 本模块不做线程重启执行 —— 进程内线程级重生属 M11 进程模型，此处只交付
    "存检查点 / 恢复重启 / 冷启动 a₀" 三件判定本体（预登记交付物口径）。
"""
from __future__ import annotations

import threading
from typing import Optional, Sequence, Tuple

#: 预注册冷启动动作 a₀（O7 裁决：静止标量；8 维 = m1 动作维度；
#: 变更 = 结构事件 + 理由，登记于 _CHANGELOG）
预注册a0: Tuple[float, ...] = (0.0,) * 8


class 监督者:
    """R1 检查点托管（单槽覆盖式）+ 冷启动 a₀。线程安全（threading.Lock 单锁）。"""

    def __init__(self):
        self._锁 = threading.Lock()
        self._检查点: Optional[dict] = None
        self.存档次数 = 0
        self.恢复查询次数 = 0

    # ---- 存（R1 周期调用；覆盖式单槽）----
    def 存检查点(self, 拍数: int, t_ns: int, 最后动作: Sequence[float],
                存档钟ns: int) -> None:
        """存最新检查点。`最后动作` 立即 tuple 化（防写后改的别名别名问题）。"""
        动作 = tuple(float(x) for x in 最后动作)
        with self._锁:
            self._检查点 = dict(拍数=int(拍数), t_ns=int(t_ns),
                                最后动作=动作, 存档钟ns=int(存档钟ns))
            self.存档次数 += 1

    # ---- 取（仅用于 R1 恢复重启；O7 ②）----
    def 恢复重启(self) -> Optional[dict]:
        """取最新检查点（只读不删，幂等）；无检查点 ⇒ None（调用方走冷启动）。"""
        with self._锁:
            self.恢复查询次数 += 1
            if self._检查点 is None:
                return None
            cp = dict(self._检查点)
            cp["最后动作"] = tuple(cp["最后动作"])   # 深到不可变层
            return cp

    # ---- 冷启动（O7：预注册 a₀，不重放、不虚构）----
    def 冷启动a0(self) -> Tuple[float, ...]:
        return 预注册a0
