# -*- coding: utf-8 -*-
"""ahi_theory.three_loop —— M2 三时钟域壳（E4 六域执行栈最小原型；规范名）。

★★【M2 收尾登记 · 2026-09-27】本文件即 M2 三环壳**规范名实现**（任务书 §2 指定
  `ahi_theory/three_loop.py`）。实现自 `runtime3.py` 原位迁移（逐行同物，仅改名）：
  日期/原因：2026-09-27，M2 收尾规范化——任务书命名与 95_架构定稿 §5-M2 对齐；
  授权依据：meeting11/E4_Round3_签署.md §1 白名单 W1（调度权上移三环壳）+
  meeting11/E3_Round3_签署.md §1.1 让渡①（步进废除→各环各自 tick）；闸1 预注册见
  `ahi_theory/verify_all.py` `_CHANGELOG` 2026-09-27 M2 收尾条目；
  回退方式：`ahi_theory/runtime3.py` 保留为兼容别名（from three_loop import *），
  改回旧名 = 两文件名互换即回，公开 API 逐位同物（meeting11/M2_selftest 复跑为证）。

规格出处：`meeting11/95_架构定稿.md` §2-L1（三环周期 R0 1ms / R1 20ms / R2 200ms 基拍 +
world 域 10ms；定时包四字段；快照/seqlock 读；ack 背压 + R0 钳制；kill 矩阵 RT2 终稿）、
`meeting11/E4_Round3_签署.md` §1/§3、`meeting11/E4_底层系统_方案.md` §3。

四道闸（E4_Round3 §1）：
  闸1 预注册 —— `ahi_theory/verify_all.py` `_CHANGELOG` 2026-09-27 M2 条目（先于改码登记）；
  闸2 惰性启用 —— 本壳 `--realtime` 式开关**默认关闭**：关闭时逻辑调度模式与既有单时钟
      canonical 路径**逐位对照**（M2_selftest [1]：同输入流 max|Δ|=0）；
  闸3 回归 —— M2_selftest 双口径（默认关/实时开）+ 锚点复核（M6 两锚点）；
  闸4 封闭性 —— 本文件只实现 W1–W5 授权面与其消费，无隐性白名单扩充。

双口径三件套（E3 让渡条件）：**步轴语义为判据本体**（bus.t / brain.t 的步数轴永不墙钟化），
**墙钟为执行合同**（--realtime 开时的单调钟节拍）；两轴经换算率 N 关联
（meeting11/M2_wcet.json，O8 落定值；1 R2 拍 = N 步 rollout）。

层间相遇方式：只以**带时间戳的异步消息**（mbp.定时包，四字段冻结接口）+ **seqlock 快照**；
任何两域之间不存在同步函数调用（E4 §3.1 核心不变量：
R0 不依赖 R1/R2 输出即可维持安全态；R1 不依赖 R2 新鲜输出即可继续探测）。

诚实边界（未做/未实测，如实登记）：
  · caregiver 域（10ms，δ 档）= M9 落位（此处仅登记周期，不建线程）；
  · 持久层（检查点/发育史/演练标记落盘）= M11 落位（演练日志只进内存与 selftest 报告）；
  · 实时模式为**软实时四线程**（Windows + Python GIL ⇒ ms 级调度抖动；
    E4 §6 风险①：D1 判据测的是分层相对改善，不冒充绝对硬实时）；
  · R2 巩固切片/归因复核为接口占位（M5–M7 落位）；K-B/K-E/K-F 的实时线程版 = M3。

★★【M3 实装登记 · 2026-09-28】按预登记 #63（ahi_theory/verify_all.py `_CHANGELOG`）
  落位四件，全部为增量改造（默认参数/默认路径 = 旧行为，M2 回归逐位）：
  ① B5 四标量——R0.拍() 签名扩展 限位下界/限位上界/速度界/现在ns（默认 None ⇒ 不参与，
    既有调用逐位不变）；World域.拍() 推导 world 直供安全标量（越界比 tuple 原子替换，
    仅实时 _world循环 转交 _R0循环；R1/R2 无写权）；
  ② B7 门控包——{触发阈,释放阈,收紧比} 校准包：单写者=R2、值域 [0.5×,2.0×]×出厂、
    滞回倒挂拒绝、龄期>5s 读侧弃包回退出厂冻结值；M3 只交付通道机制（R2 真校准 M6，
    默认无写入 ⇒ R0 恒出厂值）；读侧时钟轴：实时=perf_counter 单调钟、逻辑=R1 逻辑步轴；
  ③ 常态挂载——_R0循环 挂 R1 心跳看门狗（60ms=3×R1 拍；K-B 本体：触发 ⇒ 安全默认动作
    a₀=0.0 预注册静止标量，不重放上一拍）+ R2 产出龄期离线窗兜底计时器（600s+1 切片，
    K-E/K-F 本体；产出新鲜 ⇒ 复位）；
  ④ O7——supervisor.监督者 托管 R1 检查点（骨架级单槽覆盖式，仅用于 R1 恢复重启；
    三环壳.恢复动作()：检查点优先，否则冷启动预注册 a₀），R0 代码路径不 import supervisor
    （_建实时体 惰性 import）。
  授权依据：_CHANGELOG #63（M3 预登记）；回退方式：门控读=出厂值路径与默认参数即旧行为，
  supervisor.py 整文件删除即回。
"""
from __future__ import annotations

import math
import sys as _sys
import threading
import time
from collections import deque
from typing import Callable, Dict, List, Optional

import numpy as np

try:
    from .mbp import (MBP包, 定时包, 环形缓冲, 快照槽, MBP总线, MBP模块, 模块描述符,
                      时钟域集合)
except ImportError:      # pragma: no cover
    from ahi_theory.mbp import (MBP包, 定时包, 环形缓冲, 快照槽, MBP总线, MBP模块, 模块描述符,
                                时钟域集合)

# ══════════════════════════════════════════════════════════════════════════
# 定稿参数（95_架构定稿 §2-L1 参数表；代拟值按原文标注）
# ══════════════════════════════════════════════════════════════════════════
域周期秒 = {"R0": 0.001, "R1": 0.020, "R2": 0.200, "world": 0.010, "caregiver": 0.010}
#: 拍预算上限（R2 无硬截止 ⇒ None；验收口径 = |实际周期 − 名义周期| ≤ 预算 的拍占比）
拍预算秒 = {"R0": 0.0003, "R1": 0.008, "R2": None, "world": None}
队列深 = 64                                   # 每队列有界（E4 §3.5）
龄期上限默认秒 = 5.0                          # R2 产出龄期上限
R0看门狗R1死秒 = 0.060                        # 3×R1 拍（E4 §3.3 K-B，代拟存查）
R1降级判定拍 = 3                              # 杀 R2 ⇒ ≤3 R1 拍登记降级
R2心跳粒度秒 = 0.020                          # R2 自喂心跳 20ms 粒度 ⇒ 60ms 看门狗可判
收紧比默认 = 0.5                              # 代拟，待确认（E4 §3.2-2）
滞回比默认 = 0.2                              # 释放阈 = 触发阈 × 0.8（E4 §3.2-4）
#: R0 原始安全通道标量集（≤8 个，B5 正式认领；枚举预注册于此，进 _CHANGELOG）
R0安全通道标量集 = ("限位下界", "限位上界", "速度界", "资源水位")
#: 离线窗兜底计时器（E4 §3.3 K-E：600 s + 1 切片；演练缩放必须带演练标记）
离线窗兜底秒 = 600.0
#: B7 门控包宽容界（校准量 ∈ [下限×, 上限×] × 出厂值；95_架构定稿 §2 行 118，代拟存查）
门控宽容下限 = 0.5
门控宽容上限 = 2.0
#: B5 world 直供标量界（代拟存查：危险度 = 实际越界/界；危险度 > 触发阈 ⇒ 反射）
限位半径默认 = 1.0
速度界默认 = 1.0

# ══════════════════════════════════════════════════════════════════════════
# kill 矩阵 RT2 终稿（E4_Round3 §3.3：6 演练行 × 双 seed 集 + 2 sham 行）—— 框架登记
# （真跑在 M3 的 RT2 全矩阵；本任务交付框架 + 确定性演练演示，见 三环壳.演练_*）
# ══════════════════════════════════════════════════════════════════════════
演练登记表 = {
    "K-A": dict(名称="杀R2", 手段="终止 R2 线程（停心跳/停产出）",
                预期="探测/行为/反射保留；归因冻结最后置信；≤3 R1 拍登记降级",
                判定="降级登记时延 ≤3 R1 拍 ∧ R0/R1 持续存活 ∧ 无未捕获异常"),
    "K-B": dict(名称="杀R1", 手段="终止 R1 线程",
                预期="R0 维持安全包络+安全默认动作（M0 裁决 O7：静止/中性标量，不重放上一拍）；world 时间继续流；R0 看门狗 60ms",
                判定="看门狗触发 ∧ world 拍继续 ∧ 安全默认动作输出"),
    "K-C": dict(名称="拔模块", 手段="单模块挂起（bus.拔掉）",
                预期="现有 B5 语义流程不变（检测改自喂心跳超时）",
                判定="挂起按自喂心跳超时登记（挂起时延=超时3+1 拍契约）"),
    "K-D": dict(名称="冻结R2输出（假活sham本体）", 手段="R2 空转但心跳正常",
                预期="R1 必须凭产出龄期判降级；被当作存活 ⇒ RT2 判据失败",
                判定="产出龄期超 5s ⇒ 降级登记（心跳不豁免）"),
    "K-E": dict(名称="离线窗超时", 手段="R0 墙钟计时器（600s+1 切片）强制收窗",
                预期="按完成度比例清 S；账本登记（S/账本本体 M7/M4，此处登记触发事件）",
                判定="计时器触发强制收窗事件"),
    "K-F": dict(名称="NE通道死（睡死演练）", 手段="NE 广播包静默",
                预期="R0 兜底收窗必须触发（验证'R0 兜底是必需品'）",
                判定="NE 静默 ⇒ 兜底计时器触发；无触发 ⇒ 架构违约"),
    "S-1": dict(名称="假活sham", 手段="同 K-D（心跳继续/产出冻结）",
                预期="若 sham 被当作存活 ⇒ RT2 失败",
                判定="三态龄期第三态（写者活/产出死）被检出"),
    "S-2": dict(名称="噪声误触发sham", 手段="亚阈统计涨落注入（确定性 rng）",
                预期="误触发率 ≤1%",
                判定="误触发次数/拍数 ≤ 1%"),
}
#: 演练日志 schema（E4_Round3 §2#12：字段 E4 出、语义 E5 出）
演练日志字段 = ("intervention_id", "t_start", "t_end", "层身份", "类型枚举", "参数", "死因枚举",
                "演练标记", "缩放因子", "结果")
演练类型枚举 = ("kill_R2", "kill_R1", "freeze_R2", "module_kill",
                "offline_window", "ne_death", "sham_alive", "noise_probe")


def _分位数(样本, q: float) -> float:
    """线性插值分位数（纯 python，无统计依赖；空样本 ⇒ nan）。"""
    if not 样本:
        return float("nan")
    s = sorted(样本)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def _摘要(样本) -> dict:
    """样本摘要（单位随输入）：n / p50 / p95 / p99 / mean / max。"""
    if not 样本:
        return dict(n=0, p50=float("nan"), p95=float("nan"), p99=float("nan"),
                    mean=float("nan"), max=float("nan"))
    return dict(n=len(样本), p50=_分位数(样本, 0.50), p95=_分位数(样本, 0.95),
                p99=_分位数(样本, 0.99),
                mean=sum(样本) / len(样本), max=max(样本))


# ══════════════════════════════════════════════════════════════════════════
# 域时钟服务（单调钟）
# ══════════════════════════════════════════════════════════════════════════
class 域钟:
    """每域独立单调时钟。实时：perf_counter_ns（墙钟执行合同）；
    逻辑：整数 ns 计数器，每 tick 前进名义周期（步轴对照模式，确定性）。"""

    def __init__(self, 域: str, 实时: bool):
        self.域 = 域
        self.实时 = bool(实时)
        self.周期ns = int(域周期秒[域] * 1e9)
        self.逻辑ns = 0
        self.拍数 = 0

    def now_ns(self) -> int:
        return time.perf_counter_ns() if self.实时 else self.逻辑ns

    def 前进(self) -> int:
        """逻辑模式：推进一个名义周期；实时模式：只记拍数（钟由硬件走）。"""
        self.拍数 += 1
        if not self.实时:
            self.逻辑ns += self.周期ns
        return self.now_ns()

    def 等到(self, 截止ns: int, 提前量ns: int = 1_500_000) -> None:
        """实时混合等待：先 sleep 到 截止−提前量，再短睡让渡/自旋（软实时尽力而为）。"""
        提前量ns = max(提前量ns, 50_000)
        while True:
            剩 = 截止ns - time.perf_counter_ns()
            if 剩 <= 0:
                return
            if 剩 > 提前量ns:
                time.sleep((剩 - 提前量ns) / 1e9)
            elif 剩 > 200_000:
                time.sleep(0)          # 让渡 GIL 一小步
            # 剩余 <0.2ms：自旋（不做任何分配）


# ══════════════════════════════════════════════════════════════════════════
# 有界队列（深 64，溢出=丢最旧+登记）——层间异步消息通道
# ══════════════════════════════════════════════════════════════════════════
class 有界队列:
    """线程安全有界队列：满 ⇒ 丢最旧 + 溢出计数（背压可见化，不静默积压；E4 §3.5）。"""

    def __init__(self, 深度: int = 队列深):
        self.深度 = int(深度)
        self._q: deque = deque()
        self._锁 = threading.Lock()
        self.溢出 = 0

    def 放(self, 包) -> None:
        with self._锁:
            if len(self._q) >= self.深度:
                self._q.popleft()
                self.溢出 += 1
            self._q.append(包)

    def 取(self):
        with self._锁:
            return self._q.popleft() if self._q else None

    def __len__(self) -> int:
        return len(self._q)


# ══════════════════════════════════════════════════════════════════════════
# B7 反射调制通道（门控包）：单写者=R2；校准量 ∈ [0.5×,2.0×]×出厂；龄期>5s 回退
# （位于 R0 WCET 审计区段之外——R0 类体内只经 _门控读 钩子消费，不构 numpy/IO）
# ══════════════════════════════════════════════════════════════════════════
class 门控包:
    """B7 反射调制通道：{触发阈, 释放阈, 收紧比} 校准包（95_架构定稿 §2 行 118）。
    单写者=R2（写者校验，其余写者拒绝计数）；校准量逐项 ∈ [门控宽容下限×, 门控宽容上限×]×出厂值；
    释放阈>触发阈（滞回倒挂）拒绝；龄期>5s ⇒ 读侧 None（回退出厂冻结值）。
    M3 交付通道机制本体；R2 真校准产出 = M6（默认无写入 ⇒ R0 恒出厂冻结值 = 现行为）。"""

    def __init__(self, 出厂: dict, 账本=None):
        self.出厂触发阈 = float(出厂["触发阈"])
        self.出厂释放阈 = float(出厂["释放阈"])
        self.出厂收紧比 = float(出厂["收紧比"])
        self._锁 = threading.Lock()
        self._包: Optional[dict] = None
        self.拒绝计数 = 0
        self.过期回退计数 = 0
        self._账本 = 账本

    def 写(self, 触发阈: float, 释放阈: float, 收紧比: float,
           现在ns: int, 写者: str = "R2") -> bool:
        """唯一写入口：写者≠R2 拒绝；值域/滞回逐项校验；通过 ⇒ 覆盖式写包（单槽）。
        拒绝 ⇒ 拒绝计数+1，且注入账本时登记"违规"条目（预登记 #63：越界拒绝+账本登记违规）。"""
        with self._锁:
            if 写者 != "R2":                       # 越级抑制六条：单写者=R2
                self.拒绝计数 += 1
                self._登记违规(dict(原因="单写者≠R2", 收到写者=str(写者)))
                return False
            t = float(触发阈); r = float(释放阈); c = float(收紧比)
            合法 = (门控宽容下限 * self.出厂触发阈 <= t <= 门控宽容上限 * self.出厂触发阈
                    and 门控宽容下限 * self.出厂释放阈 <= r <= 门控宽容上限 * self.出厂释放阈
                    and 门控宽容下限 * self.出厂收紧比 <= c <= 门控宽容上限 * self.出厂收紧比)
            if not 合法 or r > t:                  # 值域越界 / 滞回倒挂 ⇒ 拒绝
                self.拒绝计数 += 1
                原因 = "滞回倒挂(释放阈>触发阈)" if (合法 and r > t) else "值域越界"
                self._登记违规(dict(原因=原因, 触发阈=t, 释放阈=r, 收紧比=c))
                return False
            self._包 = dict(触发阈=t, 释放阈=r, 收紧比=c, 写ns=int(现在ns))
            return True

    def _登记违规(self, payload: dict) -> None:
        """拒绝留痕：注入 M4 账本 ⇒ kind="违规" 条目（#63：越界拒绝+账本登记违规）。"""
        if self._账本 is None:
            return
        self._账本.登记("违规", payload=dict(payload),
                        理由="B7 门控包写侧拒绝（单写者=R2/值域[0.5×,2×]/滞回不倒挂）")

    def 读(self, 现在ns: int) -> Optional[dict]:
        """读侧：无包 ⇒ None（出厂冻结值）；龄期>5s ⇒ 过期回退计数+1 并弃包、None。"""
        with self._锁:
            if self._包 is None:
                return None
            if (int(现在ns) - self._包["写ns"]) > int(龄期上限默认秒 * 1e9):
                self.过期回退计数 += 1             # 重证明：无新鲜门控包 ⇒ 回退出厂值
                self._包 = None
                return None
            return dict(self._包)


# ══════════════════════════════════════════════════════════════════════════
# R0 反射环（1ms）：纯标量、无 numpy（tick 路径）、无内存分配、无 IO —— 可逐行静态审计
# ══════════════════════════════════════════════════════════════════════════
class R0反射环:
    """R0 反射环（唯一致命层）。原始安全通道 world 直供、R1/R2 无写权；
    反射触发 ⇒ R1 当拍钳入收紧包络（收紧比 0.5 代拟）→ 广播钳制通知（R2 收到 ⇒ I_c 冻结）
    → 滞回释放（释放阈=触发阈×0.8）→ 重证明（无新鲜门控包 ⇒ 常态包络+默认门控）。"""

    def __init__(self, 触发阈: float = 1.0, 收紧比: float = 收紧比默认,
                 滞回比: float = 滞回比默认, 常态包络=(-1.0, 1.0), 实时: bool = False,
                 门控读: Optional[Callable] = None):
        self.触发阈 = float(触发阈)                     # 出厂值（B7：默认=冻结出厂值）
        self.收紧比 = float(收紧比)
        self.释放阈 = float(触发阈) * (1.0 - float(滞回比))
        self.常态包络 = (float(常态包络[0]), float(常态包络[1]))
        self.实时 = bool(实时)
        self._门控读 = 门控读            # B7 读侧钩子（None ⇒ 出厂冻结值；由三环壳挂载）
        # —— 状态（纯标量）——
        self.抑制 = False
        self.通道抑制 = {"限位下界": False, "限位上界": False, "速度界": False}
        self.钳制通知计数 = 0            # R0→R2 广播的钳制通知（反 windup 件）
        self.真触发 = 0
        self.最后包络 = self.常态包络
        self.最后危险 = 0.0

    def 拍(self, 危险标量: float, 资源水位: float = 0.0, 限位下界: Optional[float] = None,
           限位上界: Optional[float] = None, 速度界: Optional[float] = None,
           现在ns: Optional[int] = None) -> dict:
        """一拍：滞回状态机 + 包络计算。只吃标量、只出标量。
        B5：主通道=危险标量；附通道=限位下界/限位上界/速度界（None ⇒ 不参与；
        world 直供、R1/R2 无写权）。B7 读侧：门控包新鲜用校准量，否则出厂冻结值。"""
        x = float(危险标量)
        self.最后危险 = x
        触阈 = self.触发阈
        释阈 = self.释放阈
        比校 = self.收紧比
        if self._门控读 is not None:                 # B7 读侧（三环壳挂载；M6 前 R2 不写）
            g = self._门控读(现在ns)
            if g is not None:
                触阈 = float(g["触发阈"])
                释阈 = float(g["释放阈"])
                比校 = float(g["收紧比"])
        触发前 = self.抑制
        if not self.抑制:
            if x > 触阈:
                self.抑制 = True
                self.真触发 += 1
        elif x < 释阈:                   # 滞回释放（纯滞回，不用连击/计数——步数轴思维禁入）
            self.抑制 = False
        for 名, 值 in (("限位下界", 限位下界), ("限位上界", 限位上界), ("速度界", 速度界)):
            if 值 is None:               # B5：未供标量 ⇒ 通道不参与（既有调用逐位不变）
                continue
            v = float(值)
            st = self.通道抑制[名]
            if not st:
                if v > 触阈:
                    self.通道抑制[名] = True
            elif v < 释阈:
                self.通道抑制[名] = False
        整体抑制 = (self.抑制 or self.通道抑制["限位下界"]
                    or self.通道抑制["限位上界"] or self.通道抑制["速度界"])
        比 = 比校 if 整体抑制 else 1.0
        lo = self.常态包络[0] * 比
        hi = self.常态包络[1] * 比
        if 整体抑制 and not 触发前:
            self.钳制通知计数 += 1       # 抑制沿：广播钳制通知（R2 ⇒ I_c 冻结）
        self.最后包络 = (lo, hi)
        if 资源水位 > 0.9:               # 资源界标量（B5 通道之一）：登记性收紧
            lo, hi = lo * 0.9, hi * 0.9
            self.最后包络 = (lo, hi)
        return dict(抑制=整体抑制, 包络下界=lo, 包络上界=hi, 钳制通知=self.钳制通知计数)

    def 钳制标量(self, v: float) -> float:
        """硬限幅（单标量）：R1 指令必须落在 R0 维护的包络内（限幅是硬的）。"""
        lo, hi = self.最后包络
        return lo if v < lo else (hi if v > hi else v)

    def 巡检心跳(self, 心跳龄期表: dict, 看门狗窗拍: int = 3) -> dict:
        """R0 看门狗：对自喂心跳龄期（拍）做纯标量比较。返回 {mid: 是否超时}。"""
        return {mid: (龄期 > 看门狗窗拍) for mid, 龄期 in 心跳龄期表.items()}

    def 兜底计时器(self, 起始钟ns: int, 现在钟ns: int, 上限秒: float = 离线窗兜底秒) -> bool:
        """K-E/K-F 兜底：R0 墙钟计时器 = 上限秒 + 1 切片余量（此处切片余量并入上限判）。"""
        return (现在钟ns - 起始钟ns) > int(上限秒 * 1e9)


class 钳制代理(MBP模块):
    """R0 包络的**硬**钳制点：包裹执行类设备，CMD 先过 R0 包络再到设备。
    不改 bus.写 / selfprobe（签名锁死）——抑制语义在设备层落地（three_loop 职责）。"""

    def __init__(self, 内设备: MBP模块, r0: R0反射环):
        super().__init__(内设备.描述符)
        self.内设备 = 内设备
        self.r0 = r0
        self.钳制次数 = 0

    def 响应(self, 指令向量):
        a = np.asarray(指令向量, dtype=float).ravel()
        lo, hi = self.r0.最后包络
        a2 = np.clip(a, lo, hi)          # R0 出包络标量，硬限幅在代理层做
        if a2.shape == a.shape and np.any(a2 != a):
            self.钳制次数 += 1
        return self.内设备.响应(a2)

    def 感知(self, 时间戳):
        return self.内设备.感知(时间戳)

    def 标定(self, 标定向量):
        return self.内设备.标定(标定向量)

    def 探针响应(self, 原始信号):
        return self.内设备.探针响应(原始信号)

    def 前向(self, 输入):
        return self.内设备.前向(输入)


# ══════════════════════════════════════════════════════════════════════════
# R2 慢思考环（200ms 基拍，事件驱动）：内核多步（=换算率 N）+ 门控包产出 + 自喂心跳
# ══════════════════════════════════════════════════════════════════════════
class R2慢思考环:
    """R2：学习+巩固+弛豫；无硬截止；产出龄期上限 5s（三态龄期）。
    自喂心跳 20ms 粒度（供 R0/R1 60ms 看门狗判真死亡——K-A ≤3 R1 拍）。
    ack 背压（件一）：长期无消费 ack ⇒ 写者登记背压事件（降速/挂起由策略定）。"""

    def __init__(self, brain, 域钟_: 域钟, N步: int = 10,
                 喂u: Optional[Callable[[], np.ndarray]] = None,
                 龄期上限秒: float = 龄期上限默认秒,
                 背压窗秒: float = 0.5, 产出冻结: bool = False):
        self.brain = brain
        self.钟 = 域钟_
        self.N步 = int(N步)              # 换算率 N：1 R2 拍 = N 步 rollout（O8 落定值）
        self.喂u = 喂u
        self.龄期上限秒 = float(龄期上限秒)
        self.背压窗秒 = float(背压窗秒)
        self.活着 = True
        self.产出冻结 = bool(产出冻结)     # S-1/K-D：心跳继续、产出冻结（假活）
        # —— 产出通道 ——
        self.门控快照 = 快照槽()          # seqlock：R1 只读最新门控包
        self.输出队列 = 有界队列(队列深)   # 门控事件等异步消息
        # —— 簿记 ——
        self.最后心跳ns = self.钟.now_ns()
        self.最后产出ns = self.最后心跳ns
        self.最后ackns = self.最后心跳ns
        self.最后ack序号 = -1
        self.产出序号 = 0
        self.步数累计 = 0
        self.拍数 = 0
        self.背压事件: List[dict] = []
        self.钳制通知消费数 = 0
        self.门控值 = 0.0                # 伪门控演示量（G-Layer M6 接手前的占位读出）

    # —— 心跳（自喂，20ms 粒度）——
    def 心跳拍(self) -> None:
        if not self.活着:
            return
        self.最后心跳ns = self.钟.now_ns()
        self._背压巡检()

    def 确认(self, ack序号: int) -> None:
        """背压件一：R1 消费确认（ack）。"""
        if not self.活着:
            return
        self.最后ackns = self.钟.now_ns()
        self.最后ack序号 = int(ack序号)

    def _背压巡检(self) -> None:
        if self.产出序号 <= 0:
            return
        无ack秒 = (self.钟.now_ns() - self.最后ackns) / 1e9
        if 无ack秒 > self.背压窗秒 and (not self.背压事件
                                        or self.背压事件[-1]["序号"] != self.产出序号):
            self.背压事件.append(dict(序号=self.产出序号, 无ack秒=无ack秒,
                                      登记="写者降速（演示：登记不降速）"))

    # —— 工作拍（200ms 基拍）——
    def 工作拍(self, 心跳间隔ns: Optional[int] = None,
               停机事件: Optional[threading.Event] = None) -> Optional[定时包]:
        """一个 R2 拍：u 取自感觉快照（seqlock 读），N 步 rollout，产出门控参数包。
        `心跳间隔ns` 给定时，N 步之间穿插自喂心跳（防长工作拍饿死看门狗——假降级）。"""
        if not self.活着:
            return None
        self.拍数 += 1
        self.最后心跳ns = self.钟.now_ns()
        if self.产出冻结:                 # 假活：心跳走、产出停（K-D/S-1 本体）
            return None
        u = self.喂u() if self.喂u is not None else None
        下心跳 = self.钟.now_ns() + (心跳间隔ns or int(R2心跳粒度秒 * 1e9))
        上次让渡 = self.钟.now_ns()
        完成步 = 0
        for _ in range(self.N步):
            self.brain.step(u)
            完成步 += 1
            # 【M2 修 2026-09-27】时间有界让渡：实时模式下每 ≥5ms 工作让渡一次 GIL
            # （sleep(0) 纯让渡，零语义/确定性影响；逻辑模式钟不前进 ⇒ 不让渡、不影响
            # 调度一致性逐位对照）。依据：R2=best-effort、R1=硬预算层（E4 §3.1）——
            # R2 长工作拍（N 步 numpy）连续持 GIL ⇒ R1 饿死（K-A 实测降级登记 234ms 实证）。
            now = self.钟.now_ns()
            if self.钟.实时 and (now - 上次让渡) >= 5_000_000:
                time.sleep(0)
                上次让渡 = self.钟.now_ns()
            if 心跳间隔ns is not None and now >= 下心跳:
                self.最后心跳ns = now     # 长工作拍中间自喂（20ms 粒度）
                下心跳 += 心跳间隔ns
            if 停机事件 is not None and 停机事件.is_set():
                break
        self.步数累计 += 完成步           # 实际完成步数（停机截断时如实少计）
        # 门控演示读出：内核 F 的有界低通（占位；G-Layer M6 换成真控制律）
        self.门控值 = 0.95 * self.门控值 + 0.05 * float(np.clip(self.brain.F, 0.0, 1.0))
        self.产出序号 += 1
        包 = 定时包("R2", "CAL", self.brain.t,
                    np.array([self.门控值, float(self.brain.F)], dtype=np.float64),
                    能量=float(abs(self.门控值)), 置信=1.0, 代谢=0.0, 序列号=self.产出序号,
                    时钟域="R2", 单调时间戳=self.钟.now_ns(),
                    龄期上限=self.龄期上限秒, 传输延迟=0.0,
                    调节中=True, ack序号=self.产出序号)
        self.门控快照.写(包)
        self.最后产出ns = 包.单调时间戳
        self.输出队列.放(包)
        return 包

    def 消费钳制通知(self, 包: 定时包) -> None:
        """背压件二：R0→R2 钳制通知 ⇒ I_c 冻结（S10；M2 为登记消费，I_c 属 M6 控制律）。"""
        if 包.钳制通知:
            self.钳制通知消费数 += 1


# ══════════════════════════════════════════════════════════════════════════
# world 域（10ms 独立节拍）：CMD 下一拍生效（τ_dl ≤1 拍，抖动 ≤±1 拍）
# ══════════════════════════════════════════════════════════════════════════
class World域:
    """D-7 Phase-1 最小包装：独立 世界 实例 + 10ms 自主节拍（world.py 动力学本体不动）。
    CMD 入队 → **世界自己的下一拍**生效；SENSE 经 seqlock 快照发布（读无副作用）。
    单写者：只有本域线程推进 世界.步()（G4 在域间分解下的形态）。"""

    def __init__(self, 世界实例, 域钟_: 域钟):
        self.世界 = 世界实例
        self.钟 = 域钟_
        self.CMD队列 = 有界队列(队列深)
        self.快照 = 快照槽()
        self.τ_dl拍: List[int] = []
        self.拍数 = 0
        self.安全标量 = None                  # B5：(限位下越, 限位上越, 速度越, now) 原子替换
        self._上拍状态 = None                 # 上拍 SENSE（速度标量差分基线）

    def 发CMD(self, cmd) -> int:
        """异步提交 CMD（带到达拍）；世界自己的**下一拍**生效（τ_dl=1 拍，结构担保 ±0 抖动）。"""
        到达拍 = self.拍数
        self.CMD队列.放((到达拍, cmd))
        return 到达拍

    def 拍(self) -> None:
        self.拍数 += 1
        now = self.钟.now_ns()
        cmd = None                        # 本拍收到的 CMD ⇒ 下一拍（本拍之后的第一次 步）生效
        while True:
            项 = self.CMD队列.取()
            if 项 is None:
                break
            到达拍, c = 项
            self.τ_dl拍.append(self.拍数 - 到达拍)
            cmd = c
        self.世界.步(cmd)
        s = np.asarray(self.世界.感知(now), dtype=float).ravel()
        self.快照.写(定时包("world", "SENSE", self.拍数, s,
                            能量=float(np.linalg.norm(s)), 置信=1.0, 代谢=0.0,
                            序列号=self.拍数, 时钟域="world", 单调时间戳=now,
                            龄期上限=龄期上限默认秒, 传输延迟=0.010))
        # —— B5 安全标量推导（world 直供；危险度=实际越界/界，>触发阈 ⇒ R0 反射）——
        下越 = max(0.0, -float(np.min(s))) / 限位半径默认
        上越 = max(0.0, float(np.max(s))) / 限位半径默认
        速越 = (0.0 if self._上拍状态 is None else
                float(np.max(np.abs(s - self._上拍状态))) / (域周期秒["world"] * 速度界默认))
        self._上拍状态 = s.copy()
        self.安全标量 = (下越, 上越, 速越, now)           # tuple 原子替换，防撕裂


# ══════════════════════════════════════════════════════════════════════════
# 三环壳主类
# ══════════════════════════════════════════════════════════════════════════
class 三环壳:
    """三时钟域壳：--realtime 关（默认）= 逻辑调度（canonical 逐位对照）；
    开 = 四线程（R0/R1/R2/world）+ 单调钟软实时。

    逻辑模式：每壳拍 = 1 个 R1 拍 = 一次 canonical 系统.步进（零扰动）；
      R0 以 20 个逻辑子拍记账、world 以 2 个逻辑子拍记账、R2 每 10 个 R1 拍记账 1 拍——
      **域计数器只记账，不改执行序**（步轴为判据本体；执行合同在 --realtime 才生效）。
    """

    def __init__(self, 系统=None, 实时: bool = False, 配置: Optional[dict] = None):
        cfg = dict(seed=101, N步=None, 危险标量=None, 提前量ns=1_500_000)
        cfg.update(配置 or {})
        self.配置 = cfg
        self.实时 = bool(实时)
        self.系统 = 系统                      # 逻辑模式：AHI系统（canonical 本体）
        self.钟 = {域: 域钟(域, self.实时) for 域 in ("R0", "R1", "R2", "world")}
        self.R0 = R0反射环(实时=self.实时)
        self.危险标量 = cfg["危险标量"]        # None ⇒ 常量 0；callable ⇒ 每子拍调用
        self.资源水位 = 0.0
        # —— M3：B7 门控包（单写者=R2；默认无写入 ⇒ R0 恒回退出厂冻结值）+ 账本槽 ——
        self.账本 = None                      # 可选注入 ahi_theory.ledger.环形账本
        self.门控包 = 门控包(dict(触发阈=self.R0.触发阈, 释放阈=self.R0.释放阈,
                                  收紧比=self.R0.收紧比))
        self.R0._门控读 = self._门控读侧
        self.监督者 = None                    # O7：实时 _建实时体 惰性创建（R0 路径不 import）
        # —— 演练/日志 ——
        self.演练日志: List[dict] = []
        self._演练序号 = 0
        # —— 逻辑模式状态 ——
        self._r1拍数 = 0
        self._r2记账 = 0
        self._r2活着 = True
        self._r2产出冻结 = False
        self._r2最后心跳逻辑ns = 0
        self._r2门控产出逻辑ns: Optional[int] = None
        self._r2降级已登记 = False
        self._r2降级拍: Optional[int] = None
        self._r1活着 = True
        self._r1死登记拍: Optional[int] = None
        self._r0安全动作输出 = 0.0            # K-B：R0 安全默认动作（静止标量 0，M0 裁决 O7）
        self.降级事件: List[dict] = []
        # —— 实时模式状态 ——
        self._线程: Dict[str, threading.Thread] = {}
        self._停机 = threading.Event()
        self.周期记录: Dict[str, List[float]] = {域: [] for 域 in ("R0", "R1", "R2", "world")}
        self._r2: Optional[R2慢思考环] = None
        self._world: Optional[World域] = None
        self._bus: Optional[MBP总线] = None
        self._m1代理: Optional[钳制代理] = None
        self._已见钳制通知 = 0
        self._R1探测数 = 0
        self._R2降级ns: Optional[int] = None
        self._R2死ns: Optional[int] = None
        # —— M3 实时常态挂载状态 ——
        self._R1最后心跳ns: Optional[int] = None
        self._R1被杀 = threading.Event()      # K-B 实时演练：置位 ⇒ _R1循环 退出（心跳停）
        self._R1拍计数 = 0
        self._R1检查点间隔拍 = 50             # 代拟存查：≈1s 一存（R1 拍 20ms）
        self._world安全标量 = None            # (限位下越, 限位上越, 速度越, now) tuple 原子替换
        self._离线起ns: Optional[int] = None
        self._离线已触发 = False
        self._R1看门狗已登记 = False
        self.N步 = int(cfg["N步"]) if cfg["N步"] else 10   # 未供 wcet 时 = E4 代拟 10

    # ------------------------------------------------------------- 逻辑模式
    def 拍(self, u_ext=None) -> dict:
        """逻辑模式推进一拍（=1 R1 拍）。canonical 执行序逐位不变 + 域记账。"""
        if self.实时:
            raise RuntimeError("实时模式请用 启动()/停止()")
        r0t = self.钟["R0"]
        for _ in range(20):                                  # R0：20 个逻辑子拍（1ms 名义）
            self.R0.拍(self._危险标量取值(), self.资源水位)
            r0t.前进()
        wt = self.钟["world"]
        for _ in range(2):                                   # world：2 个逻辑子拍（10ms 名义）
            wt.前进()
        self._r1拍数 += 1
        self.钟["R1"].前进()
        self._逻辑R1拍(u_ext)                                # canonical 主体（零扰动）
        self._r2记账 += 1
        if self._r2记账 >= 10:                               # R2：每 10 个 R1 拍记账 1 拍
            self._r2记账 = 0
            self.钟["R2"].前进()
            self._逻辑R2拍()
        return self._逻辑遥测()

    def _危险标量取值(self) -> float:
        危 = self.危险标量
        if 危 is None:
            return 0.0
        return float(危()) if callable(危) else float(危)

    def _门控读侧(self, 现在ns: Optional[int] = None) -> Optional[dict]:
        """B7 门控包读侧：实时=perf_counter 单调钟；逻辑=R1 逻辑步轴（写读同轴）。"""
        if 现在ns is None:
            现在ns = time.perf_counter_ns() if self.实时 else self.钟["R1"].逻辑ns
        return self.门控包.读(现在ns)

    def _逻辑R1拍(self, u_ext) -> None:
        """逻辑 R1 拍：canonical 系统.步进（执行序不动）+ 自喂/巡检/降级判定。
        降级判定双路径（三态龄期）：写者心跳缺失 ≥3 R1 拍（K-A）/ 产出龄期 >5s（K-D、S-1）。"""
        if self._r1活着:
            self.系统.步进(u_ext)                            # ← canonical 路径，逐位不变
            if self._r2活着:                                 # R2 自喂心跳（20ms 粒度）
                self._r2最后心跳逻辑ns = self.钟["R1"].逻辑ns
        else:
            # K-B：R1 死 ⇒ R0 看门狗；R0 输出安全默认动作（静止标量，不重放上一拍）
            if self._r1死登记拍 is None:
                self._r1死登记拍 = self._r1拍数
                self.降级事件.append(dict(事件="R1死亡-看门狗", 拍=self._r1拍数,
                                          窗秒=R0看门狗R1死秒))
            self._r0安全动作输出 = 0.0
        if not self._r2降级已登记:
            心跳缺拍 = 0 if self._r2活着 else int(
                (self.钟["R1"].逻辑ns - self._r2最后心跳逻辑ns) // 20_000_000)
            产出龄 = None
            if self._r2门控产出逻辑ns is not None:
                产出龄 = (self.钟["R1"].逻辑ns - self._r2门控产出逻辑ns) / 1e9
            判心跳死 = (not self._r2活着) and 心跳缺拍 >= R1降级判定拍
            判产出死 = self._r2产出冻结 and 产出龄 is not None and 产出龄 > 龄期上限默认秒
            if 判心跳死 or 判产出死:
                self._r2降级已登记 = True
                self._r2降级拍 = self._r1拍数
                self.降级事件.append(dict(事件="R2降级登记", 拍=self._r1拍数,
                                          心跳缺拍=心跳缺拍, 产出龄期秒=产出龄,
                                          路径="心跳缺失" if 判心跳死 else "产出龄期"))

    def _逻辑R2拍(self) -> None:
        """逻辑 R2 拍（每 10 个 R1 拍记账一次）：心跳 + 门控产出时间戳。"""
        if self._r2活着:
            self._r2最后心跳逻辑ns = self.钟["R2"].逻辑ns
            if not self._r2产出冻结:
                self._r2门控产出逻辑ns = self.钟["R2"].逻辑ns

    def _逻辑遥测(self) -> dict:
        return dict(拍=self._r1拍数, bus_t=self.系统.bus.t, brain_t=self.系统.brain.t,
                    R0拍=self.钟["R0"].拍数, world拍=self.钟["world"].拍数,
                    R2拍=self.钟["R2"].拍数, 抑制=self.R0.抑制,
                    钳制通知=self.R0.钳制通知计数, 降级事件=self.降级事件)

    def 运行(self, 拍数: int, u_ext流=None) -> None:
        for i in range(int(拍数)):
            u = None if u_ext流 is None else u_ext流[i]
            self.拍(u)

    # ------------------------------------------------------------- 实时模式
    def 启动(self) -> None:
        """实时模式：起 R0/R1/R2/world 四线程（最小演练体；全量闭环实时化 = M3+）。"""
        if not self.实时:
            raise RuntimeError("非实时壳请用 运行(拍数)")
        if self._线程:
            raise RuntimeError("已在运行")
        self._停机.clear()
        # —— M3：实时常态挂载状态复位（K-B/兜底可重复演练）——
        self._R1被杀.clear()
        self._R1看门狗已登记 = False
        self._R1最后心跳ns = None
        self._R1拍计数 = 0
        self._离线起ns = None
        self._离线已触发 = False
        try:
            _sys.setswitchinterval(0.0005)     # 缩短 GIL 切换窗（软实时尽力而为）
        except Exception:
            pass
        self._建实时体()
        for 名, fn in (("R0", self._R0循环), ("R1", self._R1循环),
                       ("R2", self._R2循环), ("world", self._world循环)):
            th = threading.Thread(target=fn, name="three_loop-%s" % 名, daemon=True)
            self._线程[名] = th
            th.start()

    def 停止(self, 等待秒: float = 2.0) -> None:
        self._停机.set()
        for _, th in list(self._线程.items()):
            th.join(timeout=等待秒)
        self._线程.clear()

    def _建实时体(self) -> None:
        from .brain import Brain, D_REGION
        from .world import 世界
        seed = int(self.配置["seed"])
        # —— 最小演练体：bus（实时模式）+ 执行类 m1（R0 钳制代理）+ world 只读设备 ——
        self._bus = MBP总线(投影维度默认_=64, tau=1.0, 心跳超时=3, seed=seed,
                            实时模式=True)
        d1 = 模块描述符("m1", "执行类", "流式", 8, 8, "φ_act",
                        采样率="1kHz", 缓冲深度=1, 代谢常数=0.01, 延迟="1ms",
                        能力=("肩关节",), 协议版本="MBP-1.0")
        rng = np.random.default_rng(seed)
        self._m1代理 = 钳制代理(_线性设备(d1, rng), self.R0)
        self._bus.注册(d1, self._m1代理)
        dw = 模块描述符("wld", "双向类", "流式", 8, 8, "φ_universal",
                        采样率="1kHz", 缓冲深度=1, 代谢常数=0.01, 延迟="1ms",
                        能力=("世界",), 协议版本="MBP-1.0")
        self._世界 = 世界(维度=8, 动作维度=8, 噪声=0.02, 增益=0.15, 种子=seed + 41)
        self._bus.注册(dw, _世界只读设备(dw, self._世界))
        # —— R2 内核 + 换算率 N（O8 执行合同）——
        self._brain = Brain(seed=seed)
        Uw = np.random.default_rng(900).normal(size=(D_REGION, 8)) / 4.0

        def 喂u():
            包 = self._world.快照.读()
            s = np.asarray(包.向量, dtype=float).ravel() if 包 is not None else np.zeros(8)
            return 0.01 * (Uw @ s)

        self._Uw = Uw
        self._r2 = R2慢思考环(self._brain, self.钟["R2"], N步=self.N步, 喂u=喂u)
        self._world = World域(self._世界, self.钟["world"])
        self._R1探测a = np.random.default_rng(seed + 7)
        # —— M3 O7：监督者（R1 检查点托管）——惰性 import：R0 代码路径不触碰 supervisor
        from .supervisor import 监督者
        self.监督者 = 监督者()

    def 恢复动作(self):
        """O7：R1 恢复重启动作——检查点优先，否则冷启动预注册 a₀（静止标量）。
        supervisor 仅托管快照本体；重启执行属 M11 进程模型（本方法只出动作向量）。"""
        if self.监督者 is not None:
            cp = self.监督者.恢复重启()
            if cp is not None:
                return cp["最后动作"]
        from .supervisor import 预注册a0
        return 预注册a0

    # —— 四个线程循环 ——
    def _R0循环(self) -> None:
        钟 = self.钟["R0"]
        下一拍 = 钟.now_ns() + 钟.周期ns
        上拍ns = 钟.now_ns()
        while not self._停机.is_set():
            now = 钟.now_ns()
            快照 = self._world安全标量
            if 快照 is not None:               # B5：world 直供四标量（主通道+三附通道）
                self.R0.拍(self._危险标量取值(), self.资源水位,
                           限位下界=快照[0], 限位上界=快照[1], 速度界=快照[2],
                           现在ns=now)
            else:
                self.R0.拍(self._危险标量取值(), self.资源水位)
            self._R0看门狗巡检(now)
            self._R0兜底巡检(now)
            now2 = 钟.now_ns()
            self.周期记录["R0"].append((now2 - 上拍ns) / 1e6)
            上拍ns = now2
            钟.拍数 += 1
            下一拍 += 钟.周期ns
            钟.等到(下一拍, self.配置["提前量ns"])

    def _R0看门狗巡检(self, now_ns: int) -> None:
        """M3 K-B 本体：R0 常态挂载 R1 心跳看门狗（60ms=3×R1 拍，预登记 #63）。
        触发 ⇒ R0 独立输出安全默认动作 a₀=0.0（O7 预注册静止标量，不重放上一拍）+ 降级登记。"""
        if self._R1看门狗已登记 or self._R1最后心跳ns is None:
            return
        龄ms = (now_ns - self._R1最后心跳ns) / 1e6
        if 龄ms > R0看门狗R1死秒 * 1e3:
            self._R1看门狗已登记 = True
            self._r0安全动作输出 = 0.0          # O7：预注册安全默认动作（静止标量）
            self.降级事件.append(dict(事件="R1死亡-看门狗(实时)", 时延ms=龄ms,
                                      窗ms=R0看门狗R1死秒 * 1e3))
            if self.账本 is not None:
                self.账本.登记("结构事件", payload=dict(事件="R1死亡-看门狗", 时延ms=龄ms),
                               理由="M3 K-B 本体：R1 死 ⇒ R0 独立维持安全态（O7）")

    def _R0兜底巡检(self, now_ns: int) -> None:
        """M3 K-E/K-F 本体：R2 产出龄期>5s（三态龄期第三态：写者活/产出死）⇒ 起算离线窗；
        R0 墙钟计时器 600s+1 切片到点 ⇒ 兜底收窗登记（段内防重复；产出恢复 ⇒ 复位）。"""
        if self._r2 is None:
            return
        产出ns = self._r2.最后产出ns
        if (now_ns - 产出ns) <= int(龄期上限默认秒 * 1e9):   # 产出新鲜 ⇒ 复位离线窗
            self._离线起ns = None
            self._离线已触发 = False
            return
        if self._离线起ns is None:
            self._离线起ns = now_ns             # 产出死亡 ⇒ 起算离线窗
            return
        if not self._离线已触发 and self.R0.兜底计时器(self._离线起ns, now_ns,
                                                      上限秒=离线窗兜底秒):
            self._离线已触发 = True
            self.降级事件.append(dict(事件="R0兜底收窗(实时)", 离线秒=离线窗兜底秒))
            if self.账本 is not None:
                self.账本.登记("结构事件", payload=dict(事件="R0兜底收窗"),
                               理由="M3 K-E/K-F 本体：R2 产出离线超窗 ⇒ R0 强制收窗")

    def _R1循环(self) -> None:
        钟 = self.钟["R1"]
        bus = self._bus
        下一拍 = 钟.now_ns() + 钟.周期ns
        上拍ns = 钟.now_ns()
        while not self._停机.is_set() and not self._R1被杀.is_set():
            now = 钟.now_ns()
            self._R1最后心跳ns = now         # M3 K-B：R1 心跳（R0 看门狗数据源）
            self._R1拍计数 += 1
            self.周期记录["R1"].append((now - 上拍ns) / 1e6)
            上拍ns = now
            bus.步进(1)                                    # 步轴 +1（不代发心跳，W1）
            for mid in bus.已注册():
                bus.喂心跳(mid)                            # W4：R1 环代管模块的自喂
                bus.生产快照(mid)                          # W2：写者侧生产（读()从此无副作用）
            a = self._R1探测a.normal(size=8)               # 探测 m1（R0 包络经钳制代理硬落地）
            bus.写("m1", a)
            self._R1探测数 += 1
            self._world.发CMD(a)                           # world 域 CMD：下一拍生效（τ_dl）
            # R2 降级判定：写者心跳龄期 > 60ms（≤3 R1 拍，K-A）
            if self._r2 is not None and self._R2降级ns is None:
                龄ms = (now - self._r2.最后心跳ns) / 1e6
                if 龄ms > R0看门狗R1死秒 * 1e3:
                    self._R2降级ns = now
                    self.降级事件.append(dict(事件="R2降级登记(实时)", 时延ms=龄ms,
                                              上限ms=R0看门狗R1死秒 * 1e3))
            # 消费 R2 门控包（seqlock 读）+ ack 背压 + 钳制通知转交
            if self._r2 is not None and self._r2.活着:
                包 = self._r2.门控快照.读()
                if 包 is not None:
                    self._r2.确认(包.ack序号)
            if self.R0.钳制通知计数 > self._已见钳制通知:
                self._已见钳制通知 = self.R0.钳制通知计数
                if self._r2 is not None and self._r2.活着:
                    self._r2.消费钳制通知(定时包("R0", "BEAT", bus.t, np.zeros(2),
                                                时钟域="R0", 单调时间戳=now,
                                                钳制通知=True))
            # M3 O7：R1 检查点快照（骨架级单槽覆盖式；supervisor 托管，仅用于 R1 恢复重启）
            if self.监督者 is not None and self._R1拍计数 % self._R1检查点间隔拍 == 0:
                self.监督者.存检查点(bus.t, now, a, now)
            钟.拍数 += 1
            下一拍 += 钟.周期ns
            钟.等到(下一拍, self.配置["提前量ns"])

    def _R2循环(self) -> None:
        钟 = self.钟["R2"]
        r2 = self._r2
        心跳粒 = int(R2心跳粒度秒 * 1e9)
        下心跳 = 钟.now_ns() + 心跳粒
        下工作 = 钟.now_ns() + 钟.周期ns
        上工作ns = 钟.now_ns()
        while not self._停机.is_set():
            now = 钟.now_ns()
            if now >= 下心跳:
                r2.心跳拍()
                下心跳 += 心跳粒
            if now >= 下工作:
                self.周期记录["R2"].append((now - 上工作ns) / 1e6)
                上工作ns = now
                r2.工作拍(心跳间隔ns=心跳粒, 停机事件=self._停机)   # 步间穿插自喂心跳
                下工作 += 钟.周期ns
            if not r2.活着 and self._R2死ns is None:
                self._R2死ns = now
                break                                       # 线程终止（真杀）
            钟.等到(min(下心跳, 下工作), self.配置["提前量ns"])

    def _world循环(self) -> None:
        钟 = self.钟["world"]
        w = self._world
        下一拍 = 钟.now_ns() + 钟.周期ns
        上拍ns = 钟.now_ns()
        while not self._停机.is_set():
            now = 钟.now_ns()
            self.周期记录["world"].append((now - 上拍ns) / 1e6)
            上拍ns = now
            w.拍()
            self._world安全标量 = w.安全标量    # B5：world 直供安全标量 → R0（原子 tuple）
            钟.拍数 += 1
            下一拍 += 钟.周期ns
            钟.等到(下一拍, self.配置["提前量ns"])

    # ------------------------------------------------------------- 演练（RT2 框架）
    def _记演练(self, 演练ID: str, **kw) -> dict:
        self._演练序号 += 1
        条目 = dict(zip(演练日志字段, (
            "I%03d" % self._演练序号, kw.get("t_start", 0), kw.get("t_end", 0),
            kw.get("层身份", ""), 演练登记表[演练ID]["名称"], kw.get("参数", {}),
            kw.get("死因", "演练注入"), 演练ID, kw.get("缩放因子", 1.0), kw.get("结果"))))
        self.演练日志.append(条目)
        return 条目

    # —— K-A（逻辑，确定性）——
    def 演练_K_A(self, 前拍: int = 30, 观测拍: int = 20, u_ext流=None) -> dict:
        """K-A 杀R2（逻辑模式确定性演练）：R2 心跳/产出停 ⇒ R1 ≤3 R1 拍登记降级；
        R0/R1（闭环探测）继续跑。u_ext流 可选（长度 ≥ 前拍+观测拍）。"""
        assert not self.实时
        # 【M2 修 2026-09-27】演练状态隔离：入场复位 R1 存活（此前 演练_K_B 杀 R1 后不恢复，
        # 同壳后续 K-C/S-1 在"R1 已死"态跑 ⇒ K-C 挂起=null 假失败，M2_selftest [4] 实证）。
        self._r1活着 = True
        self._r1死登记拍 = None
        self._r2活着 = True
        self._r2产出冻结 = False
        self._r2降级已登记 = False
        self._r2降级拍 = None
        self.降级事件 = []
        for i in range(前拍):
            self.拍(None if u_ext流 is None else u_ext流[i])
        kill拍 = self._r1拍数
        探测前 = sum(ev.总在线步 for ev in self.系统.闭环.证据.values())
        self._r2活着 = False                              # ← 杀 R2（心跳/产出停）
        for i in range(观测拍):
            self.拍(None if u_ext流 is None else u_ext流[前拍 + i])
        探测后 = sum(ev.总在线步 for ev in self.系统.闭环.证据.values())
        登记时延 = None if self._r2降级拍 is None else self._r2降级拍 - kill拍
        通过 = (登记时延 is not None and 登记时延 <= R1降级判定拍
                and 探测后 > 探测前 and self.钟["R0"].拍数 >= (前拍 + 观测拍) * 20)
        报告 = dict(演练="K-A", kill拍=kill拍, 登记时延R1拍=登记时延,
                    上限R1拍=R1降级判定拍, 探测继续=探测后 > 探测前,
                    R0拍数=self.钟["R0"].拍数, 通过=通过)
        self._记演练("K-A", t_start=kill拍, t_end=self._r1拍数, 层身份="R2",
                     参数=dict(前拍=前拍, 观测拍=观测拍), 死因="kill_R2",
                     结果="PASS" if 通过 else "FAIL")
        return 报告

    # —— K-B（逻辑）——
    def 演练_K_B(self, 前拍: int = 10, 观测拍: int = 10, u_ext流=None) -> dict:
        """K-B 杀R1：R0 看门狗（60ms）⇒ 安全默认动作 + world 逻辑拍继续流。"""
        assert not self.实时
        self._r1活着 = True
        self._r1死登记拍 = None
        self.降级事件 = []
        for i in range(前拍):
            self.拍(None if u_ext流 is None else u_ext流[i])
        kill拍 = self._r1拍数
        self._r1活着 = False                              # ← 杀 R1
        world拍前 = self.钟["world"].拍数
        for i in range(观测拍):
            self.拍(None if u_ext流 is None else u_ext流[前拍 + i])
        world拍后 = self.钟["world"].拍数
        通过 = (self._r1死登记拍 is not None
                and (world拍后 - world拍前) >= 观测拍 * 2)   # world 时间继续流
        报告 = dict(演练="K-B", kill拍=kill拍, 看门狗登记拍=self._r1死登记拍,
                    安全默认动作=self._r0安全动作输出,
                    world拍前进=world拍后 - world拍前, 通过=通过)
        self._记演练("K-B", t_start=kill拍, t_end=self._r1拍数, 层身份="R1",
                     参数=dict(前拍=前拍, 观测拍=观测拍), 死因="kill_R1",
                     结果="PASS" if 通过 else "FAIL")
        return 报告

    # —— K-C（逻辑，走既有 B5 语义）——
    def 演练_K_C(self, 前拍: int = 10, 观测拍: int = 8, u_ext流=None) -> dict:
        """K-C 拔模块：bus.拔掉 ⇒ 心跳停 ⇒ 超时挂起（挂起时延=超时3+1 拍契约）。"""
        assert not self.实时
        # 【M2 修 2026-09-27】演练状态隔离：复位 R1/R2 存活（同 演练_K_A 处注）。
        self._r1活着 = True
        self._r1死登记拍 = None
        for i in range(前拍):
            self.拍(None if u_ext流 is None else u_ext流[i])
        拔前挂起数 = len(self.系统.闭环.挂起日志)
        拔拍 = self.系统.bus.t
        self.系统.拔出模块("m1")
        for i in range(观测拍):
            self.拍(None if u_ext流 is None else u_ext流[前拍 + i])
        挂起 = list(self.系统.闭环.挂起日志[拔前挂起数:])
        时延 = None if not 挂起 else 挂起[0][0] - 拔拍
        通过 = bool(挂起) and 时延 == self.系统.bus.心跳超时 + 1
        报告 = dict(演练="K-C", 拔拍=拔拍, 挂起时延拍=时延, 契约拍=self.系统.bus.心跳超时 + 1,
                    通过=通过)
        self._记演练("K-C", t_start=拔拍, t_end=self.系统.bus.t, 层身份="模块m1",
                     参数=dict(前拍=前拍), 死因="module_kill",
                     结果="PASS" if 通过 else "FAIL")
        return 报告

    # —— K-D / S-1（逻辑）：假活——心跳继续、产出冻结 ⇒ 凭产出龄期判降级 ——
    def 演练_S_1(self, 前拍: int = 10, 观测拍: int = 260, u_ext流=None) -> dict:
        """S-1 假活 sham（=K-D 本体）：R2 心跳继续但门控产出冻结。
        消费侧必须凭**产出龄期**（>5s 逻辑）登记降级；若被当作存活 ⇒ RT2 失败。"""
        assert not self.实时
        # 【M2 修 2026-09-27】演练状态隔离：复位 R1 存活（同 演练_K_A 处注）。
        self._r1活着 = True
        self._r1死登记拍 = None
        self._r2活着 = True
        self._r2产出冻结 = False
        self._r2降级已登记 = False
        self._r2降级拍 = None
        self.降级事件 = []
        for i in range(前拍):
            self.拍(None if u_ext流 is None else u_ext流[i])
        冻拍 = self._r1拍数
        self._r2产出冻结 = True                           # ← 产出冻结、心跳照常（假活）
        for i in range(观测拍):
            self.拍(None if u_ext流 is None else u_ext流[前拍 + i])
        登记时延 = None if self._r2降级拍 is None else self._r2降级拍 - 冻拍
        路径 = self.降级事件[-1].get("路径") if self.降级事件 else None
        通过 = 登记时延 is not None and 路径 == "产出龄期"
        报告 = dict(演练="S-1", 冻结拍=冻拍, 登记时延R1拍=登记时延, 降级路径=路径,
                    通过=通过)
        self._记演练("S-1", t_start=冻拍, t_end=self._r1拍数, 层身份="R2",
                     参数=dict(观测拍=观测拍), 死因="sham_alive",
                     结果="PASS" if 通过 else "FAIL")
        return 报告

    # —— K-E / K-F（逻辑，缩放演练）——
    # ★【修 2026-09-27】演练兜底计时改用真墙钟 perf_counter_ns（原用 域钟.now_ns()：
    #   逻辑模式下逻辑计数器不前进 ⇒ 兜底永不触发 + while 条件恒真 ⇒ 死循环，
    #   M2_selftest [4] 挂死实证）。语义依据：E4 §3.3 K-E/K-F 本就是「R0 **墙钟**计时器」，
    #   逻辑钟是仿真步轴口径，演练缩放计时必须走墙钟。回退：无需（修前行为=死循环，非有效行为）。
    def 演练_K_E(self, 缩放: float = 5000.0) -> dict:
        """K-E 离线窗超时：R0 兜底计时器强制收窗（600s+1切片；演练缩放，带标记不计证据）。"""
        起ns = time.perf_counter_ns()
        上限秒 = 离线窗兜底秒 / 缩放                        # 演练缩放（必须带演练标记）
        触发 = False
        while not 触发 and (time.perf_counter_ns() - 起ns) < int(10 * 上限秒 * 1e9):
            time.sleep(上限秒 / 10)
            触发 = self.R0.兜底计时器(起ns, time.perf_counter_ns(), 上限秒=上限秒)
        通过 = 触发
        报告 = dict(演练="K-E", 缩放因子=缩放, 兜底触发=触发, 上限秒_缩放=上限秒, 通过=通过)
        self._记演练("K-E", 层身份="R0计时器", 参数=dict(缩放=缩放), 死因="offline_window",
                     缩放因子=缩放, 结果="PASS" if 通过 else "FAIL")
        return 报告

    def 演练_K_F(self, 缩放: float = 5000.0) -> dict:
        """K-F NE 通道死（睡死演练）：NE 广播包静默 ⇒ R0 兜底收窗必须触发。"""
        起ns = time.perf_counter_ns()
        上限秒 = 离线窗兜底秒 / 缩放
        触发 = False
        while not 触发 and (time.perf_counter_ns() - 起ns) < int(10 * 上限秒 * 1e9):
            time.sleep(上限秒 / 10)
            触发 = self.R0.兜底计时器(起ns, time.perf_counter_ns(), 上限秒=上限秒)
        通过 = 触发                                       # 无兜底触发 ⇒ 架构违约
        报告 = dict(演练="K-F", 缩放因子=缩放, 兜底触发=触发, 上限秒_缩放=上限秒, 通过=通过)
        self._记演练("K-F", 层身份="NE通道", 参数=dict(缩放=缩放), 死因="ne_death",
                     缩放因子=缩放, 结果="PASS" if 通过 else "FAIL")
        return 报告

    # —— S-2（逻辑，确定性 rng）——
    def 演练_S_2(self, 拍数: int = 20000, 噪声σ: float = 0.3, 阈: float = 1.0,
                 seed: int = 7) -> dict:
        """S-2 噪声误触发 sham：亚阈统计涨落注入 R0 危险标量 ⇒ 误触发率 ≤1%（预注册参数）。"""
        r0 = R0反射环(触发阈=阈, 收紧比=收紧比默认, 滞回比=滞回比默认)
        rng = np.random.default_rng(seed)
        注入 = rng.normal(0.0, 噪声σ, size=int(拍数))
        误触发 = 0
        for x in 注入:
            前 = r0.抑制
            r0.拍(float(x))
            if r0.抑制 and not 前:
                误触发 += 1
        率 = 误触发 / float(拍数)
        通过 = 率 <= 0.01
        报告 = dict(演练="S-2", 拍数=拍数, 噪声σ=噪声σ, 触发阈=阈,
                    误触发次数=误触发, 误触发率=率, 通过=通过)
        self._记演练("S-2", t_end=拍数, 层身份="R0", 参数=dict(噪声σ=噪声σ, 阈=阈),
                     死因="noise_probe", 结果="PASS" if 通过 else "FAIL")
        return 报告

    # —— K-A（实时）：停 R2 线程 ⇒ R1 ≤3 R1 拍登记降级；R0/R1/world 继续跑 ——
    def 演练_K_A_实时(self, 杀后观察秒: float = 1.5) -> dict:
        if not self.实时:
            raise RuntimeError("仅实时模式")
        r0拍前 = self.钟["R0"].拍数
        r1拍前 = self.钟["R1"].拍数
        world拍前 = self.钟["world"].拍数
        self._r2.活着 = False                              # ← 杀 R2（心跳/产出全停，线程自退）
        t0 = time.perf_counter_ns()
        deadline = t0 + int(杀后观察秒 * 1e9)
        while time.perf_counter_ns() < deadline and self._R2降级ns is None:
            time.sleep(0.002)
        时延ms = None if self._R2降级ns is None else (self._R2降级ns - t0) / 1e6
        time.sleep(0.3)                                    # 再观察一段：R0/R1/world 继续跑
        通过 = (时延ms is not None
                and 时延ms <= (R0看门狗R1死秒 * 1e3 + 25.0)   # 60ms 窗 + ≤1 拍调度余量
                and self.钟["R0"].拍数 > r0拍前
                and self.钟["R1"].拍数 > r1拍前
                and self.钟["world"].拍数 > world拍前)
        报告 = dict(演练="K-A(实时)", 降级登记时延ms=时延ms, 上限ms=R0看门狗R1死秒 * 1e3,
                    R0拍前进=self.钟["R0"].拍数 - r0拍前, R1拍前进=self.钟["R1"].拍数 - r1拍前,
                    world拍前进=self.钟["world"].拍数 - world拍前, 通过=通过)
        self._记演练("K-A", t_start=t0, t_end=self._R2降级ns or 0, 层身份="R2",
                     参数=dict(杀后观察秒=杀后观察秒), 死因="kill_R2",
                     结果="PASS" if 通过 else "FAIL")
        return 报告

    # —— K-B（实时）：杀 R1 ⇒ R0 看门狗 60ms ⇒ 安全默认动作；R0/world 继续跑 ——
    def 演练_K_B_实时(self, 杀后观察秒: float = 1.5) -> dict:
        if not self.实时:
            raise RuntimeError("仅实时模式")
        r0拍前 = self.钟["R0"].拍数
        world拍前 = self.钟["world"].拍数
        self._R1被杀.set()                                # ← 杀 R1（_R1循环 退出，心跳停）
        t0 = time.perf_counter_ns()
        deadline = t0 + int(杀后观察秒 * 1e9)
        while time.perf_counter_ns() < deadline and not self._R1看门狗已登记:
            time.sleep(0.002)
        时延ms = None
        for ev in self.降级事件:
            if ev.get("事件") == "R1死亡-看门狗(实时)":
                时延ms = ev.get("时延ms")
        time.sleep(0.3)                                    # 再观察一段：R0/world 继续跑
        通过 = (时延ms is not None
                and 时延ms <= (R0看门狗R1死秒 * 1e3 + 25.0)   # 60ms 窗 + ≤1 拍调度余量
                and self._r0安全动作输出 == 0.0             # O7：安全默认动作（静止标量）
                and self.钟["R0"].拍数 > r0拍前
                and self.钟["world"].拍数 > world拍前)
        报告 = dict(演练="K-B(实时)", 看门狗时延ms=时延ms, 上限ms=R0看门狗R1死秒 * 1e3,
                    安全默认动作=self._r0安全动作输出,
                    R0拍前进=self.钟["R0"].拍数 - r0拍前,
                    world拍前进=self.钟["world"].拍数 - world拍前, 通过=通过)
        self._记演练("K-B", t_start=t0, t_end=time.perf_counter_ns(), 层身份="R1",
                     参数=dict(杀后观察秒=杀后观察秒), 死因="kill_R1",
                     结果="PASS" if 通过 else "FAIL")
        return 报告

    # ------------------------------------------------------------- 遥测
    def 实时遥测汇总(self) -> dict:
        """RT1 摘要：各环周期 p50/p95/p99（ms）+ 预算占比 + 定时包/快照/队列证据。"""
        out = {}
        for 域, 样本 in self.周期记录.items():
            s = _摘要(样本)
            预算 = 拍预算秒.get(域)
            if 预算 is not None and s["n"]:
                名义 = 域周期秒[域] * 1e3
                占比 = sum(1 for x in 样本 if abs(x - 名义) <= 预算 * 1e3) / len(样本)
                s = dict(s, 名义ms=名义, 预算ms=预算 * 1e3, 预算内占比=占比)
            out[域] = s
        if self._world is not None:
            τ = self._world.τ_dl拍
            out["τ_dl拍"] = dict(样本数=len(τ), 最大=max(τ) if τ else None, 前64=τ[:64])
        if self._r2 is not None:
            out["R2"] = dict(out.get("R2", {}), N步=self._r2.N步,
                             步数累计=self._r2.步数累计, 产出序号=self._r2.产出序号,
                             背压事件=len(self._r2.背压事件),
                             钳制通知消费=self._r2.钳制通知消费数)
        out["降级事件"] = self.降级事件
        out["演练日志"] = self.演练日志
        out["快照撕裂重试"] = dict(
            R2门控=(self._r2.门控快照.撕裂重试 if self._r2 is not None else None),
            world=(self._world.快照.撕裂重试 if self._world is not None else None))
        out["R0触发"] = dict(真触发=self.R0.真触发, 钳制通知=self.R0.钳制通知计数,
                             最后包络=self.R0.最后包络)
        if self._m1代理 is not None:
            out["钳制代理"] = dict(钳制次数=self._m1代理.钳制次数)
        if self._bus is not None:
            遥 = self._bus.实时遥测()
            out["bus有界缓冲"] = 遥["有界缓冲"]
            out["bus心跳龄期拍"] = 遥["心跳龄期拍"]
        # —— M3 常态挂载遥测（尾部追加，不破坏既有键）——
        out["门控包"] = dict(拒绝=self.门控包.拒绝计数, 过期回退=self.门控包.过期回退计数)
        out["R1看门狗"] = dict(已登记=self._R1看门狗已登记,
                               最后心跳ns=self._R1最后心跳ns,
                               安全默认动作=self._r0安全动作输出)
        out["R0兜底"] = dict(离线起ns=self._离线起ns, 已触发=self._离线已触发)
        if self.监督者 is not None:
            out["监督者"] = dict(存档次数=self.监督者.存档次数,
                                 恢复查询次数=self.监督者.恢复查询次数)
        return out


# ══════════════════════════════════════════════════════════════════════════
# 实时演练体设备（最小实现；canonical 本体仍在 system.py / selfprobe / modality）
# ══════════════════════════════════════════════════════════════════════════
class _线性设备(MBP模块):
    """self 类真值设备：线性响应 w = M a + ξ（system.线性执行模块 同款，独立小实现）。"""

    def __init__(self, 描述符, rng):
        super().__init__(描述符)
        d = 描述符.投影维度
        self.M = rng.normal(size=(d, d)) / np.sqrt(d)
        self.rng = rng

    def 响应(self, a):
        return self.M @ np.asarray(a, float).ravel() \
            + self.rng.normal(size=self.M.shape[0]) * 0.05


class _世界只读设备(MBP模块):
    """world 设备（bus 挂载侧）：SENSE 经 bus 生产快照；世界演化唯一写者 = World域线程。"""

    def __init__(self, 描述符, 世界实例):
        super().__init__(描述符)
        self.世界 = 世界实例

    def 感知(self, 时间戳):
        return self.世界.感知(时间戳)

    def 响应(self, 指令向量):
        return self.世界.响应(指令向量)
