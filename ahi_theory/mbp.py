# -*- coding: utf-8 -*-
"""ahi_theory.mbp —— MBP（Module Bus Protocol）模块总线（AHI 2.2 / R1）。

规格出处：`03_AHI/AHI2.1/完整` §5.1–§5.12；`.trae/specs/add-ahi22-n1n4-and-mbp/spec.md`。
覆盖：B1 包结构 / B2 描述符与注册 / B3 SENSE / B4 CMD·RESP / B5 心跳与离线 /
      B6 自我探测 `self_t = σ(−e/τ)` / B7 CAL / B8 多模块独立 / B9 换身体。

★★ 事实更正（spec D4，已取证）★★
  任务书 §5.11 称本包是「现有 `BroadcastPacket` 的扩展，只加三个字段」。
  **该对象在 AHI2.1 的 `ahi_theory/` 内不存在**（它在母项目 `03_AHI/核心/接口协议.py`；
  `ahi_p0/wiring.py` 是另一套未接入 `brain.py` 的器官总线）。因此本文件是 MBP 的
  **全新定义**，且**不 import** 母项目 / `ahi_p0` / `brain.py`，保持 `ahi_theory` 包内自洽。

★ 设计纪律（本项目通用）
  · 判据可失败、非恒真：不做 `hasattr` 探测、不读回自己写的超参、不用字符串正则凑通过；
    本文件不 import `brain.py`，也不被 `brain.py` import。
  · 「自我探测」判别力靠**实测 AUC** 与其**未学习对照**共同证明（见 meeting5/MBP_selftest.py），
    而不是靠常量比较。
  · 未实测数字一律标注。

命名沿用任务书的中文标识符（MBP包 / 模块ID / 包类型 / …）。
"""
from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field as _dc_field, fields as _dc_fields
from typing import Optional, Sequence

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# 【M2 底座改造登记 · 2026-09-27 · 白名单逐处对照版】
# 授权依据：meeting11/95_架构定稿.md §2-L1（C1 闭合）；
#   meeting11/E4_Round3_签署.md §1（锁死面 + 白名单 W1–W5 + 四道闸）；
#   meeting11/E3_Round3_签署.md §1.1（让渡清单六处 + 条件三件套）。
# 两清单对照（出入登记，以交集为准）：
#   E3 让渡六处 = E4 白名单五处（W1↔E3①、W2↔E3②、W4↔E3④、W5↔E3⑤、W3 采样率数值化↔E3⑥）
#   + E3③「定时包四字段纯追加」——E4 把这一处归**锁死面 2「新增冻结接口」**而非白名单条目
#   （归属差异如实登记：E4 视其为冻结接口追加、E3 视其为让渡改动；实现=纯追加定时包子类，
#   两边要求都满足，交集=「九字段语义不动 + 字段清单冻结」）。
# 锁死面（绝对不动）：MBP包九字段语义 / 载荷哈希校验 / 单调序列号 / 降级包约定 /
#   B9 档案接口 / 各模块方法签名 / 定时包冻结字段清单（定时包冻结字段 常量）。
# 改造形式（四道闸）：闸1 预注册=本块+`verify_all._CHANGELOG` 2026-09-27 M2 条目（先于改码）；
#   闸2 惰性开关=总线形参 `实时模式`（默认 False）；闸3 回归=meeting5/MBP_selftest.py 逐位
#   + meeting11/M2_selftest + M6 锚点两值；闸4 封闭性=白名单仅此五处，永不隐性扩充。
# 双口径（仿真步/墙钟，E3 条件三件套之轴纪律）：**步轴（bus.t / brain.t）为判据本体，
#   永不墙钟化**；`定时包.单调时间戳` 实时模式=perf_counter_ns（墙钟执行合同）/
#   非实时=int(bus.t)×1e6 ns（仿真步换算）——两口径并存，判据只吃步轴。
# 逐处登记（改动 → 证据 → 回退方式）：
#   [W1 = E3让渡①]【2026-09-27】步进() 实时模式不再代发心跳（上帝循环废除，调度权上移
#     three_loop 三环壳，各环各自 tick）。证据：INT G1 双时钟推进。
#     回退：实时模式=False ⇒ 旧代发路径逐位（diff=0）。
#   [W2 = E3让渡②]【2026-09-27】读() 实时模式走 快照槽（seqlock，读不触设备/不推进 EMA/
#     不出队/不增读序列号）+ 新增 写者侧 生产快照()。证据：INT G2 同拍双消费者互偷。
#     回退：实时模式=False ⇒ 旧有副作用读逐位。
#   [E3③ = E4 锁死面2]【2026-09-27】定时包 子类纯追加冻结接口字段（时钟域/单调时间戳/
#     龄期上限/传输延迟/调节中/ack序号/钳制通知/offline旗标/演练标记/附加），
#     九字段语义不动。回退：纯新增类，不使用即无痕。
#   [W3 = E3让渡⑥]【2026-09-27】采样率/延迟字符串→数值（采样率数值()/延迟数值()），
#     入 定时包.传输延迟；描述符字段不动（指纹不变）。证据：病 1 字符串无消费者。
#     回退：纯新增函数。
#   [W4 = E3让渡④]【2026-09-27】心跳自喂 喂心跳()（被监控环自喂）+ 巡检()（R0 侧龄期
#     查表）；实时模式 是否离线() 改「在线标志 ∧ 心跳新鲜」。证据：病 6 自己喂自己的狗。
#     回退：实时模式=False ⇒ 旧公式逐位；喂心跳()/巡检() 纯新增。
#   [W5 = E3让渡⑤]【2026-09-27】裸 list→环形缓冲（实时模式：事件队列深 64 / 探测记录
#     4096，溢出=丢最旧+计数，不静默）。证据：病 6 内存无界；E4 §3.5。
#     回退：实时模式=False ⇒ 仍为裸 list。
# 实证：M2_selftest [1]（三环壳逻辑模式 canonical 逐位对照 max|Δ|=0）、[2]（默认关等价）；
#   meeting5/MBP_selftest.py 逐位通过（既有判据不受影响）。
# ═══════════════════════════════════════════════════════════════════════════

# ------------------------------------------------------------------ 常量
包类型集合 = ("DESC", "SENSE", "CMD", "RESP", "BEAT", "CAL")
模块类集合 = ("感知类", "执行类", "双向类", "计算类")
时间特性集合 = ("快照", "流式", "事件")
协议版本默认 = "MBP-1.0"
投影维度默认 = 512

#: 已知模态 → 固定投影器（§5.9「已知类型提供固定投影器」）。其余投影器名一律视为未知模态 ⇒ 可在线学。
已知投影器 = ("φ_vol", "φ_vis", "φ_aud", "φ_act", "φ_universal")

#: 描述符数值编码用的枚举
_模块类码 = {"感知类": 0, "执行类": 1, "双向类": 2, "计算类": 3}
_时间特性码 = {"快照": 0, "流式": 1, "事件": 2}


def _种子(*parts) -> int:
    """稳定种子（禁止用 `hash()`：字符串哈希逐进程随机化，会破坏可复现性）。"""
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:12], 16)


def _sha16(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()[:16]


def 载荷哈希(向量) -> str:
    """B1：`校验` = 对 `向量` 的 float64 字节做 sha256 取前 16 位十六进制。

    对载荷的**任何**改动（哪怕一个 ULP）都会改变字节流 ⇒ 校验失败。
    """
    a = np.ascontiguousarray(np.asarray(向量, dtype=np.float64))
    return _sha16(a.tobytes())


# ================================================================== B1 包结构
@dataclass
class MBP包:
    """MBP 包（九字段，§5.3）。"""

    模块ID: str
    包类型: str
    时间戳: int
    向量: np.ndarray
    能量: float = 0.0
    置信: float = 1.0
    代谢: float = 0.0
    序列号: int = 0
    校验: str = ""

    def __post_init__(self):
        if self.包类型 not in 包类型集合:
            raise ValueError("非法包类型：%r（合法：%s）" % (self.包类型, 包类型集合))
        self.向量 = np.ascontiguousarray(np.asarray(self.向量, dtype=np.float64)).copy()
        if not self.校验:
            self.校验 = 载荷哈希(self.向量)

    def 验证(self) -> bool:
        """重算载荷哈希并与 `校验` 比对 ⇒ 能识破对 `向量` 的篡改。"""
        return self.校验 == 载荷哈希(self.向量)

    def 深拷贝(self) -> "MBP包":
        return MBP包(self.模块ID, self.包类型, self.时间戳, self.向量,
                     self.能量, self.置信, self.代谢, self.序列号, self.校验)


def 字段名() -> tuple:
    """MBP包 的九字段名（供验收逐条比对，非 `hasattr` 探测）。"""
    return tuple(f.name for f in _dc_fields(MBP包))


# ------------------------------------------------------------------ 【M2·W3】定时包（冻结接口，纯追加）
#: 定时包冻结接口追加字段（E4_Round3 §1 锁死面 2："本轮新增冻结接口"——逐字对应登记）
定时包冻结字段 = ("时钟域", "单调时间戳", "龄期上限", "传输延迟",
                  "调节中", "ack序号", "钳制通知", "offline旗标", "演练标记", "附加")
#: 时钟域枚举（95_架构定稿 §2-L1 六域；caregiver/持久 域 M9/M11 落位，此处先登记枚举）
时钟域集合 = ("R0", "R1", "R2", "world", "caregiver", "持久")

_采样率单位表 = {"khz": 1e3, "mhz": 1e6, "hz": 1.0}
_延迟单位表 = {"ms": 1e-3, "us": 1e-6, "ns": 1e-9, "s": 1.0}


def 采样率数值(字符串: str) -> float:
    """【M2·W3】采样率字符串 → 数值 Hz（病 1 根治：字符串无消费者 → 数值入定时包）。
    仅做解析，不改 描述符 结构（指纹不变）。无法解析 ⇒ ValueError（可失败，不静默）。"""
    s = str(字符串).strip().lower()
    for suf, k in _采样率单位表.items():
        if s.endswith(suf):
            return float(s[: -len(suf)]) * k
    return float(s)


def 延迟数值(字符串: str) -> float:
    """【M2·W3】延迟字符串 → 数值秒。无法解析 ⇒ ValueError（可失败，不静默）。"""
    s = str(字符串).strip().lower()
    for suf, k in _延迟单位表.items():
        if s.endswith(suf):
            return float(s[: -len(suf)]) * k
    return float(s)


@dataclass
class 定时包(MBP包):
    """【M2·W3·2026-09-27】定时包 = MBP包（九字段语义锁死）+ 追加冻结接口字段。

    追加字段逐条对应 E4_Round3 §1 锁死面 2 的冻结接口清单；消费侧凭
    `单调时间戳 + 传输延迟` 的龄期与 `龄期上限` 做降级判定（置信承载"数据多旧"）。
    """

    时钟域: str = "R1"
    单调时间戳: int = 0            # ns（写者域内单调钟读数；实时路径 = perf_counter_ns）
    龄期上限: float = 5.0          # 秒；超龄 ⇒ 消费侧按降级包处理（复用降级语义）
    传输延迟: float = 0.0          # 秒（W3：数值字段，源自描述符.延迟 解析）
    调节中: bool = False           # S10 三态龄期原语：调节中的变量不回退、不积分
    ack序号: int = -1              # 背压件一：消费确认（R1→R2 长期无 ack ⇒ 写者降速）
    钳制通知: bool = False         # 背压件二：R0→R2 反 windup 通知（I_c 冻结）
    offline旗标: bool = False      # MBP offline 旗标（E1 旗标滞回消费）
    演练标记: str = ""             # 演练 schema id（演练窗互斥审计）；空 = 非演练包
    附加: dict = _dc_field(default_factory=dict)   # caregiver 延迟留白等（系统侧置 NULL）

    def __post_init__(self):
        super().__post_init__()
        if self.时钟域 not in 时钟域集合:
            raise ValueError("非法时钟域：%r（合法：%s）" % (self.时钟域, 时钟域集合))

    def 深拷贝(self) -> "定时包":
        q = 定时包(self.模块ID, self.包类型, self.时间戳, self.向量,
                   self.能量, self.置信, self.代谢, self.序列号, self.校验,
                   时钟域=self.时钟域, 单调时间戳=self.单调时间戳, 龄期上限=self.龄期上限,
                   传输延迟=self.传输延迟, 调节中=self.调节中, ack序号=self.ack序号,
                   钳制通知=self.钳制通知, offline旗标=self.offline旗标,
                   演练标记=self.演练标记, 附加=dict(self.附加))
        return q


# ------------------------------------------------------------------ 【M2·W5】有界环形缓冲
class 环形缓冲:
    """【M2·W5·2026-09-27】有界环形缓冲：满 ⇒ 丢最旧 + 溢出计数（不静默积压；E4 §3.5）。

    仅实现 mbp 既有路径用到的 list 接口子集（append / pop(0) / len / bool / clear / 迭代 / 索引）。
    非实时（默认）路径不使用本类 ⇒ 行为逐位不变。
    """

    def __init__(self, 上限: int):
        self.上限 = int(上限)
        self._q = deque()
        self.溢出数 = 0

    def append(self, x) -> None:
        if len(self._q) >= self.上限:
            self._q.popleft()
            self.溢出数 += 1
        self._q.append(x)

    def pop(self, i: int = 0):
        if i == 0:
            return self._q.popleft()
        return self._q.pop(i)   # 负索引/中段弹出（既有路径未用，兜底）

    def popleft(self):
        return self._q.popleft()

    def clear(self) -> None:
        self._q.clear()

    def __len__(self) -> int:
        return len(self._q)

    def __bool__(self) -> bool:
        return len(self._q) > 0

    def __iter__(self):
        return iter(self._q)

    def __getitem__(self, i):
        return self._q[i]


# ------------------------------------------------------------------ 【M2·W2】快照槽（seqlock）
class 快照槽:
    """【M2·W2·2026-09-27】单写多读 seqlock 快照槽（G2 根治：读() 去副作用）。

    写者：seq←奇 → 载荷引用单次赋值（对象一次性构建）→ seq←偶；
    读者：seq₁ → 载荷 → seq₂，seq₁==seq₂ 且为偶 ⇒ 一致读；否则重试并计 `撕裂重试`。
    载荷为不可变引用（numpy 数组按构建即定）⇒ GIL 下引用读原子；seqlock 为形式守卫。
    """

    __slots__ = ("_seq", "_包", "撕裂重试", "写版本")

    def __init__(self):
        self._seq = 0
        self._包 = None
        self.撕裂重试 = 0
        self.写版本 = 0

    def 写(self, 包) -> None:
        self._seq += 1          # 奇 = 写入中
        self._包 = 包
        self._seq += 1          # 偶 = 稳定
        self.写版本 += 1

    def 读(self):
        """读快照：seq₁ → 载荷 → seq₂ 一致性校验。
        【M2·W2 修 2026-09-27】空槽（尚无写入：seq 偶且载荷 None 且稳定）**直接返回 None、
        不计撕裂**——空 ≠ 撕裂（原实现把空槽读也计入 撕裂重试，污染实时遥测判读，
        M2_selftest [3] 实测 176 次假撕裂实证）。仅 seq 为奇（写入中）或 seq 变化（读中改写）
        才计撕裂。默认（非实时）路径不使用本类 ⇒ 行为逐位不变。"""
        for _ in range(8):
            s1 = self._seq
            if s1 & 1:                      # 写者正在写 ⇒ 重试
                self.撕裂重试 += 1
                continue
            包 = self._包
            if 包 is None:                  # 空槽：非撕裂，直接返回空（消费侧走降级语义）
                return None
            if s1 == self._seq:
                return 包
            self.撕裂重试 += 1
        return self._包                     # 兜底（重试上限后取最后引用；计数可见）


# ================================================================== B2 描述符
@dataclass
class 模块描述符:
    """模块描述符（§5.4）。三个正交维度：时间特性 / 维度 / 投影器（§5.5）。"""

    模块ID: str
    模块类: str
    时间特性: str
    原始维度: int
    投影维度: int = 投影维度默认
    投影器: str = "φ_universal"
    采样率: str = "1kHz"
    缓冲深度: int = 1
    代谢常数: float = 0.01
    延迟: str = "1ms"
    标定状态: str = "未标定"
    能力: tuple = ()
    协议版本: str = 协议版本默认

    def __post_init__(self):
        if self.模块类 not in 模块类集合:
            raise ValueError("非法模块类：%r（合法：%s）" % (self.模块类, 模块类集合))
        if self.时间特性 not in 时间特性集合:
            raise ValueError("非法时间特性：%r（合法：%s）" % (self.时间特性, 时间特性集合))
        self.原始维度 = int(self.原始维度)
        self.投影维度 = int(self.投影维度)
        self.缓冲深度 = int(self.缓冲深度)
        self.代谢常数 = float(self.代谢常数)
        self.能力 = tuple(self.能力)

    def 指纹(self) -> str:
        s = "|".join([self.模块ID, self.模块类, self.时间特性, str(self.原始维度),
                      str(self.投影维度), self.投影器, self.采样率, str(self.缓冲深度),
                      "%.6g" % self.代谢常数, self.延迟, self.标定状态,
                      ",".join(self.能力), self.协议版本])
        return _sha16(s.encode("utf-8"))

    def 向量(self) -> np.ndarray:
        """DESC 包的载荷：描述符的可校验数值编码（内核收到 DESC 后可核对一致性）。"""
        return np.array([_模块类码[self.模块类], _时间特性码[self.时间特性],
                         self.原始维度, self.投影维度, self.缓冲深度,
                         self.代谢常数, float(len(self.能力))], dtype=np.float64)


# ================================================================== 投影器 §5.9
class 通用投影器:
    """`φ: R^n → R^d`（§5.9）。已知模态用固定投影器（`可学=False`）；
    未知模态初始化随机 φ，在探测循环里**在线学习**（本类即最小二乘在线版）。
    """

    def __init__(self, 原始维度: int, 投影维度: int = 投影维度默认,
                 seed: int = 0, 可学: bool = True):
        self.n = int(原始维度)
        self.d = int(投影维度)
        rng = np.random.default_rng(int(seed) % (2 ** 63))
        self.W = rng.normal(size=(self.d, self.n)) / np.sqrt(max(self.n, 1))
        self.b = np.zeros(self.d)
        self.可学 = bool(可学)
        self._S = np.zeros((self.n + 1, self.n + 1))
        self._B = np.zeros((self.d, self.n + 1))
        self.已收敛 = False
        self.n更新 = 0

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        return self.W @ x + self.b

    def 指纹(self) -> str:
        return _sha16(np.ascontiguousarray(self.W, dtype=np.float64).tobytes())

    def 更新(self, x, y) -> None:
        if not self.可学:
            return
        xa = np.append(np.asarray(x, dtype=float).ravel(), 1.0)
        y = np.asarray(y, dtype=float).ravel()
        self._S += np.outer(xa, xa)
        self._B += np.outer(y, xa)
        self.n更新 += 1
        if self.n更新 < self.n + 1:      # 欠定阶段不反演，避免秩 1 逆爆炸
            return
        eps = 1e-6 * max(float(np.trace(self._S)) / max(self.n + 1, 1), 1e-9)
        M = self._B @ np.linalg.inv(self._S + eps * np.eye(self.n + 1))
        self.W = M[:, :-1]
        self.b = M[:, -1]

    def 误差(self, x, y) -> float:
        p = self(np.asarray(x, dtype=float).ravel())
        return float(np.sum((np.asarray(y, dtype=float).ravel() - p) ** 2))


# ================================================================== 在线响应模型 §5.7
class 在线响应模型:
    """`ŵ = f(a)`：执行类/双向类模块的**响应模型**（不是模块内部公式）。

    f 为线性映射 `W·[a;1]`：
      · `初始化="恒等"`（默认，仅当 指令维度==响应维度）—— 初始 f 即恒等映射；
      · `初始化="零"` —— 零线性映射（用作「未学习」对照）。
    在线最小二乘：`S += [a;1][a;1]^T`、`B += w[a;1]^T`、`W = B·S^{-1}`（相对岭防爆）。
    预测误差 `e = ‖w − ŵ‖²`。**AI 学的是响应模型，模块内部是 PID / 网络 / 常数都无所谓。**
    """

    def __init__(self, 指令维度: int, 响应维度: int, 初始化: str = "恒等", ridge: float = 1e-3):
        self.ma = int(指令维度)
        self.mw = int(响应维度)
        self.初始化 = str(初始化)
        self.W = np.zeros((self.mw, self.ma + 1))
        if self.初始化 == "恒等" and self.ma == self.mw:
            self.W[:, :self.ma] = np.eye(self.ma)
        self._S = np.zeros((self.ma + 1, self.ma + 1))
        self._B = np.zeros((self.mw, self.ma + 1))
        self.ridge = float(ridge)
        self.n更新 = 0

    def _aug(self, a) -> np.ndarray:
        return np.append(np.asarray(a, dtype=float).ravel(), 1.0)

    def 预测(self, a) -> np.ndarray:
        return self.W @ self._aug(a)

    def 更新(self, a, w) -> None:
        xa = self._aug(a)
        w = np.asarray(w, dtype=float).ravel()
        self._S += np.outer(xa, xa)
        self._B += np.outer(w, xa)
        self.n更新 += 1
        if self.n更新 < self.ma + 1:     # 欠定阶段保持初始 f
            return
        eps = self.ridge * max(float(np.trace(self._S)) / max(self.ma + 1, 1), 1e-12)
        self.W = self._B @ np.linalg.inv(self._S + eps * np.eye(self.ma + 1))

    def 误差(self, a, w) -> float:
        return float(np.sum((np.asarray(w, dtype=float).ravel() - self.预测(a)) ** 2))

    def 指纹(self) -> str:
        return _sha16(np.ascontiguousarray(self.W, dtype=np.float64).tobytes())


# ================================================================== Device 侧基类
class MBP模块:
    """被总线挂载的模块（Device 侧基类）。总线只经由**描述符 + 四类接口**与之交互。

    `在线=True` 表示模块在插着（会按步进发心跳）；`拔掉()` 置 False ⇒ 心跳停 ⇒ 离线。
    """

    def __init__(self, 描述符: 模块描述符):
        self.描述符 = 描述符
        self.在线 = True

    # --- 感知类 / 双向类：SENSE ---
    def 感知(self, 时间戳: int):        # 返回**原始维度**向量；事件型无事件时返回 None
        return np.zeros(self.描述符.原始维度)

    # --- 执行类 / 双向类：CMD → RESP ---
    def 响应(self, 指令向量: np.ndarray) -> np.ndarray:
        return np.zeros(self.描述符.投影维度)

    # --- B7：CAL 往返 ---
    def 标定(self, 标定向量: np.ndarray) -> np.ndarray:
        return np.zeros(self.描述符.投影维度)

    # --- §5.9：未知模态投影器在线学习用的探针响应（原始信号 → 母语空间目标）---
    def 探针响应(self, 原始信号: np.ndarray) -> np.ndarray:
        return np.zeros(self.描述符.投影维度)

    # --- 计算类：前向 ---
    def 前向(self, 输入):
        return np.zeros(self.描述符.投影维度)


class _记录:
    """总线侧的一个模块槽位。"""

    def __init__(self, 描述符: 模块描述符, 设备: MBP模块, 投影器: 通用投影器,
                 响应模型初始化: str = "恒等"):
        self.描述符 = 描述符
        self.设备 = 设备
        self.投影器 = 投影器
        self.序列号 = 0
        self.最后心跳 = 0
        self.离线 = False
        self.上帧 = None
        self.帧计数 = 0
        self.事件队列 = []
        self.标定参数 = None
        self.最后误差 = float("nan")
        self.探测记录 = []          # [(a, w, e_pre), ...]【M2·W5】实时模式下替换为 环形缓冲
        self.快照槽 = None          # 【M2·W2】实时模式下为 快照槽（seqlock）；默认 None ⇒ 零影响
        self.响应模型 = None
        if 描述符.模块类 in ("执行类", "双向类"):
            self.响应模型 = 在线响应模型(描述符.投影维度, 描述符.投影维度,
                                         初始化=响应模型初始化)

    def 复位时序(self, t: int):
        self.最后心跳 = int(t)
        self.离线 = False


# ================================================================== 总线（Host）
class MBP总线:
    """MBP 总线（Host）。**内核只认包，不认硬件**（§5.1）。

    · 自我档案（持久身份）：身份锚点向量 / 协议版本 / 内核投影器指纹 / 探测 τ / 投影维度。
      这些**不随身体改变** ⇒ 换模块后逐位不变（B9）。
    · 身体图式（每次插拔重建）：已注册模块集合 + 各模块投影器指纹 + 标定状态 + 描述符指纹。
    """

    def __init__(self, 投影维度默认_: int = 投影维度默认, tau: float = 1.0,
                 心跳超时: int = 3, seed: int = 0, 协议版本: str = 协议版本默认,
                 响应模型初始化: str = "恒等",
                 实时模式: bool = False, 实时配置: Optional[dict] = None):
        self.投影维度 = int(投影维度默认_)
        self.tau = float(tau)                    # self_t = σ(−e/τ)，τ 默认 1.0
        self.心跳超时 = int(心跳超时)
        self.协议版本 = str(协议版本)
        self.响应模型初始化 = str(响应模型初始化)
        self.t = 0
        self._seed = int(seed)
        self._rng = np.random.default_rng(int(seed))
        self._表: dict = {}
        # ---- 【M2·W1–W5·2026-09-27】惰性开关（四道闸·闸2）：默认 False ⇒ 既有行为逐位不变 ----
        self.实时模式 = bool(实时模式)
        self.实时配置 = dict(队列深=64,          # E4 §3.1：队列深 64
                             探测记录上限=4096,  # E4 §3.1：环形 4096 条
                             龄期上限秒=5.0,     # R2 产出龄期上限（三态龄期默认档）
                             时钟域默认="R1")
        if 实时配置:
            self.实时配置.update(dict(实时配置))

        # ---- 持久身份（自我档案）----
        anc = self._rng.normal(size=16)
        self.身份锚点 = anc / np.linalg.norm(anc)
        # 内核自己的「母语」投影器 φ_universal（内核拥有，与身体无关 ⇒ 指纹不变）
        self._内核投影器 = 通用投影器(4, self.投影维度,
                                       seed=_种子("内核", int(seed), self.投影维度), 可学=False)

    # ---------------------------------------------------------- B2 注册 / 注销 / 查询
    def _建投影器(self, d: 模块描述符) -> 通用投影器:
        if d.投影器 in 已知投影器:
            return 通用投影器(d.原始维度, d.投影维度,
                              seed=_种子("已知", d.投影器, d.原始维度, d.投影维度), 可学=False)
        return 通用投影器(d.原始维度, d.投影维度,
                          seed=_种子("未知", self._seed, d.模块ID, d.投影器), 可学=True)

    def 注册(self, 描述符: 模块描述符, 设备: Optional[MBP模块] = None) -> MBP包:
        """模块上电 → 广播 DESC → 内核校验 → 注册。"""
        if 设备 is None:
            设备 = MBP模块(描述符)
        rec = _记录(描述符, 设备, self._建投影器(描述符),
                     响应模型初始化=self.响应模型初始化)
        if self.实时模式:      # 【M2·W5】裸 list → 有界环形缓冲（溢出=丢最旧+计数）＋ 快照槽（W2）
            rec.事件队列 = 环形缓冲(self.实时配置["队列深"])
            rec.探测记录 = 环形缓冲(self.实时配置["探测记录上限"])
            rec.快照槽 = 快照槽()
        rec.复位时序(self.t)
        self._表[描述符.模块ID] = rec
        rec.序列号 += 1
        return MBP包(描述符.模块ID, "DESC", self.t, 描述符.向量(),
                     能量=float(np.linalg.norm(描述符.向量())), 置信=1.0,
                     代谢=描述符.代谢常数, 序列号=rec.序列号)

    def 注销(self, 模块ID: str) -> bool:
        return self._表.pop(模块ID, None) is not None

    def 查询(self, 模块ID: str) -> Optional[模块描述符]:
        rec = self._表.get(模块ID)
        return None if rec is None else rec.描述符

    def 取设备(self, 模块ID: str) -> Optional[MBP模块]:
        rec = self._表.get(模块ID)
        return None if rec is None else rec.设备

    def 取投影器(self, 模块ID: str) -> Optional[通用投影器]:
        rec = self._表.get(模块ID)
        return None if rec is None else rec.投影器

    def 取响应模型(self, 模块ID: str) -> Optional[在线响应模型]:
        rec = self._表.get(模块ID)
        return None if rec is None else rec.响应模型

    def 取标定参数(self, 模块ID: str):
        rec = self._表.get(模块ID)
        return None if rec is None else rec.标定参数

    def 已注册(self) -> tuple:
        return tuple(sorted(self._表.keys()))

    # ---------------------------------------------------------- B5 心跳 / 离线 / 降级
    def 是否离线(self, 模块ID: str) -> bool:
        rec = self._表.get(模块ID)
        if rec is None:
            return True
        if self.实时模式:
            # 【M2·W4·2026-09-27】自喂心跳语义：在线 = 在线标志 ∧ 心跳新鲜。
            # 环停止自喂（层死/挂起）⇒ 超时判离线；与旧公式同一超时常量（默认 3 拍）。
            rec.离线 = rec.设备.在线 and (self.t - rec.最后心跳) > self.心跳超时
            return rec.离线
        if rec.设备.在线:
            rec.离线 = False
        else:
            rec.离线 = (self.t - rec.最后心跳) > self.心跳超时
        return rec.离线

    def 喂心跳(self, 模块ID: str) -> bool:
        """【M2·W4·2026-09-27】被监控环自喂心跳（实时模型）：只登记新鲜度，不造包、不代发。
        与 心跳() 的区别：心跳() 是总线代发的 B5 语义（保留不动），本方法是设备/环
        在自己节拍里的自喂入口（狗不再自己喂自己——病 6 根治，R0 巡检可判真死亡）。"""
        rec = self._表.get(模块ID)
        if rec is None:
            return False
        rec.最后心跳 = self.t
        rec.离线 = False
        return True

    def 巡检(self) -> dict:
        """【M2·W4·2026-09-27】R0 侧巡检视角：各模块自喂心跳龄期（拍）。纯读、无副作用。"""
        return {mid: int(self.t - rec.最后心跳) for mid, rec in sorted(self._表.items())}

    def 心跳(self, 模块ID: str, 载荷=None) -> MBP包:
        rec = self._表.get(模块ID)
        if rec is None:
            return self._降级包(模块ID, "BEAT")
        rec.最后心跳 = self.t
        rec.离线 = False
        p = (np.zeros(rec.描述符.投影维度) if 载荷 is None
             else np.asarray(载荷, dtype=float).ravel())
        rec.序列号 += 1
        return MBP包(模块ID, "BEAT", self.t, p, 序列号=rec.序列号)

    def 步进(self, 步数: int = 1) -> int:
        """推进内核时钟：在线的模块会随步进发心跳；被拔掉的模块心跳停 ⇒ 超时后离线。

        【M2·W1·2026-09-27】实时模式下本方法**不再代发心跳**（上帝循环废除——调度权上移
        three_loop 三环壳，各环各自 tick、各模块自喂心跳 + R0 巡检）；步轴（bus.t）仅 +1。
        默认（非实时）路径逐位保持旧行为。
        """
        for _ in range(int(步数)):
            self.t += 1
            if self.实时模式:      # 【M2·W1/W4】不代发心跳：各环自喂 + R0 巡检
                continue
            for rec in self._表.values():
                if rec.设备.在线:
                    rec.最后心跳 = self.t
                    rec.离线 = False
                else:
                    rec.离线 = (self.t - rec.最后心跳) > self.心跳超时
        return self.t

    def 拔掉(self, 模块ID: str) -> bool:
        rec = self._表.get(模块ID)
        if rec is None:
            return False
        rec.设备.在线 = False
        return True

    def 插上(self, 模块ID: str) -> bool:
        rec = self._表.get(模块ID)
        if rec is None:
            return False
        rec.设备.在线 = True
        rec.最后心跳 = self.t
        rec.离线 = False
        return True

    def _降级包(self, 模块ID: str, 包类型_: str, 维度: Optional[int] = None) -> MBP包:
        """安全降级值：零向量、置信=0、能量=0、序列号=-1。**不抛异常、不崩总线。**"""
        return MBP包(模块ID, 包类型_, self.t,
                     np.zeros(self.投影维度 if 维度 is None else int(维度)),
                     能量=0.0, 置信=0.0, 代谢=0.0, 序列号=-1)

    # ---------------------------------------------------------- B3 读（SENSE）
    def 读(self, 模块ID: str) -> MBP包:
        """`读(模块ID) -> MBP包`（SENSE）。三种时间特性有各自的取数行为（§5.5）。

        【M2·W2·2026-09-27】实时模式：读() 退化为"取最新快照"（seqlock 语义）——
        不触设备、不推进 EMA/上帧、事件改为显式队列消费（单消费者契约）、
        不增读侧序列号（G2 根治：同拍双消费者互偷不再可能）。
        默认（非实时）路径逐位保持旧行为。
        """
        if self.实时模式:
            return self._读快照(模块ID)
        rec = self._表.get(模块ID)
        if rec is None or self.是否离线(模块ID):
            return self._降级包(模块ID, "SENSE")
        d = rec.描述符
        if d.模块类 not in ("感知类", "双向类"):
            return self._降级包(模块ID, "SENSE", d.投影维度)

        conf = 1.0
        if d.时间特性 == "流式":                    # 每步更新
            x = rec.设备.感知(self.t)
            if x is None:
                载荷 = (np.zeros(d.投影维度) if rec.上帧 is None else rec.上帧)
            else:
                载荷 = rec.投影器(np.asarray(x, dtype=float).ravel())
                rec.上帧 = 载荷
        elif d.时间特性 == "快照":                  # 缓冲，N 步一帧
            rec.帧计数 += 1
            N = max(int(d.缓冲深度), 1)
            if rec.上帧 is None or rec.帧计数 >= N:
                x = rec.设备.感知(self.t)
                rec.上帧 = rec.投影器(np.asarray(x, dtype=float).ravel())
                rec.帧计数 = 0
            载荷 = rec.上帧
        else:                                       # 事件：队列，非阻塞
            if not rec.事件队列:
                x = rec.设备.感知(self.t)
                if x is not None:
                    rec.事件队列.append(rec.投影器(np.asarray(x, dtype=float).ravel()))
            if rec.事件队列:
                载荷 = rec.事件队列.pop(0)
            else:
                return self._降级包(模块ID, "SENSE", d.投影维度)   # 空事件：非阻塞降级
        rec.序列号 += 1
        return MBP包(模块ID, "SENSE", self.t, 载荷,
                     能量=float(np.linalg.norm(载荷)), 置信=conf,
                     代谢=d.代谢常数, 序列号=rec.序列号)

    # ------------------------------------------------------ 【M2·W2】实时模型：快照读/写者侧生产
    def _读快照(self, 模块ID: str) -> MBP包:
        """【M2·W2·2026-09-27】快照读（无副作用）：seqlock 取最新快照；事件类 = 显式出队消费。
        空快照/离线 ⇒ 降级定时包（置信=0、序列号=-1，复用降级约定）。"""
        rec = self._表.get(模块ID)
        if rec is None or self.是否离线(模块ID):
            return self._降级定时包(模块ID, "SENSE")
        d = rec.描述符
        if d.模块类 not in ("感知类", "双向类"):
            return self._降级定时包(模块ID, "SENSE", d.投影维度)
        if d.时间特性 == "事件":
            if rec.事件队列:
                return rec.事件队列.pop(0)
            return self._降级定时包(模块ID, "SENSE", d.投影维度)   # 空事件：非阻塞降级
        包 = rec.快照槽.读()
        if 包 is None:
            return self._降级定时包(模块ID, "SENSE", d.投影维度)
        return 包

    def 生产快照(self, 模块ID: str) -> bool:
        """【M2·W2·2026-09-27】写者侧生产：设备由**自己时钟域的节拍**推进（W2 语义）。
        由持有该模块的环（R1/world 域）在其自身 tick 内调用；读() 从此与设备推进解耦。
        流式每拍产帧；快照类按 缓冲深度 N 拍产一帧（帧控在写者侧）；事件类入有界事件队列。"""
        rec = self._表.get(模块ID)
        if rec is None or not rec.设备.在线 or rec.快照槽 is None:
            return False
        d = rec.描述符
        if d.模块类 not in ("感知类", "双向类"):
            return False
        t = self.t
        域 = self.实时配置["时钟域默认"]
        龄限 = float(self.实时配置["龄期上限秒"])
        延迟 = 延迟数值(d.延迟)          # 【M2·W3】字符串 → 数值入包
        单调 = time.perf_counter_ns() if self.实时模式 else int(t) * 1_000_000
        if d.时间特性 == "事件":
            x = rec.设备.感知(t)
            if x is not None:
                载荷 = rec.投影器(np.asarray(x, dtype=float).ravel())
                rec.序列号 += 1
                rec.事件队列.append(定时包(模块ID, "SENSE", t, 载荷,
                                           能量=float(np.linalg.norm(载荷)), 置信=1.0,
                                           代谢=d.代谢常数, 序列号=rec.序列号,
                                           时钟域=域, 单调时间戳=单调,
                                           龄期上限=龄限, 传输延迟=延迟))
            return True
        取帧 = True
        if d.时间特性 == "快照":        # 帧控移到写者侧（读侧零状态）
            rec.帧计数 += 1
            N = max(int(d.缓冲深度), 1)
            取帧 = (rec.上帧 is None) or (rec.帧计数 >= N)
            if 取帧:
                rec.帧计数 = 0
        if 取帧:
            x = rec.设备.感知(t)
            载荷 = rec.投影器(np.asarray(x, dtype=float).ravel())
            rec.上帧 = 载荷
            rec.序列号 += 1
            rec.快照槽.写(定时包(模块ID, "SENSE", t, 载荷,
                                 能量=float(np.linalg.norm(载荷)), 置信=1.0,
                                 代谢=d.代谢常数, 序列号=rec.序列号,
                                 时钟域=域, 单调时间戳=单调,
                                 龄期上限=龄限, 传输延迟=延迟))
        return True

    def _降级定时包(self, 模块ID: str, 包类型_: str, 维度: Optional[int] = None) -> 定时包:
        """【M2·W2】降级约定的定时包形态：九字段语义与 _降级包 逐位一致 + 定时冻结字段。"""
        p = self._降级包(模块ID, 包类型_, 维度)
        return 定时包(p.模块ID, p.包类型, p.时间戳, p.向量, p.能量, p.置信, p.代谢,
                      p.序列号, p.校验, 时钟域=self.实时配置["时钟域默认"],
                      单调时间戳=(time.perf_counter_ns() if self.实时模式
                                  else int(self.t) * 1_000_000),
                      龄期上限=float(self.实时配置["龄期上限秒"]))

    def 实时遥测(self) -> dict:
        """【M2·W4/W5】底座遥测：心跳龄期（拍）+ 有界缓冲水位/溢出计数（不静默纪律）。"""
        out = dict(实时模式=self.实时模式, 心跳龄期拍=self.巡检())
        缓冲 = {}
        for mid, rec in sorted(self._表.items()):
            for 名, obj in (("事件队列", rec.事件队列), ("探测记录", rec.探测记录)):
                if isinstance(obj, 环形缓冲):
                    缓冲["%s.%s" % (mid, 名)] = dict(长度=len(obj), 溢出=obj.溢出数)
            if rec.快照槽 is not None:
                缓冲["%s.快照槽" % mid] = dict(写版本=rec.快照槽.写版本,
                                               撕裂重试=rec.快照槽.撕裂重试)
        out["有界缓冲"] = 缓冲
        return out

    # ---------------------------------------------------------- B4 写（CMD → RESP）
    def 写(self, 模块ID: str, 指令, 学习: bool = True) -> MBP包:
        """`写(模块ID, 指令) -> MBP包`（发 CMD，收 RESP）。

        **在线响应模型**：用这组 CMD/RESP 对更新 `ŵ=f(a)`，记录**更新前**的预测误差
        （更新前误差才是"我预测得准不准"的诚实读数）。
        """
        rec = self._表.get(模块ID)
        if rec is None or self.是否离线(模块ID):
            return self._降级包(模块ID, "RESP")
        d = rec.描述符
        if d.模块类 not in ("执行类", "双向类"):
            return self._降级包(模块ID, "RESP", d.投影维度)

        a = (指令.向量 if isinstance(指令, MBP包) else
             np.asarray(指令, dtype=float).ravel())
        rec.序列号 += 1
        cmd = MBP包(模块ID, "CMD", self.t, a, 能量=float(np.linalg.norm(a)),
                    置信=1.0, 代谢=d.代谢常数, 序列号=rec.序列号)
        w = np.asarray(rec.设备.响应(a), dtype=float).ravel()
        e_pre = rec.响应模型.误差(a, w)
        if 学习:
            rec.响应模型.更新(a, w)
        rec.最后误差 = e_pre
        rec.探测记录.append((a.copy(), w.copy(), e_pre))
        rec.序列号 += 1
        return MBP包(模块ID, "RESP", self.t, w, 能量=float(np.linalg.norm(w)),
                     置信=1.0, 代谢=d.代谢常数, 序列号=rec.序列号)

    # ---------------------------------------------------------- 交换（双向类）
    def 交换(self, 包: MBP包) -> MBP包:
        """`交换(包) -> MBP包`（双向类）。"""
        return self.写(包.模块ID, 包.向量, 学习=True)

    # ---------------------------------------------------------- 计算类 前向
    def 前向(self, 输入, 模块ID: Optional[str] = None) -> np.ndarray:
        """`前向(输入) -> np.ndarray`（计算类）。未指定 `模块ID` 时取首个计算类模块。"""
        if 模块ID is None:
            for mid, rec in self._表.items():
                if rec.描述符.模块类 == "计算类":
                    模块ID = mid
                    break
        rec = self._表.get(模块ID)
        if rec is None or self.是否离线(模块ID) or rec.描述符.模块类 != "计算类":
            return np.zeros(self.投影维度)
        return np.asarray(rec.设备.前向(输入), dtype=float).ravel()

    # ---------------------------------------------------------- B6 自我探测
    def self_t(self, e: float) -> float:
        """`self_t = σ(−e/τ)`。e 越小 ⇒ 越像"自己的模块"。
        ★ e ≥ 0 ⇒ self_t ∈ (0, 0.5]；判据靠**排序**（AUC），不靠绝对阈值。"""
        return float(1.0 / (1.0 + np.exp(min(e / max(self.tau, 1e-12), 700.0))))

    def 探测(self, 模块ID: str, 指令, 学习: bool = True) -> dict:
        """一次 CMD/RESP 探测：返回预测误差 e 与自我探测值 self_t。"""
        rec = self._表.get(模块ID)
        resp = self.写(模块ID, 指令, 学习=学习)
        if rec is None:
            return dict(e=float("nan"), self_t=self.self_t(float("inf")), resp=resp, 离线=True)
        e = rec.最后误差
        return dict(e=float(e), self_t=self.self_t(float(e)), resp=resp,
                    离线=bool(rec.离线), 序列号=resp.序列号)

    # ---------------------------------------------------------- B7 标定
    def 标定(self, 模块ID: str, 标定向量=None) -> MBP包:
        """CAL 双向往返，标定参数入档，`标定状态: 未标定 → 已标定`。"""
        rec = self._表.get(模块ID)
        if rec is None or self.是否离线(模块ID):
            return self._降级包(模块ID, "CAL")
        d = rec.描述符
        if 标定向量 is None:
            v = np.zeros(d.原始维度)
        else:
            v = np.asarray(标定向量, dtype=float).ravel()[:d.原始维度]
        resp = np.asarray(rec.设备.标定(v), dtype=float).ravel()
        rec.标定参数 = {"请求": v.copy(), "响应": resp.copy(), "时间戳": int(self.t)}
        d.标定状态 = "已标定"
        rec.序列号 += 1
        return MBP包(模块ID, "CAL", self.t, resp, 能量=float(np.linalg.norm(resp)),
                     置信=1.0, 代谢=d.代谢常数, 序列号=rec.序列号)

    # ---------------------------------------------------------- §5.9 投影器在线学习
    def 在线学投影器(self, 模块ID: str, 轮数: int = 200, seed: int = 0, 测试尺度: float = 1.0,
                     探测数: int = 50) -> dict:
        """模块插上 → 内核无对应投影器 ⇒ 随机 φ → 发测试信号、观察响应、更新 φ。

        返回收敛前后的预测误差（对照：同一批测试信号上的 `‖y − φ(x)‖²`）。
        """
        rec = self._表.get(模块ID)
        if rec is None or self.是否离线(模块ID):
            return {}
        d = rec.描述符
        rng = np.random.default_rng(int(seed))
        误差前 = []
        for i in range(int(轮数)):
            x = rng.normal(size=d.原始维度) * 测试尺度
            y = np.asarray(rec.设备.探针响应(x), dtype=float).ravel()
            if i < 20:
                误差前.append(rec.投影器.误差(x, y))
            rec.投影器.更新(x, y)
        误差后 = []
        for _ in range(int(探测数)):
            x = rng.normal(size=d.原始维度) * 测试尺度
            y = np.asarray(rec.设备.探针响应(x), dtype=float).ravel()
            误差后.append(rec.投影器.误差(x, y))
        rec.投影器.已收敛 = True
        return dict(误差前=float(np.mean(误差前)), 误差后=float(np.mean(误差后)),
                    轮数=int(轮数), 指纹=rec.投影器.指纹())

    # ---------------------------------------------------------- B9 自我档案 / 身体图式
    def 自我档案(self) -> dict:
        """**持久身份**：不随身体（模块集合）改变。"""
        return {
            "身份锚点": self.身份锚点.copy(),
            "协议版本": self.协议版本,
            "内核投影器指纹": self._内核投影器.指纹(),
            "探测tau": float(self.tau),
            "投影维度": int(self.投影维度),
        }

    def 自我档案哈希(self) -> str:
        a = self.自我档案()
        s = "|".join([np.ascontiguousarray(a["身份锚点"], dtype=np.float64).tobytes().hex(),
                      a["协议版本"], a["内核投影器指纹"],
                      "%.6g" % a["探测tau"], str(a["投影维度"])])
        return _sha16(s.encode("utf-8"))

    def 身体图式(self) -> dict:
        """**每次插拔重建**：已注册模块集合 + 各模块投影器指纹 + 标定状态。"""
        out = {}
        for mid, rec in sorted(self._表.items()):
            d = rec.描述符
            out[mid] = {
                "模块类": d.模块类, "时间特性": d.时间特性,
                "原始维度": d.原始维度, "投影维度": d.投影维度,
                "投影器": d.投影器, "投影器指纹": rec.投影器.指纹(),
                "标定状态": d.标定状态, "描述符指纹": d.指纹(),
                "在线": bool(rec.设备.在线),
            }
        return out

    def 身体图式哈希(self) -> str:
        b = self.身体图式()
        s = "|".join("%s{%s}" % (k, ",".join("%s=%s" % (a, b2) for a, b2 in v.items()))
                     for k, v in sorted(b.items()))
        return _sha16(s.encode("utf-8"))

    def 投影器指纹表(self) -> dict:
        return {mid: rec.投影器.指纹() for mid, rec in sorted(self._表.items())}

    # --------------------------------------------- R4 纯追加（2026-09-26f）：投影器挂载
    def 挂投影器(self, 模块ID: str, 投影器) -> bool:
        """【R4 纯追加】把模块槽位的投影器替换为外部实现（如 ahi_theory.modality.慢特征投影器）。

        仅**新增**方法：B2 注册仍按 §5.9 自动建投影器（既有行为与签名逐位不变）；
        本方法供 §5.9「未知模态在线学投影器」的**状态化**实现（多尺度慢特征库）挂载之用——
        `mbp.通用投影器` 为无状态线性映射，状态化 φ 需要承载 EMA 内部状态。
        """
        rec = self._表.get(模块ID)
        if rec is None:
            return False
        rec.投影器 = 投影器
        return True


# ================================================================== 工具
def AUC(正样本: Sequence[float], 负样本: Sequence[float]) -> float:
    """Mann–Whitney U 的等价 AUC：正样本（应判为 self）分数 > 负样本（应判为 world）的概率。

    连续值无并列，用秩和公式；无随机性、可复现。
    """
    正 = np.asarray(正样本, dtype=float)
    负 = np.asarray(负样本, dtype=float)
    n, m = len(正), len(负)
    if n == 0 or m == 0:
        return float("nan")
    全部 = np.concatenate([正, 负])
    秩 = 全部.argsort(kind="stable").argsort(kind="stable") + 1.0
    R = float(秩[:n].sum())
    return float((R - n * (n + 1) / 2.0) / (n * m))


# ================================================================== M5 纯追加（2026-09-28 #65）
def 模块角色向量(描述符, D: int = 1024, 种子偏移: int = 0) -> np.ndarray:
    """【M5 纯追加（#65）】「模块ID→角色向量」只读映射（95_架构定稿 L6 行 359）。

    种子=模块描述符指纹（`模块描述符.指纹()`，可复算、可校验、与模块注册事件绑定）；
    D=1024 与 ahi_theory.hdc_core 同律（结构先验）。只读：不接触总线任何可变状态，
    不改包结构/九字段签名/载荷哈希；输入为 模块描述符 实例（或任何带 .指纹() 的对象）。
    hdc_core 身体图式投影 Φ_body 经此映射取得各已注册模块的角色向量（R_mod 族之外的
    模块特异通道）；同模块跨进程/跨时间映射逐位确定（指纹含协议版本与标定状态）。
    """
    fp = 描述符.指纹() if hasattr(描述符, "指纹") else str(描述符)
    seed = int(fp, 16) + int(种子偏移)
    rng = np.random.default_rng(seed)
    return np.sign(rng.standard_normal(D)).astype(np.float32)
