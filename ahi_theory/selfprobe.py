# -*- coding: utf-8 -*-
"""ahi_theory.selfprobe —— R3 自我探测循环闭环（AHI 2.2）。

规格出处：`03_AHI/AHI2.1/完整` §5.7（在线响应模型）/ §5.8（自我探测=CMD/RESP 循环副产物）/
§5.10（热插拔/换身体流程）/ §6.4 W6；`.trae/specs/add-ahi22-self-probe-loop/spec.md`（S1–S5）。

★★ 本文件是**纯新增闭环层**：不 import `brain.py`，也不被 `brain.py` import（零锚点代价）；
   复用 `mbp.py` 的 MBP总线 / 在线响应模型 f / `self_t = σ(−e/τ)`（B4/B5/B6/B7/B9）与
   `world.py` 的最小世界（§6.2），只做"状态机 + 重归因 + 档案持久"的**循环化**，
   不新造探测原理。`mbp.py` 本文件**一行未改**。

★★ 定稿与可选启用（Task 1.1，2026-09-26e）：
   本模块为**可选启用**的纯新增闭环层——默认**不被任何既有验证/运行路径自动加载**：
   `ahi_theory/__init__.py` 为空、`brain.py` 不 import 本模块、`verify_all.py` 仅在
   **S 组（verify_S1…verify_S5）被显式调用时**才惰性 import 本模块（沿用 R2 世界层的惰性模式）
   ⇒ 既有 P1–P8 / N 组 / W 组判定路径零影响（零锚点代价实证：meeting8/R3_anchor_zero_cost.txt，
   锚点 3f753ad00d91285c / f69bdaf723bbaec1 命中、运行前后 h/m/h_slow/g/F 的 max|Δ|=0）。
   启用方式：① `run_verify.py --self`（S 组 S1–S5，纯追加段）；② 直接
   `from ahi_theory.selfprobe import 自探测循环, 外生世界模块`。

★ 归因分数（预注册，写定于任何 R3 数据之前）
  任务书 self_t = σ(−e/τ)。e 的绝对尺度跨模块不可比（不同模块响应能量差 3 个量级），
  故状态机用**每模块在线估计的 τ̂_m = 该模块 RESP 能量均值（‖w‖² 的历史均值，不含当前）**：
      ρ  = e / τ̂_m        （预测方差未解释率，量纲自由：0=完全可预测，1=完全不可预测）
      self_t* = σ(−ρ)      （与 self_t=σ(−e/τ) 同形，τ := τ̂_m）
  同时全程记录任务书原始口径 self_t_raw = bus.self_t(e)（τ=1）。
  结构依据（非数据拟合）：self 类模块学习后 ρ → 噪声本底/响应能量 ≈ 10⁻³；
  world 类模块（响应与指令无关）ρ ≈ 1。

★ 归属状态机（预注册）
  每模块 unknown → self / world：
    · 滑动窗 W=20 个 ρ 样本（窗满才判）；
    · 窗均值 ρ̄ ≤ 0.40（指令模型解释 ≥60% 响应方差）⇒ self 带内；
      窗均值 ρ̄ ≥ 0.75（解释 ≤25%）⇒ world 带内；
    · 连续 N=10 次同一带内 ⇒ 迁移并锁定（置信 = |ρ̄−0.575|/0.575 截断 [0,1]）。
  结构安全性（对任意 d=8 模块成立，非经验调参）：f 的欠定阶段恰 ma+1=9 次探测，
  此间窗均值可落入 world 带的**连续窗数至多 9（探测 20..28，探测 29 起窗口不含欠定样本）
  < N=10** ⇒ self 类模块不可能在 f 收敛前被误迁 world。

★★ P1 修复（2026-09-26g，INT 通电诊断 `meeting10/INT_问题清单.md` P1 的最小机制改动；
   判据阈值/比较式一字未改；参数写定于跑任何修复后数据之前）
  · 缺陷：上述结构性防误迁论证只对 d=8 成立（欠定段 ma+1=9 < 稳定步 10）。d=16 时欠定段
    =17 个恒等初始化伪样本（实测 ρ≈3.7–15，只会把 ρ̄ **向上**推），探测 20–29 的前 10 个窗
    判定全部含欠定样本 ⇒ ρ̄≥0.75 连击 10 次 ⇒ 在最小二乘收敛**之前**误锁 world 且永不复核
    （复现探针 `meeting10/INT_p1_probe_prefix.txt`：d16 seed101/202 均在探测 29 锁 world、
    ρ̄=1.59/1.49；d8 对照 36/38 收敛 self 与 R3 口径一致）。
  · 修法①（欠定期推迟判定，只门控 world 带连击）：世界带连击累计的前提 = 窗口全部样本
    来自响应模型脱离欠定之后（探测数 ≥ (ma+1)+窗长，**余量=窗长**，即滑动窗内不再含任何
    欠定伪样本）。**为何只门控 world 侧**（结构性，写定于看数据前）：伪样本 ρ≫1 只可能把
    ρ̄ 推**进 world 带**、不可能推进 self 带 ⇒ 含伪样本的窗落 self 带是**保守偏置**（更难
    而非更易）⇒ self 侧无需门控，这同时保证 d=8 既有行为逐位不变（S 组/M5 双集与 R3 已登记
    口径复跑对照见 `meeting10/INT_p1_regression_postfix.txt`）。ma = 描述符.投影维度 =
    响应模型指令维度（mbp._记录 构造口径）；伪样本恰为前 ma+1 拍（mbp.写 先记 e_pre 再 更新
    ⇒ 第 ma+1 拍预测仍用恒等 f）。**按探测数门控而非按响应模型 n更新**：sham（学习=False）
    的 n更新恒 0，若按 n更新 则 sham 永不判定 ⇒ 破坏 S2/S3 已登记 sham 读数；探测数口径对
    sham 只是延后迁移、末态不变。
  · 修法②（锁后一次性复核，**v2 双尺度反证**）：
    - v1（单尺度：连续 复核步=稳定步=10 个窗均值落**对侧带**即解锁）在 S 组回归（seed0–11）
      实测**不稳健**：外生世界源（world_exo）的 ρ≈1 但随世界慢模态起伏（ρ(A)=0.95 ⇒ 相关
      时标 ~20 步），窗 20 均值可**连续 11–21 个 ≤0.40**（慢能量谷）⇒ 虚假解锁：12 seed 中
      多个出现虚假二次迁移（S2 收敛步 359–1502）、seed1 误归 self（乱归）。证据（如实留档）：
      `meeting10/INT_p1_regression_postfix_v1_单尺度复核失效.txt/.json`。
    - v2 反证 = **双尺度同带**：短窗（窗长=20）均值与**复核长窗（复核长窗=100=5×窗长，须
      集满）**均值**同时**落对侧带、连续 复核步=10 拍 ⇒ 解锁回 unknown 重新归因一次（每段
      至多一次：插入/插回/换装复位时清零；解锁后由既有 unknown→锁定状态机自然重锁，若重锁
      回原侧不再复核 ⇒ 防抖动）。**依据（写定于 v2 回归之前，量来自 v1 失效证据）**：长窗
      100 ≈ 5×世界慢模态相关时标(20)，实测外生世界源 100 块均值 ≥0.58 恒在 world 带（短暂
      能量谷不构成强反证）；真 self 模块 ρ≈0.003 持续 ⇒ 双尺度同时满足、复核可达。
    - 复核事件登记于 `复核日志`（回归场景预期触发 0 次）。

★ 重归因流程（§5.10）
  插入：注册(DESC) → 标定(CAL) → 归因 unknown 起；
  拔出：设备.在线=False → 心跳超时（复用 MBP B5）→ 挂起（停止探测、保留事件流）；
  插回：恢复在线 → 状态回 unknown、清窗/连击（f 与 τ̂ 保留：同一设备响应模型仍有效）；
  换装（同 ID 不同能力）：拔掉+注销 → 重注册（f 全新）→ 标定 → 状态回 unknown、清全部证据
  ⇒ 重新标定 + 重新归因；自我档案（身份锚点/协议版本/内核投影器指纹/τ/投影维度，
  即 MBP B9 定义）全程不动。

★ 世界融合（spec SubTask 0.2）
  · R2 世界实例作为**执行器**挂载（双向类）：其 RESP = Ga+ξ 是"我方 CMD 的可预测响应"
    ⇒ self 侧证据（W2/W6 口径）；
  · `外生世界模块` 包装另一 R2 世界实例：每次被探测，该世界以**自主时钟**前进一步
    （动作=None），返回 SENSE 增量 Δx=(A−I)x+ξ —— 与探测指令无关（W1 独立性）
    ⇒ "无对应 CMD 的 SENSE 变化" ⇒ world 侧证据。

★ sham 对照（预注册；沿用 B6/W6 的"f 冻结"做法）
  · sham零：f 初始化="零" 且 学习=False ⇒ f≡0；
  · sham随机：前 10 次探测（学习=False）估 τ̂_m 后，冻结 W ~ N(0, σ²)，
    σ² = τ̂_m/(mw·(ma+1)) ⇒ E‖f(a)‖² = τ̂_m（**输出能量匹配的随机映射**）。
    （R2 W6 的固定 0.1 尺度在跨能量通道对比下会产生尺度伪影——低能量通道被随机偏置
     惩罚更重——故此处按每模块能量匹配，使 sham 与真值**同形同尺度**。）
  sham 全程 学习=False ⇒ 任何"收敛"都不来自学习。
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

try:                     # 与 world.py 相同的导入容错
    from .mbp import MBP总线, MBP模块, 模块描述符
except ImportError:      # pragma: no cover
    from ahi_theory.mbp import MBP总线, MBP模块, 模块描述符


# ------------------------------------------------------------ 预注册常量（勿按数据回调）
窗长 = 20          # 滑动窗 W
稳定步 = 10        # 稳定 N（> 9 = 欠定阶段最长伪带长 ⇒ 结构性防误迁）
阈self = 0.40      # ρ̄ ≤ 0.40：指令模型解释 ≥60% 响应方差 ⇒ self 带内
阈world = 0.75     # ρ̄ ≥ 0.75：解释 ≤25% ⇒ world 带内
带中点 = 0.575     # 两阈中点（置信归一用）
复核步 = 稳定步    # 锁后复核的反证连击长度（=稳定步 10 < 窗长，与窗参数联动；2026-09-26g P1 修法②，写定于看修复后数据之前）
复核长窗 = 100     # 锁后复核的长窗（=5×窗长，须集满才可判反证；覆盖世界慢模态能量起伏；2026-09-26g P1 修法② v2，依据 v1 失效证据写定）


def _σ(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(min(float(x), 700.0))))


# ================================================================== 外生世界源
class 外生世界模块(MBP模块):
    """world 侧对照源（§6.4 W6 / spec 0.2）：包装一个 R2 `世界` 实例。

    每次被探测，该世界以**自主时钟**前进一步（动作=None），返回 SENSE 增量
    Δx = x' − x = (A−I)x + ξ —— 与探测指令 a **无关**（W1 独立性）⇒ 纯外生事件，
    即"无对应 CMD 的 SENSE 变化"。
    """

    def __init__(self, 描述符: 模块描述符, 世界实例):
        super().__init__(描述符)
        self.世界 = 世界实例

    def 响应(self, 指令向量) -> np.ndarray:
        x0 = np.asarray(self.世界.状态, dtype=float).copy()
        x1 = np.asarray(self.世界.步(None), dtype=float)      # 自主步进：动作=None
        return x1 - x0

    def 标定(self, 标定向量) -> np.ndarray:
        return np.zeros(self.描述符.投影维度)


# ================================================================== 证据槽
class _证据:
    """单模块归属状态机的证据槽（含事件流）。"""

    __slots__ = ("窗", "能量历史", "self连击", "world连击", "状态", "置信",
                 "收敛步", "探测数", "起始t", "冻结已做", "总在线步", "事件流",
                 "挂起t", "反证连击", "已复核", "长窗")

    def __init__(self, t0: int):
        self.窗 = deque(maxlen=窗长)
        self.能量历史 = []          # ‖w‖² 历史（τ̂_m 的样本）
        self.self连击 = 0
        self.world连击 = 0
        self.状态 = "unknown"
        self.置信 = 0.0
        self.收敛步 = None          # 本段收敛时的探测数（段内相对）
        self.探测数 = 0
        self.起始t = int(t0)
        self.冻结已做 = False
        self.总在线步 = 0
        self.挂起t = None
        self.事件流 = []            # (t, e, rho, self_t*, self_t_raw, 状态)
        self.反证连击 = 0           # P1 修法②：锁后对侧带强反证的连击
        self.已复核 = False         # P1 修法②：本段是否已用过一次性复核
        self.长窗 = deque(maxlen=复核长窗)   # P1 修法② v2：锁定期的长窗 ρ（双尺度反证用）

    def 复位(self, t: int, 全清: bool) -> None:
        """重归因：状态回 unknown。全清=True（换装）连 τ̂/冻结标记一起清。"""
        self.窗.clear()
        self.self连击 = 0
        self.world连击 = 0
        self.状态 = "unknown"
        self.置信 = 0.0
        self.收敛步 = None
        self.探测数 = 0
        self.起始t = int(t)
        self.反证连击 = 0           # 新归因段：复核机会重置（每段至多一次）
        self.已复核 = False
        self.长窗.clear()
        if 全清:
            self.能量历史 = []
            self.冻结已做 = False


# ================================================================== 闭环层
class 自探测循环:
    """R3 闭环层：持续探测循环 + 归属状态机 + 重归因 + 档案/图式快照（复用 MBP B9）。"""

    def __init__(self, 总线: MBP总线, 学习: bool = True,
                 冻结方式: Optional[str] = None, seed: int = 0):
        self.bus = 总线
        self.学习 = bool(学习)
        self.冻结方式 = 冻结方式            # None | "零"（配合总线初始化） | "随机映射"
        self.rng指令 = np.random.default_rng(int(seed) * 13 + 5)   # 指令流（跨变体对齐）
        self.rng冻结 = np.random.default_rng(int(seed) * 31 + 777)
        self.证据: dict = {}
        self.异常隔离 = []                  # (t, mid, 摘要) —— 异常被隔离，不崩循环
        self.迁移日志 = []                  # (t, mid, label, 收敛探测数, rhō, 置信)
        self.复核日志 = []                  # (t, mid, 原状态, 探测数, rhō) —— P1 修法②解锁事件
        self.挂起日志 = []                  # (t, mid)
        self.插拔日志 = []                  # (t, 事件, mid)

    # ---------------- 场景编排接口（§5.10 流程） ----------------
    def 插入模块(self, 描述符: 模块描述符, 设备: MBP模块, 标定向量=None) -> None:
        """插入：广播 DESC → 注册 → 标定 → 归因 unknown 起。"""
        self.bus.注册(描述符, 设备)
        self.证据[描述符.模块ID] = _证据(self.bus.t)
        if 标定向量 is None:
            标定向量 = np.zeros(描述符.原始维度)
        self.bus.标定(描述符.模块ID, 标定向量)
        self.插拔日志.append((self.bus.t, "插入", 描述符.模块ID))

    def 拔出模块(self, mid: str) -> None:
        self.bus.拔掉(mid)
        self.插拔日志.append((self.bus.t, "拔出", mid))

    def 插回模块(self, mid: str) -> None:
        self.bus.插上(mid)      # 状态机在步进中检测"恢复在线"⇒ 重归因
        self.插拔日志.append((self.bus.t, "插回", mid))

    def 换装模块(self, mid: str, 新描述符: 模块描述符, 新设备: MBP模块,
                 标定向量=None) -> None:
        """同 ID 换能力不同的模块：拔掉 → 注销 → 重注册（f 全新）→ 标定 → 重新归因。"""
        self.bus.拔掉(mid)
        self.bus.注销(mid)
        self.bus.注册(新描述符, 新设备)
        self.证据[mid] = _证据(self.bus.t)      # 状态回 unknown、清全部证据（含 τ̂）
        if 标定向量 is None:
            标定向量 = np.zeros(新描述符.原始维度)
        self.bus.标定(mid, 标定向量)
        self.插拔日志.append((self.bus.t, "换装", mid))

    # ---------------- 主循环 ----------------
    def 步进(self) -> None:
        """推进一拍：总线时钟 +1，对每个在线 CMD 类模块做一次 CMD/RESP 探测并更新状态机。"""
        self.bus.步进(1)
        t = self.bus.t
        for mid in self.bus.已注册():
            ev = self.证据.get(mid)
            if ev is None:
                continue
            if self.bus.是否离线(mid):
                if ev.状态 != "suspended":
                    ev.状态 = "suspended"
                    ev.挂起t = t
                    self.挂起日志.append((t, mid))
                continue
            if ev.状态 == "suspended":
                # 恢复在线 ⇒ 重归因（§5.10：插回 ⇒ 重新探测归属；f/τ̂ 保留）
                ev.复位(t, 全清=False)
                self.插拔日志.append((t, "重归因(插回)", mid))
            ev.总在线步 += 1
            try:
                self._探测一次(mid, ev, t)
            except Exception as exc:  # noqa: BLE001 —— 异常隔离（§5.11），不崩循环
                self.异常隔离.append((t, mid, "%s: %s" % (type(exc).__name__, exc)))

    def _探测一次(self, mid: str, ev: _证据, t: int) -> None:
        d = self.bus.查询(mid)
        a = self.rng指令.normal(size=d.投影维度)
        r = self.bus.探测(mid, a, 学习=self.学习)
        w = r["resp"].向量
        e = float(r["e"])
        ev.探测数 += 1
        能量 = float(np.sum(np.asarray(w, dtype=float) ** 2))
        # τ̂_m：响应能量在线均值（历史不含当前 ⇒ 无自我读回；首拍用当前能量，无 nan）
        tau_m = float(np.mean(ev.能量历史)) if ev.能量历史 else 能量
        rho = e / max(tau_m, 1e-12)
        ev.能量历史.append(能量)
        self_t星 = _σ(rho)
        self_t_raw = self.bus.self_t(e)
        # sham随机：τ̂ 估计完成后一次性冻结为能量匹配随机映射（此后不再学习）
        if (self.冻结方式 == "随机映射" and (not ev.冻结已做) and ev.探测数 == 10):
            m = self.bus.取响应模型(mid)
            if m is not None:
                sigma2 = float(np.mean(ev.能量历史)) / (m.W.shape[0] * m.W.shape[1])
                m.W = self.rng冻结.normal(size=m.W.shape) * float(np.sqrt(max(sigma2, 1e-18)))
            ev.冻结已做 = True
        # 归属状态机（窗满才判）
        ev.窗.append(rho)
        if len(ev.窗) >= 窗长 and ev.状态 == "unknown":
            rhō = float(np.mean(ev.窗))
            # ★ P1 修法①（2026-09-26g）：欠定期推迟判定 —— 只门控 world 带连击的累计：
            #   窗口全部样本来自响应模型脱离欠定之后（探测数 ≥ (ma+1)+窗长，余量=窗长）才允许
            #   累计 world 连击。伪样本只抬高 ρ̄ ⇒ 不可能造伪 self 证据 ⇒ self 带分支不动
            #   （d=8 既有行为逐位不变）。详见文件头「P1 修复」登记块。
            窗含欠定 = ev.探测数 < (d.投影维度 + 1 + 窗长)
            if rhō <= 阈self:
                ev.self连击 += 1
                ev.world连击 = 0
            elif rhō >= 阈world and not 窗含欠定:
                ev.world连击 += 1
                ev.self连击 = 0
            else:
                ev.self连击 = 0
                ev.world连击 = 0
            if ev.self连击 >= 稳定步:
                ev.状态 = "self"
                ev.收敛步 = ev.探测数
            elif ev.world连击 >= 稳定步:
                ev.状态 = "world"
                ev.收敛步 = ev.探测数
            if ev.收敛步 is not None:
                self.迁移日志.append((t, mid, ev.状态, ev.收敛步,
                                      float(np.mean(ev.窗)),
                                      min(1.0, abs(float(np.mean(ev.窗)) - 带中点) / 带中点)))
        # ★ P1 修法②（2026-09-26g，v2 双尺度反证）：锁后一次性复核 —— 短窗与长窗（复核长窗=100，
        #   须集满）均值**同时**落对侧带、连续 复核步 拍（强反证）⇒ 解锁回 unknown 重新归因一次
        #   （每段至多一次；解锁后由上方 unknown 分支自然重锁）。v1 单尺度判据被外生世界源的
        #   慢能量谷击穿（虚假解锁/乱归），证据见文件头登记块与
        #   meeting10/INT_p1_regression_postfix_v1_单尺度复核失效.txt。
        if len(ev.窗) >= 窗长 and ev.状态 in ("self", "world") and not ev.已复核:
            rhō_锁 = float(np.mean(ev.窗))
            短对侧 = ((ev.状态 == "self" and rhō_锁 >= 阈world)
                      or (ev.状态 == "world" and rhō_锁 <= 阈self))
            ev.长窗.append(rho)
            长对侧 = False
            if len(ev.长窗) >= 复核长窗:
                rhō_长 = float(np.mean(ev.长窗))
                长对侧 = ((ev.状态 == "self" and rhō_长 >= 阈world)
                          or (ev.状态 == "world" and rhō_长 <= 阈self))
            if 短对侧 and 长对侧:
                ev.反证连击 += 1
            else:
                ev.反证连击 = 0
            if ev.反证连击 >= 复核步:
                self.复核日志.append((t, mid, ev.状态, ev.探测数, rhō_锁, rhō_长))
                ev.已复核 = True
                ev.状态 = "unknown"
                ev.置信 = 0.0
                ev.收敛步 = None
                ev.self连击 = ev.world连击 = ev.反证连击 = 0
                ev.长窗.clear()
        if ev.状态 in ("self", "world") and len(ev.窗):
            ev.置信 = min(1.0, abs(float(np.mean(ev.窗)) - 带中点) / 带中点)
        ev.事件流.append((t, e, rho, self_t星, self_t_raw, ev.状态))

    # ---------------- 读出 ----------------
    def 归因表(self) -> dict:
        return {mid: ev.状态 for mid, ev in sorted(self.证据.items())}

    def 事件流(self, mid: str):
        return self.证据[mid].事件流

    def 完整性(self) -> dict:
        """每模块：在线步数 vs 事件流长度（无异常时应逐位相等）。"""
        return {mid: dict(在线步=ev.总在线步, 事件=len(ev.事件流))
                for mid, ev in sorted(self.证据.items())}
