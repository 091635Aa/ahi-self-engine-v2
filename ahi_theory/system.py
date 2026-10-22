# -*- coding: utf-8 -*-
"""ahi_theory.system —— AHI 2.2 系统装配器（Task 1.1 / INT 规格 `add-ahi22-integration`）。

把 R1–R4 五层装成**一个可运行的 AHI 系统**（AI 自主智能体运行形态的骨架）：
    装配(配置) → 步进(u_ext) → 状态汇总() → 关闭()
装配内容 = Brain（一行不改）+ MBP 总线（含挂投影器/φw 在线学）+ World（R2 双向类执行器）
    + 闭环感知兼容（SelfProbeLoop 混合注册表路径）+ 模态源（文本流式挂 φ_txt + 事件变体）
    + 执行类模块（m1 常驻；可选"非我"对照 wm1=随机响应执行类）。

★ 通电纪律（逐条落实 `meeting10/INT_问题清单.md` §2 的隐患登记，装配器是这些纪律的**执行者**）：
  · G1 单时钟：总线时钟**只**由 `闭环.步进()` 推进（每拍恰好 +1）；φ_txt 学习**手动驱动**
    （`观测`/`重构`），**禁止**调用内含 `总线.步进(1)` 的 `在线学文本投影` ⇒ 无双时钟。
  · G2 单消费者：感知类（txt_0/evt_0）由闭环独占总线读；内核 u 只用 **wld 的总线 SENSE**
    （闭环不读 wld —— wld 是双向类，闭环仅 CMD 探测）与 txt/evt 的**设备原始帧 × 固定投影**
    （不经总线二次读）⇒ 无事件被偷/EMA 双推进。
  · G4 单写者：闭环是 wld 的唯一 CMD 发起者 ⇒ 世界每拍经探测恰好演化一步（"持续演化"由此
    保证），不叠加任何直接 `世界.步()` 写。
  · G7 独立随机源：Brain/总线/闭环/世界/文本源/事件源/内核投影 各自独立 Generator（本类
    只以不同整数种子派生，不共享 Generator 对象）。
  · G8 内核喂入：u_ext(D_REGION=24) 由**固定种子投影**（种子 900，跨 run 不变）从
    wld SENSE（世界维度）与 txt/evt 原始帧（64）映射而来，u 尺度 0.01（与通电探针同口径）。
  · G9/G11 独立 φ 实例：每模块独立挂投影器实例；事件型 evt_0 **不挂状态化 φ**（M1 事件口径）。
  · G12 检查点才调 `brain.state()`：`步进` 内不调（state() 消耗 brain.rng）；只在 `状态汇总` 调。
  · G13 显式驱动 φ 学习：每拍仅当取到新帧才 `观测`，到点 `重构`（无任何库内调用方）。
  · CMD 类宽度默认 d=8（判据覆盖域）；世界执行器宽度可配（默认 8；通电诊断 P1 的 d>8
    误归因已由 2026-09-26g selfprobe 修复收口，I1 场景以 d=16 实证）。
★ 归因口径（R3/W6/S2）：执行类线性响应=w=Ma+ξ ⇒ self；随机响应/外生世界源 ⇒ world；
  世界执行器（RESP=Ga+ξ，我方 CMD 可预测响应）⇒ **self**；事件型感知类在预注册参数下
  归因窗不满 ⇒ unknown（P2 口径内行为，如实登记，不修）。
"""
from __future__ import annotations

import zlib
from typing import Optional

import numpy as np

try:                     # 与 world.py/selfprobe.py 相同的导入容错
    from .brain import Brain, D_REGION
    from .mbp import MBP总线, MBP模块, 模块描述符
    from .world import 世界, 世界模块, 世界描述符
    from .modality import 文本源模块, 文本流描述符, 慢特征投影器, 闭环感知兼容
except ImportError:      # pragma: no cover
    from ahi_theory.brain import Brain, D_REGION
    from ahi_theory.mbp import MBP总线, MBP模块, 模块描述符
    from ahi_theory.world import 世界, 世界模块, 世界描述符
    from ahi_theory.modality import 文本源模块, 文本流描述符, 慢特征投影器, 闭环感知兼容


# ------------------------------------------------------------ 默认配置（全部显式登记，不静默取库默认）
默认配置 = dict(
    seed=101,               # 全局种子（各组件按 seed*1000+k 派生，通电探针同口径）
    总线维度=512,            # MBP 总线投影维度默认_
    心跳超时=3,
    执行维度=8,              # CMD 执行类宽度（判据覆盖域）
    世界维度=8,              # 世界执行器宽度（I1 场景以 16 运行 = P1 修复的装配内实证）
    非我对照=True,           # G4/W 组建议：注册表追加"非我"对照位 wm1（随机响应执行类）
    u尺度=0.01,              # 内核喂入尺度（G8）
    投影种子=900,            # 内核喂入固定投影（G8；跨 run 不变）
    φw在线学=True,           # wld 的 φw 在线学投影器（轮数 200，通电探针同口径）
    φ学习=True,              # 显式驱动 φ_txt 观测/重构（G13）
    预热=200,                # 世界自主预热步数（消除初态弛豫瞬态，S/M 场景同口径）
)


# ------------------------------------------------------------ 场景设备（装配器自带的默认设备类）
class 线性执行模块(MBP模块):
    """self 类真值设备：线性响应 w = M a + ξ（verify_all._s_场景 同款）。"""

    def __init__(self, 描述符, seed=0, 噪声=0.05):
        super().__init__(描述符)
        d = 描述符.投影维度
        self.rng = np.random.default_rng(int(seed))
        self.M = self.rng.normal(size=(d, d)) / np.sqrt(d)
        self.噪声 = float(噪声)

    def 响应(self, a):
        return self.M @ np.asarray(a, float).ravel() \
            + self.rng.normal(size=self.M.shape[0]) * self.噪声

    def 标定(self, v):
        return np.zeros(self.描述符.投影维度)


class 随机响应模块(MBP模块):
    """world 类真值设备（"非我"对照位）：响应与指令无关、纯随机，与 self 设备同尺度。"""

    def __init__(self, 描述符, seed=0):
        super().__init__(描述符)
        self.rng = np.random.default_rng(int(seed))

    def 响应(self, a):
        return self.rng.normal(size=self.描述符.投影维度) * float(np.sqrt(1.0 + 0.05 ** 2))

    def 标定(self, v):
        return np.zeros(self.描述符.投影维度)


def 执行类描述符(mid, 能力, d=8):
    return 模块描述符(mid, "执行类", "流式", d, d, "φ_act",
                      采样率="1kHz", 缓冲深度=1, 代谢常数=0.01, 延迟="1ms",
                      能力=tuple(能力), 协议版本="MBP-1.0")


def _z(x):
    """None → 零向量兜底（设备首帧前）。"""
    return np.zeros(1) if x is None else np.asarray(x, dtype=float).ravel()


# ================================================================== 装配器
class AHI系统:
    """AHI 2.2 系统装配器：一键把五层装成一个可步进的整体。

    用法：
        sys_ = AHI系统(dict(世界维度=16))     # 装配（= 装配(配置)）
        sys_.步进()                          # 每拍一次（u_ext 可选外注入，叠加在组合驱动上）
        快照 = sys_.状态汇总()                # 检查点调用（内部才调 brain.state()）
        末态 = sys_.关闭()                    # 结束运行，返回末态汇总
    生命周期（§5.10）：插入执行模块 / 拔出模块 / 插回模块 / 换装执行模块（经闭环流程）。
    """

    def __init__(self, 配置: Optional[dict] = None):
        cfg = dict(默认配置)
        cfg.update(配置 or {})
        self.配置 = cfg
        seed = int(cfg["seed"])
        ds = lambda k: seed * 1000 + k                    # 组件种子派生（通电探针同口径）

        # ---- R1 内核 + 总线 ----
        self.brain = Brain(seed=seed)                     # 内核：默认参数，一行不改
        self.bus = MBP总线(投影维度默认_=int(cfg["总线维度"]), tau=1.0,
                           心跳超时=int(cfg["心跳超时"]), seed=seed)

        # ---- R3/R4 闭环（混合注册表必须用兼容子类 —— G3）----
        self.闭环 = 闭环感知兼容(self.bus, 学习=True, seed=seed)

        # ---- R1 执行类：m1 常驻（self 真值）+ 可选"非我"对照 wm1（world 真值，G4 建议位）----
        d1 = 执行类描述符("m1", ("肩关节",), d=int(cfg["执行维度"]))
        self.闭环.插入模块(d1, 线性执行模块(d1, seed=ds(1)))
        if cfg["非我对照"]:
            dw1 = 执行类描述符("wm1", ("外部泵",), d=int(cfg["执行维度"]))
            self.闭环.插入模块(dw1, 随机响应模块(dw1, seed=ds(12)))

        # ---- R4 模态源：txt_0 流式（挂状态化 φ_txt）+ evt_0 事件变体（不挂状态化 φ，G11）----
        dt = 文本流描述符("txt_0", "流式", 1)
        self.设备txt = 文本源模块(dt, seed=ds(21))
        self.闭环.插入模块(dt, self.设备txt)
        self.φtxt = 慢特征投影器(64, seed=ds(22))
        self.挂txtok = bool(self.bus.挂投影器("txt_0", self.φtxt))     # 独立 φ 实例（G9）
        de = 文本流描述符("evt_0", "事件", 1)
        self.设备evt = 文本源模块(de, seed=ds(31), 事件模式=True)
        self.闭环.插入模块(de, self.设备evt)

        # ---- R2 世界（双向类执行器；预热入平稳；闭环独占写 —— G4）----
        self.world = 世界(维度=int(cfg["世界维度"]), 动作维度=int(cfg["世界维度"]),
                          噪声=0.02, 增益=0.15, 种子=ds(41))
        for _ in range(int(cfg["预热"])):
            self.world.步(None)
        self.描述符w = 世界描述符(int(cfg["世界维度"]), int(cfg["世界维度"]), "wld")
        self.闭环.插入模块(self.描述符w, 世界模块(self.描述符w, self.world, seed=ds(41)))
        self.φw学习 = {}
        if cfg["φw在线学"]:
            self.φw学习 = {k: (float(v) if isinstance(v, (int, float)) else v)
                           for k, v in self.bus.在线学投影器(
                               "wld", 轮数=200, seed=ds(42), 探测数=50).items()}

        # ---- 内核喂入固定投影（G8：种子 900，跨 run 不变）----
        rng_u = np.random.default_rng(int(cfg["投影种子"]))
        wd = int(cfg["世界维度"])
        self.Uw = rng_u.normal(size=(D_REGION, wd)) / 4.0
        self.Ut = rng_u.normal(size=(D_REGION, 64)) / 8.0
        self.Ue = rng_u.normal(size=(D_REGION, 64)) / 8.0

        # ---- 运行簿记 ----
        self.t拍 = 0
        self.evt_有效读 = 0
        self.喂包校验失败 = 0
        self._已关闭 = False

    # ------------------------------------------------ 生命周期（§5.10，经闭环流程）
    def 插入执行模块(self, mid: str, 能力, d: Optional[int] = None,
                     设备: Optional[MBP模块] = None) -> None:
        d = int(d or self.配置["执行维度"])
        描述符 = 执行类描述符(mid, 能力, d=d)
        设备 = 设备 or 线性执行模块(描述符, seed=self._设备种子(mid, 0))
        self.闭环.插入模块(描述符, 设备)

    def 换装执行模块(self, mid: str, 新能力, d: Optional[int] = None,
                     设备: Optional[MBP模块] = None) -> None:
        d = int(d or self.配置["执行维度"])
        新描述符 = 执行类描述符(mid, 新能力, d=d)
        设备 = 设备 or 线性执行模块(新描述符, seed=self._设备种子(mid, 500))
        self.闭环.换装模块(mid, 新描述符, 设备)

    def _设备种子(self, mid: str, 偏移: int = 0) -> int:
        """跨进程可复现的设备种子（不能用内置 hash()——字符串哈希按进程随机化）。"""
        return int(self.配置["seed"]) * 1000 + (zlib.crc32(mid.encode("utf-8")) % 977) + 偏移

    def 拔出模块(self, mid: str) -> None:
        self.闭环.拔出模块(mid)

    def 插回模块(self, mid: str) -> None:
        self.闭环.插回模块(mid)

    # ------------------------------------------------ 主循环（单时钟 + 单消费者 + 单写者）
    def 步进(self, u_ext: Optional[np.ndarray] = None) -> None:
        """推进一拍：闭环步进（总线时钟 +1、全模块探测/归因、世界演化一步）→ φ 学习 → 内核。"""
        if self._已关闭:
            raise RuntimeError("AHI系统 已关闭，不能再 步进")
        t0 = self.bus.t
        self.闭环.步进()                      # G1：唯一的总线时钟推进者
        if self.bus.t != t0 + 1:
            raise RuntimeError("单时钟破坏：bus.t %s → %s" % (t0, self.bus.t))
        t = self.bus.t
        # φ_txt 学习（G13 手动驱动；仅当本拍取到新帧）
        if self.配置["φ学习"]:
            if self.设备txt._上次ts == t and self.设备txt.最后帧 is not None:
                self.φtxt.观测(self.设备txt.最后帧)
            if t >= self.φtxt.首次重构 and t % self.φtxt.重构周期 == 0:
                self.φtxt.重构()
        # 内核喂入（G2 单消费者：wld=总线 SENSE；txt/evt=设备原始帧×固定投影）
        s = self.bus.读("wld")
        if not s.验证() or s.置信 != 1.0:
            self.喂包校验失败 += 1
        u = self.配置["u尺度"] * (self.Uw @ np.asarray(s.向量, dtype=float).ravel()
                                  + self.Ut @ _z(self.设备txt.最后帧)
                                  + self.Ue @ _z(self.设备evt.最后帧))
        if u_ext is not None:
            u = u + np.asarray(u_ext, dtype=float).ravel()
        self.brain.step(u)
        # 事件型有效读计数（探针口径）
        if self.设备evt._本步切换 and self.设备evt._上次ts == t:
            self.evt_有效读 += 1
        self.t拍 += 1

    # ------------------------------------------------ 检查点读出（G12：只有这里调 state()）
    def 状态汇总(self) -> dict:
        """五段汇总：内核状态摘要 + 总线模块表 + 归因表 + 世界状态 + 模态探针值。"""
        st = self.brain.state()
        模块表 = {}
        for mid in self.bus.已注册():
            d = self.bus.查询(mid)
            模块表[mid] = dict(模块类=d.模块类, 时间特性=d.时间特性,
                               原始维度=d.原始维度, 投影维度=d.投影维度,
                               在线=(not self.bus.是否离线(mid)),
                               φ指纹=self.bus.投影器指纹表().get(mid))
        归因 = self.闭环.归因表()
        世界态 = np.asarray(self.world.状态, dtype=float)
        return dict(
            内核=dict(t=int(self.brain.t), F=float(self.brain.F),
                      self_err=float(self.brain.self_err),
                      saturation=float(self.brain.saturation()),
                      m范数=float(np.linalg.norm(st["m"])),
                      h范数=float(np.linalg.norm(st["h"]))),
            总线模块表=模块表,
            归因=dict(归因表=归因, 迁移数=len(self.闭环.迁移日志),
                      复核触发数=len(self.闭环.复核日志), 挂起数=len(self.闭环.挂起日志),
                      异常隔离数=len(self.闭环.异常隔离)),
            世界=dict(t=int(self.world.t), 维度=int(self.配置["世界维度"]),
                      范数=float(np.linalg.norm(世界态)), 有限=bool(np.all(np.isfinite(世界态)))),
            模态=dict(φtxt指纹=self.φtxt.指纹(), φtxt已收敛=bool(self.φtxt.已收敛),
                      φtxt_n重构=int(self.φtxt.n重构), evt_有效读=self.evt_有效读,
                      txt末帧能量=float(np.sum(_z(self.设备txt.最后帧) ** 2))),
            运行=dict(拍=self.t拍, 时钟bus=int(self.bus.t), 喂包校验失败=self.喂包校验失败),
        )

    # ------------------------------------------------ 关闭
    def 关闭(self) -> dict:
        """结束运行：返回末态汇总（幂等）。"""
        末态 = self.状态汇总()
        末态["已关闭"] = True
        self._已关闭 = True
        return 末态
