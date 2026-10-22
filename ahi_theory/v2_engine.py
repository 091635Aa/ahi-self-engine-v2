# -*- coding: utf-8 -*-
"""ahi_theory/v2_engine.py —— M15.5 · V2 引擎（六层架构 V2 五模块合龙：装配器+运行时，影子模式）

预登记：ahi_theory/verify_all.py `_CHANGELOG` #75（2026-10-01『M15.5 V2 引擎联动预登记』，先于改码）。
规格出处：meeting11/95_架构定稿.md（六层机制 §2 行 129–304 + §3.1 K 组判据式 行 377–379
  + 行 429–441 N1 替代判定式）+ meeting14/00_人格专项会议纪要.md §6 诚实边界
  + meeting15/00_自我思考会议纪要.md §5#6/§7（"引擎级真值待 V2 引擎供数"交付条款——
  本模块即该供数面的首轮通电实现）。

影子模式纪律（红线）：
  · 纯新增装配器；默认不加载（无模块级副作用——消费方显式 import 才实例化）；
  · 不 import brain/world/system/three_loop/mbp（影子纪律）；只组合五个既有影子模块
    （hdc_core/g_layer/ecae/personality/selfthink——全部构造注入级引用，五模块本体零改动）
    + numpy + ahi_theory.ledger；
  · 既有模块一行不改：全部按公开接口调用——g_layer（喂步/regime_ok/consolidation_permit/
    注入kill/recovery_ok/scc_state/post_replay_quiet/e_c史/控制环写入数）、ecae（喂步/压力/
    请求离线相/进入离线相/离线步/结束离线相/get_四通道/压力/离线占比/C范数/列印迹）、
    hdc_core（步进R1/维护R2/导出占据链）、personality（观测R1/维护R2/步进推进/物化窗口/窗历史）、
    selfthink（喂R1/触发判定/自省episode/e滚动中位/滚动高分位/叙事/命中率）；
  · 零拟合红线：全文件无 lstsq / linalg.inv·pinv / linalg.solve / scipy.optimize / 参数搜索构造
    （C4 tokenize 级可检，V2_selftest [1] 自证）；装配常量全部为结构常量（下采样步长/节拍比/
    复测间隔），无任何数据驱动标定；
  · 账本纪律：装配口径以 kind="结构事件" 入账本（引擎不新增任何先验自由度——全部机制常数
    归属五个既有模块的预注册清单）。

装配接线（五模块合龙）：
  ecae(M7).喂步(pe=e_t, z=x_{t-1} 摘要, a=a_{t-1} 动作向量, z_next=x_t)——z 摘要=hdc x 下采样；
  hdc(M5).维护R2(consolidation_permit←g_layer.consolidation_permit())——巩固前置门；
  g_layer(M6).ACh ← ecae.get_四通道()['ACh']——逐拍真值化赋值（同 pe 流下与 g 内部 EMA 逐位
    相等：同 α=0.001 同递推式；赋值=接线语义的显式承载）；
  selfthink(M15).喂R1(e_t, a星序号, v_t=DA−5HT, G差=top1−top2)——全部=hdc/ecae 返回量；
  personality(M14).观测R1(e_t, a星序号)/维护R2(k星, tag=v_t)——hdc 返回量+ecae 价值标量；
  ecae.治理层=g_layer 实例（起飞门/post_replay_quiet 联动——ecae 构造注入）。

时钟轴（影子装配口径【候选】，O8 换算率开放）：
  引擎基轴=通电 R1 拍；1 维护拍=20 R1 拍（personality.维护节拍 同源）；ecae 拍轴=g_layer 步轴
  =通电 R1 拍（每 R1 拍各喂一次）；离线相期间 hdc 停步进/selfthink 停触发/personality 停喂
  （五模块冻结语义；g_layer 不在冻结清单照常观测——每离线步喂 offline_flag=True 供情境双源），
  ecae 只走 离线步 状态机。

通路：
  通电步进R1(u_t, 帧新鲜=True)：hdc.步进R1(u,a_prev)→ecae.喂步(pe 摘要)→g_layer.喂步+ACh 真值化
    →selfthink.喂R1→personality.观测R1→全模块快照 dict（次序按消费关系：ecae 的 z 摘要/pe
    消费 hdc 返回量，故 hdc 先行）。
  维护R2(q)：regime 供数复测→g_layer.consolidation_permit 判定→hdc.维护R2(q,e_last,a星,0,
    价值标量=v_t,permit)→人格.步进推进+selfthink 触发判定（触发时 自省episode）→
    人格.维护R2+物化窗口→全快照（末尾自动 离线相调度[可关]）。
  离线相调度()：S>S_crit ∧ g_layer 起飞门放行 ⇒ 进入离线相（ecae.离线相步循环 R_r 切片；
    NE 突变中断[离线pe供给]；醒后 post_replay_quiet 经 ecae.结束离线相→治理层.记录离线相结束
    联动）。#92 G0f-2：每离线相末（离线步循环退出后、状态回 wake 前）调用 hdc.离线压实()
    一次（睡眠依赖记忆巩固；WCET 新键 离线压实ns、摘要新增 IM压实合并数——纯遥测新增键）。
  kill注入()/恢复判定()/scc供数()：K3⊕K4 引擎级供数面（G.注入kill/recovery_ok 两段式）。
  导出()：占据链 k*(t)（hdc）、活动量 e_t 序列（R1 轴）、S_t 低频差分（‖ΔS‖/√D 维护轴）、
    四通道序列、Π 窗史、自省事件史、β 信号史、离线相/选择史、WCET 计量表（每通路
    perf_counter_ns）。

供数面注入（全部 None=内生缺省；供数面=95_架构定稿 治理层"供数接口"语义的装配承载，
夹具臂使用时须诚实标注——G6/E7 同律）：
  regime供数：callable()->序列，供 g.regime_ok 复测（None⇒引擎内生活动量 e 窗）；
  活动量供给：callable()->float，供 g.喂步 x_act（None⇒e_t 内生活动量）；
  体制量供给：callable()->float，供 g.喂步 x_regime（None⇒复用 x_act 单流模式，g_layer 原语义）；
  离线pe供给：callable(离线步序)->Optional[float]，供 ecae.离线步 NE（None⇒离线相 NE 不更新）。

诚实边界（影子期首测口径）：本模块是五模块的通电装配与供数面，不新增任何机制与自由度；
  K1–K4 引擎级测量为影子期首测（合成世界流驱动，正式双集判据接入另行预注册）；离线占比带
  调节、run0 分位带校准、照顾者域 M9/R_care 子域、K3′ 单带 sham、O1/O2/O3/O6/O8/O10
  开放项不虚设；已注册模块数=0（影子期无身体域注册表，Φ_body 零向量——honest 缺省）；
  压平标记窗内恒 False（O4 待 E3 补签，未实装）。
"""
from __future__ import annotations

import math
import time
from array import array
from typing import Callable, Optional

import numpy as np

from ahi_theory.hdc_core import D_VSA, MASTER_SEED, HDC核心
from ahi_theory.g_layer import G层核心
from ahi_theory.ecae import ECAE核心, ECAE种子
from ahi_theory.personality import PERS_SEED, 人格剖面器
from ahi_theory.selfthink import SELF_SEED, 自省核心
from ahi_theory.ledger import 环形账本
from ahi_theory.readout_head import 预测算头

__all__ = ["V2引擎", "R1拍每维护", "regime复测间隔维护拍"]

#: 1 维护拍 = 20 R1 拍（personality.维护节拍 同源；结构常量）
R1拍每维护 = 20
#: g.regime_ok 供数复测间隔（维护拍；结构常量【候选】——装配口径，变更=结构事件）
regime复测间隔维护拍 = 256


class V2引擎:
    """V2 引擎（五模块合龙装配器+运行时）：通电步进R1 / 维护R2 / 离线相调度 / kill 注入 / 导出。

    构造注意：首次构造 G层核心 会经 k_battery.sigma_target 触发 A11 偏差标定（一次性，
    模块级缓存，约 3–6s 纯 CPU；同进程后续构造零开销——g_layer 文档原文）。
    """

    def __init__(self, 输入维度: int = 16, 动作维度: int = 4, q维度: int = 8,
                 seed: int = MASTER_SEED, 账本=None, z维度: int = 128,
                 控制环: bool = True):
        self.账本 = 账本 if 账本 is not None else 环形账本()
        # ── 五模块实例化（构造序=依赖序；种子域预注册互不相交：hdc=1024/ECAE=2048/
        #    personality=2048/selfthink=3072——与各模块预注册一致，本模块零新种子域）──
        self.hdc = HDC核心(输入维度=int(输入维度), 动作维度=int(动作维度), q维度=int(q维度),
                           账本=self.账本, 种子=seed)
        self.G = G层核心(账本=self.账本, 控制环=控制环)
        self.E = ECAE核心(z维度=int(z维度), 账本=self.账本, 治理层=self.G, 种子=ECAE种子)
        self.人格 = 人格剖面器(账本=self.账本, 种子=PERS_SEED)
        self.selfthink = 自省核心(self.hdc, self.人格, 账本=self.账本, 种子=SELF_SEED)
        assert D_VSA % int(z维度) == 0, "z维度 须整除 D=1024（下采样结构常量）"
        self._z步 = D_VSA // int(z维度)
        # ── 运行时状态 ──
        self.状态 = "wake"                       # wake / offline（离线相五模块冻结）
        self.R1拍数 = 0
        self.维护拍数 = 0
        self.自动离线调度 = True                 # 维护R2 末尾自动 离线相调度（K 观测窗互斥场景可关）
        self.自动自省 = True
        self.离线pe供给: Optional[Callable] = None
        self.regime供数: Optional[Callable] = None
        self.活动量供给: Optional[Callable] = None
        self.体制量供给: Optional[Callable] = None
        self._a_prev = np.zeros(int(动作维度), dtype=np.float32)
        self._x_prev: Optional[np.ndarray] = None
        self._pe_last: Optional[float] = None
        self._e_last: Optional[float] = None
        self._v_last = 0.0
        self._NE突变 = False
        self._S_prev: Optional[np.ndarray] = None
        self._上次regime维护 = -10 ** 9
        self._最近g拍: Optional[dict] = None
        self.离线相史: list = []                 # 每离线相摘要（步数/完成度/中断/S销账后）
        self.离线选择史 = array("q")             # 每离线步重放槽位（-1=无选择）
        # ── 遥测（K1–K4 引擎级供数面）──
        self.e序列 = array("d")                  # R1 轴活动量 e_t（stale=NaN）
        self.k星链: list = []                    # R2 维护轴 k*(t)（镜像 hdc.导出占据链）
        self.ΔS序列 = array("d")                 # S_t 低频差分 ‖S_t−S_{t-1}‖/√D（维护轴）
        self._chDA = array("d")
        self._chNE = array("d")
        self._chACh = array("d")
        self._ch5HT = array("d")
        self.β史 = array("d")
        self.自省事件: list = []
        self.WCET = dict(R1拍ns=array("q"), R2维护ns=array("q"), 离线步ns=array("q"),
                         自省ns=array("q"), regime复测ns=array("q"),
                         离线压实ns=array("q"))              # #92 G0f-2：离线相末压实计量（新键，纯遥测）
        # ── #127/#128 C*′ 私有熵机（独立 rng 实例——不插入既有种子派生序列，缺省零消费逐位不变）
        #    #128 修订：私有熵=2-bit 量化上下文 c_t∈{0..3}（每拍 fresh iid）——S_c 进 world 转移核、
        #    P_c 供 M 键绑定；上下文内键相似性保持 ⇒ M 可泛化（全 fresh 码致键全正交的教训见 #128）
        self.私有熵开关 = False
        self._rng_priv = np.random.default_rng(int(seed) + 90001)   # 私有种子域（预注册偏移）
        self._S表 = self._rng_priv.standard_normal((4, int(输入维度))).astype(np.float32)
        self._P表 = np.sign(self._rng_priv.standard_normal((4, D_VSA))).astype(np.float32)
        self._P表[self._P表 == 0] = 1.0
        self._Φ_q8 = (self._rng_priv.standard_normal((int(q维度), D_VSA))
                      / np.sqrt(D_VSA)).astype(np.float32)       # q8=Φ_q8·r（q 内源化）
        self._私有缓存: Optional[dict] = None
        # ── #137 读出头转正（V2 引擎持有实例；缺省关=零副作用逐位不变）──
        self.读出头开关 = False
        self.读出头 = 预测算头(种子=int(seed))
        self._头f_prev: Optional[np.ndarray] = None
        self._û_last: Optional[np.ndarray] = None
        self._u窗 = np.zeros((8, int(输入维度)), dtype=np.float64)
        self._最近私有: Optional[dict] = None
        self._装配口径入账()

    # ── 装配口径入账（结构事件；引擎零新先验自由度） ──────────────────────────
    def _装配口径入账(self) -> None:
        if self.账本 is None:
            return
        self.账本.登记("结构事件",
                       payload={"事件": "V2 装配口径登记",
                                "口径": {"z维度": int(self.E.z维度),
                                         "z摘要下采样步": self._z步,
                                         "R1拍每维护": R1拍每维护,
                                         "regime复测间隔维护拍": regime复测间隔维护拍,
                                         "g步轴与ecae拍轴": "通电 R1 拍（影子装配口径【候选】，O8 开放）",
                                         "已注册模块数": 0,
                                         "压平标记窗内": False}},
                       理由="M15.5 #75 预登记：V2 引擎装配常量（结构常量非先验自由度——"
                            "全部机制常数归属五模块既有预注册清单；下采样步长/节拍比/复测间隔"
                            "为装配口径，变更=结构事件；零拟合：无任何数据驱动标定）")

    # ── 内部辅助 ─────────────────────────────────────────────────────────────
    def _下采样(self, v: Optional[np.ndarray]) -> np.ndarray:
        """z 摘要装配下采样 x[::步]（结构常量；None⇒零向量）。"""
        if v is None:
            return np.zeros(self.E.z维度, dtype=np.float32)
        return np.asarray(v, dtype=np.float32).ravel()[::self._z步].copy()

    def _活动量窗(self) -> Optional[np.ndarray]:
        """引擎内生活动量窗（regime 供数缺省源=活动量 e 序列末 ≤65536 拍；<4096⇒None 不测量）。"""
        n = len(self.e序列)
        if n < 4096:
            return None
        取 = min(65536, n)
        return np.frombuffer(self.e序列, dtype=np.float64,
                             count=取, offset=(n - 取) * 8).copy()

    # ── #127/#128 C*′ 私有熵生成（每 R1 拍一次；私有流不落盘、日志只出哈希） ─────────────
    def 生成私有(self) -> dict:
        """生成当拍私有熵：c（2-bit 上下文，fresh iid）、s16=该上下文的 world 耦合模式、
        q8（q 内源化投影）。私有熵开关=False ⇒ 恒等（s16=0、q8=0、c=0）⇒ 消费方逐位不变。
        键绑定用 P_c[上一拍 c]——由 私有P(c) 取用。"""
        if not self.私有熵开关:
            return dict(c=0, s16=np.zeros(self.hdc.动作维度, dtype=np.float32),
                        q8=np.zeros(8, dtype=np.float32))
        c = int(self._rng_priv.integers(0, 4))
        return dict(c=c, s16=self._S表[c].copy(),
                    q8=(self._Φ_q8 @ np.sign(self._rng_priv.standard_normal(D_VSA))
                        ).astype(np.float32))

    def 私有P(self, c: int) -> np.ndarray:
        """上下文 c 的键绑定码（双极 1024；缺省关闭时调用方不取用）。"""
        return self._P表[int(c) % 4]

    # ── #137 读出头对外接口（runner/仪表读取；不影响引擎内部时序） ──────────────
    def 最近私有(self):
        """当拍私有熵三元组（开关 on 时由 通电步进R1 内部生成并缓存）。"""
        return self._最近私有

    def û快照(self):
        """当拍读出头预测 û_t（开关 on 且已有 f_prev 时非 None）。"""
        return self._û_last

    # ── 通电步进R1（单 R1 拍通路） ────────────────────────────────────────────
    def 通电步进R1(self, u_t, a_prev=None, 帧新鲜: bool = True, 内生偏置=None,
                   意图先验=None, 意图强度: float = 1.0, efe_mode: bool = False,
                   学习门: float = None, 私有码: float = None,
                   私有上下文: int = None) -> dict:
        """单 R1 拍通路：hdc.步进R1(u,a_prev)→ecae.喂步(pe 摘要)→g_layer.喂步+ACh 真值化
        →selfthink.喂R1→personality.观测R1→全模块快照 dict。
        a_prev=执行动作供数：None⇒上一拍 a星（闭环缺省）；显式向量⇒外供动作
        （M5_selftest 模式簇喂流同律——影子期世界代理流的操作通道，M5 检1 实证：
        结构化外动作⇒键稳定⇒M 学习⇒e 下降；零动作⇒零键恒 e=1 死锁，禁用）。
        内生偏置=#105/#106 M2 可选透传（None⇒逐位不变；语义见 meeting17/m2_intrinsic.py）。
        离线相期间拒收（五模块冻结语义：hdc 停步进/selfthink 停触发/personality 停喂）。"""
        if self.状态 == "offline":
            return dict(离线=True, 冻结=True, R1拍数=self.R1拍数,
                        理由="离线相五模块冻结语义（hdc 停步进/selfthink 停触发/personality 停喂）")
        t0 = time.perf_counter_ns()
        # (0) #137 读出头（开关 on 时）：预测 û_t（f_prev 信息截止 t−1）→ 内部生成私有熵
        if self.读出头开关:
            if self._头f_prev is not None:
                self._û_last = self.读出头.预测(self._头f_prev)
            self._最近私有 = self.生成私有()
        self.R1拍数 += 1
        u = np.asarray(u_t, dtype=np.float32).ravel()
        a_供数 = (self._a_prev if a_prev is None
                  else np.asarray(a_prev, dtype=np.float32).ravel())
        # (1) M5 · hdc 步进R1（steps 1–6）
        if 私有上下文 is not None:
            self.hdc.设M上下文(私有上下文)                 # #132 C*‴：M 实例按上下文切换
        rec = self.hdc.步进R1(u, a_供数, 帧新鲜=帧新鲜, 内生偏置=内生偏置,
                              意图先验=意图先验, 意图强度=意图强度, efe_mode=efe_mode,
                              学习门=学习门, 私有码=私有码)
        e_t = rec["e"]
        x_t = rec["x"]
        # (1.5) #137 读出头：观察 u_t 后 NLMS 学习 + 构建 f_t（u窗/a星/s16_t——信息截止 t）
        if self.读出头开关:
            if self._头f_prev is not None:
                self.读出头.学习(self._头f_prev, u)
            self._u窗 = np.roll(self._u窗, -1, axis=0)
            self._u窗[-1] = u
            私 = self._最近私有
            self._头f_prev = self.读出头.构建特征(
                self._u窗, rec["a星"], 私["s16"] if 私 is not None else None)
        if 帧新鲜 and e_t is not None:
            self._pe_last = float(e_t)
        pe = float(e_t) if e_t is not None else (self._pe_last if self._pe_last is not None else 0.0)
        # (2) M7 · ecae 喂步（z 摘要=hdc x 下采样；拍轴=通电 R1 拍——装配口径【候选】）
        z_prev = self._下采样(self._x_prev)
        z_cur = self._下采样(x_t) if x_t is not None else None
        a_vec = self._下采样(np.sign(self.hdc.Φ_act @ a_供数).astype(np.float32))
        e_rec = self.E.喂步(pe=pe, z=z_prev, a=a_vec, z_next=z_cur, 帧新鲜=帧新鲜)
        self._v_last = float(e_rec["v"])
        self._NE突变 = bool(e_rec["NE突变"])
        ch = dict(DA=float(e_rec["DA"]), NE=float(e_rec["NE"]),
                  ACh=float(e_rec["ACh"]), **{"5HT": float(e_rec["5HT"])})
        # (3) M6 · g_layer 喂步（步轴=通电 R1 拍——装配口径【候选】）+ ACh 真值化
        x_act = (float(self.活动量供给()) if self.活动量供给 is not None
                 else (float(e_t) if e_t is not None else None))
        x_reg = (float(self.体制量供给()) if self.体制量供给 is not None else None)
        obs = self.G.喂步(pe=pe, offline_flag=False, x_act=x_act, x_regime=x_reg, 帧新鲜=帧新鲜)
        if obs is not None:
            self._最近g拍 = obs
        self.G.ACh = ch["ACh"]                       # 真值化（同 pe 流下与 g 内部 EMA 逐位相等）
        # (4) M15 · selfthink 喂R1（e_t/v_t/G差=hdc/ecae 返回量）
        G表 = rec["G表"]
        if len(G表) >= 2:
            g序 = sorted(G表)
            G差 = float(g序[1] - g序[0])
        else:
            G差 = 0.0
        self.selfthink.喂R1(e_t, rec["a星序号"], v_t=self._v_last, G差=G差)
        # (5) M14 · personality 观测R1
        self.人格.观测R1(e_t, rec["a星序号"], 帧新鲜=帧新鲜)
        # (6) 状态推进 + 遥测
        self._a_prev = np.asarray(rec["a星"], dtype=np.float32)
        if x_t is not None:
            self._x_prev = x_t
        self._e_last = e_t if e_t is not None else self._e_last
        self.e序列.append(float("nan") if e_t is None else float(e_t))
        self._chDA.append(ch["DA"])
        self._chNE.append(ch["NE"])
        self._chACh.append(ch["ACh"])
        self._ch5HT.append(ch["5HT"])
        self.β史.append(float(self.G.β_slow))
        ns = time.perf_counter_ns() - t0
        self.WCET["R1拍ns"].append(ns)
        return dict(拍=self.R1拍数, e=e_t, a星序号=rec["a星序号"], a星=rec["a星"],
                    候选ê=rec["ê表"], 候选G=rec["G表"], 意图cos表=rec["意图cos表"],
                    候选u_a=rec["u_a表"], EFE表=rec["EFE表"],
                    v_t=self._v_last, 四通道=ch, β=float(self.G.β_slow),
                    β状态=self.G.β状态(), 状态=self.状态,
                    k星=(int(self.hdc.占据链[-1]) if self.hdc.占据链 else -1),
                    WCETms=ns / 1e6,
                    模块=dict(hdc=rec, ecae=e_rec, g_layer=self._最近g拍,
                              personality=dict(窗数=len(self.人格.窗历史), 阶段=self.人格.stage),
                              selfthink=dict(拍=self.selfthink.拍数, 阶段=self.selfthink.stage)))

    # ── 维护R2（单维护拍通路；每 R1拍每维护 拍一次） ──────────────────────────
    def 维护R2(self, q, 节律违反度: float = 0.0, sat: float = 1.0) -> dict:
        """单维护拍通路：regime 供数复测→consolidation_permit 判定→hdc.维护R2（permit 前置门）
        →人格.步进推进+selfthink 触发判定（触发时 自省episode）→人格.维护R2+物化窗口→全快照。
        末尾自动 离线相调度（自动离线调度=True 时；离线相步独立计量，不占 R2 预算）。"""
        if self.状态 == "offline":
            return dict(离线=True, 冻结=True, 维护拍数=self.维护拍数)
        t0 = time.perf_counter_ns()
        self.维护拍数 += 1
        # (1) regime 供数复测（供数面：regime供数 callable 或 引擎内生活动量窗；
        #     独立 WCET 通道计量——K1 仪器全窗测量的成本不计入 R2 维护预算口径）
        if self.维护拍数 - self._上次regime维护 >= regime复测间隔维护拍:
            self._上次regime维护 = self.维护拍数
            tr0 = time.perf_counter_ns()
            x_reg = (self.regime供数() if self.regime供数 is not None else self._活动量窗())
            if x_reg is not None and len(x_reg) >= 4096:
                self.G.regime_ok(np.asarray(x_reg, dtype=np.float64), ctx="wake")
            self.WCET["regime复测ns"].append(time.perf_counter_ns() - tr0)
        # (2) M6 · consolidation_permit 判定（ACh 真值化已逐拍维持）
        permit = self.G.consolidation_permit()
        压 = self.E.压力()
        # (3) M5 · hdc 维护R2（steps 7–9；permit=g_layer 供数；已注册模块数=0 影子缺省）
        qv = np.asarray(q, dtype=np.float32).ravel()
        r2 = self.hdc.维护R2(q=qv, e_last=self._e_last, a星=self._a_prev,
                             已注册模块数=0, 价值标量=self._v_last,
                             consolidation_permit=bool(permit["permit"]))
        k星 = int(r2["k星"])
        self.k星链.append(k星)
        if self._S_prev is not None:
            self.ΔS序列.append(float(np.linalg.norm(
                r2["S"].astype(np.float64) - self._S_prev)) / math.sqrt(D_VSA))
        else:
            self.ΔS序列.append(0.0)
        self._S_prev = r2["S"].astype(np.float64)
        # (4) M14 · 步进推进 + M15 · 触发判定（触发时 自省episode）
        self.人格.步进推进()
        episode = None
        if self.自动自省:
            分型 = self.selfthink.触发判定(
                e_滚动中位=self.selfthink.e滚动中位(),
                e_高分位=self.selfthink.滚动高分位(),
                S_警戒=bool(压["可辨识"] and 压["S"] > 压["S_crit"]),
                viability_越界=bool(self.E.注意捕获),
                NE_突变=self._NE突变,
                队列空=True, 无未决R1=True,
                非离线=(self.状态 == "wake"),
                consolidation_permit=bool(permit["permit"]))
            if 分型 is not None:
                ts0 = time.perf_counter_ns()
                _S_introp, 立档 = self.selfthink.自省episode(触发源="自发自省", 标记=5)
                self.WCET["自省ns"].append(time.perf_counter_ns() - ts0)
                episode = dict(分型=分型, 拍=self.R1拍数, 维护拍=self.维护拍数,
                               置信=(None if 立档[1] is None else float(立档[1])))
                self.自省事件.append(episode)
        # (5) M14 · 维护R2 + 物化窗口
        self.人格.维护R2(k星, self._v_last)
        π = self.人格.物化窗口()
        ns = time.perf_counter_ns() - t0
        self.WCET["R2维护ns"].append(ns)
        # (6) 离线相调度（自动；离线步独立计量）
        离线 = self.离线相调度() if self.自动离线调度 else dict(调度="停用")
        return dict(维护拍=self.维护拍数, k星=k星, permit=permit, 压力=压,
                    巩固跳过=r2["巩固跳过"], 压平数=r2["压平数"], 驱逐数=r2["驱逐数"],
                    自省=episode,
                    Π窗=(None if π is None else dict(窗序=π["窗序"],
                                                     分量={k: float(π[k]) for k in π["可用"]})),
                    离线相=离线, 状态=self.状态, WCETms=ns / 1e6,
                    模块=dict(hdc=dict(k星=k星, 巩固跳过=r2["巩固跳过"],
                                       压平数=r2["压平数"], 驱逐数=r2["驱逐数"]),
                              ecae=dict(S=float(self.E.S), ACh=float(self.E.ACh)),
                              g_layer=self._最近g拍,
                              personality=(None if π is None else π["窗序"]),
                              selfthink=(None if episode is None else episode["分型"])))

    # ── 离线相调度（S 压力→起飞门→R_r 切片执行→销账→醒后静默） ────────────────
    def 离线相调度(self, 离线pe供给: Optional[Callable] = None) -> dict:
        """S>S_crit ∧ g_layer 起飞门放行 ⇒ 进入离线相（ecae.离线相步循环 R_r 切片）。
        五模块冻结语义：hdc 停步进/selfthink 停触发/personality 停喂（g_layer 照常观测——
        每离线步喂 offline_flag=True 供情境双源）；NE 突变⇒立即中断；
        醒后 post_replay_quiet 经 ecae.结束离线相→治理层.记录离线相结束 联动。"""
        if self.状态 == "offline":
            return self._离线相执行(进入=None, 离线pe供给=离线pe供给)
        压 = self.E.压力()
        if not 压["可辨识"]:
            return dict(调度="不可辨识", 理由="run0 未热（S_crit 不可辨识，不伪装）")
        if not 压["请求"]:
            return dict(调度="无请求", S=float(压["S"]),
                        S_crit=(float(压["S_crit"]) if 压["S_crit"] is not None else None))
        req = self.E.请求离线相()          # 起飞门=治理层 regime_ok(wake) 缓存（ecae 构造注入）
        if not req["允许"]:
            return dict(调度="拦截", 请求=True, 理由=req.get("理由", ""))
        ent = self.E.进入离线相()
        self.状态 = "offline"
        return self._离线相执行(进入=ent, 离线pe供给=离线pe供给)

    def _离线相执行(self, 进入=None, 离线pe供给: Optional[Callable] = None) -> dict:
        供给 = 离线pe供给 if 离线pe供给 is not None else self.离线pe供给
        步数 = 0
        末: Optional[dict] = None
        while True:
            pe = (float(供给(步数)) if 供给 is not None else None)
            ts0 = time.perf_counter_ns()
            r = self.E.离线步(pe=pe)
            self.WCET["离线步ns"].append(time.perf_counter_ns() - ts0)
            if not r.get("离线", False):
                break                      # 无进行中离线相（防御；正常不达）
            步数 += 1
            self.离线选择史.append(int(r["选择"]) if r.get("选择") is not None else -1)
            # 治理层照常观测（不在冻结清单）：情境双源 offline 旗标=引擎离线相位
            obs = self.G.喂步(pe=None, offline_flag=True, x_act=None, x_regime=None, 帧新鲜=True)
            if obs is not None:
                self._最近g拍 = obs
            self.G.ACh = self.E.get_四通道()["ACh"]
            self.β史.append(float(self.G.β_slow))
            if r.get("离线结束") or r.get("中断"):
                末 = r
                break
        # #92 G0f-2：离线相末 W-HOME 压实（ecae 离线步循环退出后、状态回 wake 前；
        # 睡眠依赖记忆巩固——干净 seed 井近正交⇒零触发零改动，非饱和路径逐位不变）
        ts压实 = time.perf_counter_ns()
        压实数 = self.hdc.离线压实()
        self.WCET["离线压实ns"].append(time.perf_counter_ns() - ts压实)
        self.状态 = "wake"
        摘要 = dict(调度="执行", 进入=进入, 步数=步数,
                    完成度=(末 or {}).get("完成度"), 中断=bool((末 or {}).get("中断", False)),
                    S销账后=float(self.E.S), IM压实合并数=int(压实数))
        self.离线相史.append(摘要)
        return 摘要

    # ── K3⊕K4 引擎级供数面 ────────────────────────────────────────────────────
    def kill注入(self, t_步: Optional[int] = None) -> int:
        """K3 kill 事件注入（G.注入kill；步轴=g_layer 步轴）。"""
        return self.G.注入kill(t_步)

    def 恢复判定(self, kill_id: int, scc_alive: Optional[bool] = None) -> dict:
        """K3⊕K4 两段式判定（G.recovery_ok；scc_alive 供数=scc供数 或 调用方显式）。"""
        return self.G.recovery_ok(kill_id, scc_alive=scc_alive)

    def scc供数(self, adj) -> dict:
        """K4 SCC 遥测供数（G.scc_state；adj=补偿流邻接夹具——G6 同律，须标注）。"""
        return self.G.scc_state(adj)

    # ── 导出面（K1–K4 引擎级供数） ────────────────────────────────────────────
    def WCET汇总(self) -> dict:
        out = {}
        for k, arr in self.WCET.items():
            if len(arr) == 0:
                out[k] = dict(n=0, 均值ms=float("nan"), p99ms=float("nan"), 最大ms=float("nan"))
                continue
            a = np.asarray(arr, dtype=np.float64) / 1e6
            out[k] = dict(n=int(len(a)), 均值ms=float(a.mean()),
                          p99ms=float(np.percentile(a, 99)), 最大ms=float(a.max()))
        return out

    def 导出(self) -> dict:
        """占据链 k*(t)（hdc）/活动量 e_t 序列（R1 轴）/S_t 低频差分/四通道序列/Π 窗史/
        自省事件史/β 信号史/离线相与选择史/WCET 计量表/运行统计。"""
        return dict(
            占据链=self.hdc.导出占据链(),
            k星链=list(self.k星链),
            活动量=np.asarray(self.e序列, dtype=np.float64),
            S低频差分=np.asarray(self.ΔS序列, dtype=np.float64),
            四通道=dict(DA=np.asarray(self._chDA, dtype=np.float64),
                        NE=np.asarray(self._chNE, dtype=np.float64),
                        ACh=np.asarray(self._chACh, dtype=np.float64),
                        **{"5HT": np.asarray(self._ch5HT, dtype=np.float64)}),
            β史=np.asarray(self.β史, dtype=np.float64),
            Π窗史=list(self.人格.窗历史),
            自省事件史=list(self.自省事件),
            离线相史=list(self.离线相史),
            离线选择史=list(self.离线选择史),
            WCET=self.WCET汇总(),
            统计=dict(R1拍数=self.R1拍数, 维护拍数=self.维护拍数,
                      离线相数=self.E.离线相数, 离线步数=self.E.离线步数,
                      自省次数=len(self.自省事件), Π窗数=len(self.人格.窗历史),
                      NE突变数=self.E.突变数, H写入数=self.E.写入数,
                      IM井数=len(self.hdc.IM),
                      ecae拍数=self.E.拍数, g步数=self.G.拍数,
                      控制环写入数=self.G.控制环写入数(),
                      状态=self.状态, 账本统计=self.账本.统计()))
