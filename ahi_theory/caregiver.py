# -*- coding: utf-8 -*-
"""ahi_theory/caregiver.py —— M9 · 照顾者域（影子模式）：SENSE_C 三项分解 + D3 三开关 + 10ms 节拍

预登记：ahi_theory/verify_all.py `_CHANGELOG` #79（2026-10-01『M9 照顾者域预登记』，先于改码）。
规格出处：meeting11/95_架构定稿.md §2 L1 caregiver 域（行 83/113/127：10ms 节拍/计算自足/
  δ∈{40,100,200}ms/延迟留白字段）+ L5 照顾者（行 315–317：SENSE_C(t)=C_own(t)+κ·f_应答(CMD(t−δ))+ξ、
  三开关=D3 实验本体）+ D2 κ 匹配 ±5%（行 350/396）+ A7 双约束（行 228–229）。

影子模式纪律（红线）：
  · 纯新增；**默认不加载**（无模块级副作用——消费方显式 import 才实例化）；
  · **不 import brain/world/system/three_loop/mbp/v2_engine/hdc_core**（影子纪律红线；CMD 由调用方
    从 V2 引擎 hdc 动作流只读喂入，本模块不反向依赖引擎；ledger 仅构造带账本时惰性 try-import，
    用于「变更清单扩充=结构事件」预授权路径，ledger.py 源文件零改动）；
  · **零拟合红线**：全文件无 lstsq / linalg.inv·pinv / solve / scipy.optimize / 参数搜索构造
    （C4 审计器 tokenize 级可检，C9_selftest [3] 自证）；tanh/正态抽样=既有机制原语非拟合；
  · **计算自足**（95 L1 caregiver 域原文）：产出只依赖喂入的 CMD/C_own 流与自身预注册常量，
    不依赖任何系统侧回传（personality/g_layer/引擎读数一概不读）；
  · 诚实边界：本影子版只产出 SENSE_C 流供 personality 的 P_depend 维消费（personality 一行不改——
    P_depend 因果影子版联动在 C9_selftest [9] 完成）；长时程 D2/D3 行为判据、R_care 子域激活、
    ecae.A7资格 实流待 V2 供数与课程引擎另行预注册接入，不虚设。

SENSE_C 分解（逐 tick 记录，机器可检）：
  SENSE_C(t) = C_own(t) + κ_eff · f_应答(CMD(t−δ)) + ξ(t)
    C_own  ：自身状态信号（供数接口，喂入；影子期 V2 联动臂以引擎 e_t 只读代理——标注）
    f_应答 ：预注册平滑应答函数 f(c)=tanh(⟨c, R_resp⟩)，R_resp=双极/√cmd维度
             （CARE_SEED=4096 单 rng 顺序派生；候选身份——95 L5 未定形式，候选不虚设定稿）
    κ_eff  ：0<κ<1 主档 0.6（候选）；still-face 开关⇒κ_eff=0（应答项消失）；缺席开关⇒CMD 停供
             （喂拍不再消费 cmd，应答项=0）；陌生人开关⇒换 seed 重生 R_resp（f 函数换 seed，ξ 流不变）
    ξ(t)   ：~N(0,σ_ξ)，σ_ξ=0.05（候选，预注册）；独立 ξ 流（种子域内顺序派生，开关不换流）
    δ      ：∈{40,100,200}ms 主档 100ms（=5 R1 拍换算【O8 换算率开放，标注候选】；
             caregiver 节拍 10ms=0.5 R1 拍语义 ⇒ δ 主档=10 tick；δtick=round(δ毫秒/节拍毫秒)）

时钟域（95 L1 caregiver 域）：10ms 独立节拍（影子化——tick 由调用方驱动，本模块不自转时钟）；
  逐 tick 记录含 延迟留白=None 字段（系统侧不可读占位，A9 半闭口径）。
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np

__all__ = ["CARE_SEED", "照顾者节拍毫秒", "δ档毫秒", "δ主档毫秒", "κ主档", "σ_ξ", "照顾者"]

# ── §0 预注册常量（95 L1/L5；变更 = kind="参数变更" 账本结构事件） ────────────────────
CARE_SEED = 4096               # 照顾者单 rng 种子（R_resp/ξ 流/分量锚 顺序派生；
                               # 种子域与 hdc 1024/personality 2048/selfthink 3072 互不相交）
照顾者节拍毫秒 = 10.0          # caregiver 域独立节拍（95 L1 行 83；=0.5 R1 拍语义）
R1拍毫秒 = 20.0                # 1 R1 拍=20ms（95 L1 行 80；δ 的 R1 拍换算基准【O8 开放，候选】）
δ档毫秒 = (40.0, 100.0, 200.0)  # δ 三档（95 L1 行 83/113：40=近完美对照档/100 主档/200 大延迟档）
δ主档毫秒 = 100.0              # E2 主档（95 L1 行 113「E2 主档 δ=100ms」）
κ主档 = 0.6                    # 0<κ<1 主档（候选——L5 行 315「0<κ<1」未定值）
σ_ξ = 0.05                     # ξ~N(0,σ_ξ) 预注册初值（候选）
D_锚 = 1024                    # P_depend 分量锚维度（与 personality.D_P 同律；结构先验）
组空阈 = 0.05                  # P_depend 影子编码「组空⇒零向量」判据阈（C9 [9] 消费口径，预注册）


def _双极(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.where(rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32)


class 照顾者:
    """照顾者域影子（10ms 节拍；SENSE_C 三项分解 + D3 三开关）。

    O(1)/tick 纪律：喂拍 只动标量与定容 deque（O(1)；无数组分配——cmd 史 deque maxlen=δtick+1）；
    get_流() 一次性物化 numpy 流（消费面，O(n) 一次性）。"""

    def __init__(self, cmd维度: int = 4, 账本=None, 种子: int = CARE_SEED,
                 κ: float = κ主档, δ毫秒: float = δ主档毫秒, σ_ξ: float = σ_ξ):
        if not (0.0 < float(κ) < 1.0):
            raise ValueError("κ 须 0<κ<1（95 L5 原文）；still-face 用 开关 而非 κ=0 构造")
        if float(δ毫秒) not in δ档毫秒:
            raise ValueError("δ毫秒 须 ∈ %r（95 L1 三档）" % (δ档毫秒,))
        self.cmd维度 = int(cmd维度)
        self.种子 = int(种子)
        self._账本 = 账本
        self.κ = float(κ)
        self.δ毫秒 = float(δ毫秒)
        self.σ_ξ = float(σ_ξ)
        self.δtick = int(round(self.δ毫秒 / 照顾者节拍毫秒))     # 主档 100ms→10 tick（=5 R1 拍）
        rng = np.random.default_rng(self.种子)                   # 单 rng 顺序派生（预注册顺序，不挑）
        self.R_resp = _双极(rng, self.cmd维度) / math.sqrt(self.cmd维度)   # f_应答 投影（候选）
        self._ξrng = np.random.default_rng(self.种子 + 1)        # ξ 独立流（域内顺序派生；开关不换）
        # 逐 tick 状态（O(1)）
        self.tick = 0
        self.模式 = "在场"                                        # 在场 / still_face / 缺席 / 陌生人*
        self.开关史: list = []                                    # (tick, 动作) 审计
        self._cmd史: deque = deque([None] * (self.δtick + 1), maxlen=self.δtick + 1)
        self._记录: list = []                                     # 逐 tick 记录（影子期全留；消费面物化）
        self._参数预注册入账()

    # —— 构造辅助：预注册入账（键前缀 "照顾者_"；账本 None⇒跳过） ——
    def _参数预注册入账(self) -> None:
        if self.账本无效():
            return
        账本 = self._账本
        键值 = (
            ("照顾者_CARE_SEED", CARE_SEED,
             "M9 #79 预注册初值：照顾者单 rng 种子（R_resp/ξ 流/分量锚顺序派生；种子域 "
             "4096 与 1024/2048/3072 互不相交）【定稿（结构）】"),
            ("照顾者_节拍毫秒", 照顾者节拍毫秒,
             "M9 #79 预注册：caregiver 域 10ms 独立节拍=0.5 R1 拍语义（95 L1 行 83）【定稿】"),
            ("照顾者_δ档毫秒", list(δ档毫秒),
             "M9 #79 预注册：δ 三档 {40,100,200}ms（95 L1 行 83/113）【定稿】"),
            ("照顾者_δ主档毫秒", δ主档毫秒,
             "M9 #79 预注册：δ 主档 100ms（E2 主档，95 L1 行 113）【定稿】"),
            ("照顾者_δ_R1拍换算", δ主档毫秒 / R1拍毫秒,
             "M9 #79 预注册：δ 主档=5 R1 拍换算（R1拍毫秒=20ms；O8 换算率开放）【候选】"),
            ("照顾者_κ主档", κ主档,
             "M9 #79 预注册初值：0<κ<1 主档（95 L5 行 315 未定值）【候选】"),
            ("照顾者_σ_ξ", σ_ξ,
             "M9 #79 预注册初值：ξ~N(0,σ_ξ)【候选】"),
            ("照顾者_f_应答形式", "tanh(⟨c,R_resp⟩)，R_resp=双极/√cmd维度",
             "M9 #79 预注册：平滑应答函数形式（95 L5 未定形式；零拟合——无数据驱动标定）【候选】"),
        )
        try:
            from ahi_theory import ledger as _ledger
        except Exception:
            _ledger = None
        if _ledger is not None and hasattr(_ledger, "变更清单"):
            新键 = [k for k, _, _ in 键值 if k not in _ledger.变更清单]
            if 新键:
                账本.登记("结构事件",
                          payload={"事件": "变更清单扩充", "键族": "照顾者_", "新增键": 新键},
                          理由="M9 #79 预注册：照顾者参数键族入清单"
                               "（ledger 预授权：扩充清单本身=结构事件；运行时扩充，"
                               "ledger.py 源文件零改动）")
                _ledger.变更清单 = _ledger.变更清单 | frozenset(新键)
        for 键, 值, 理由 in 键值:
            账本.登记("参数变更", payload={"键": 键, "值": 值},
                      理由=理由 + "（先验不调；变更须再次结构事件）")

    def 账本无效(self) -> bool:
        return getattr(self, "_账本", None) is None

    # —— D3 三开关（实验本体；95 L5 行 316 原文三态） ——
    def set_开关(self, 动作: str, 新种子: Optional[int] = None) -> dict:
        """still_face（κ_eff→0，应答项消失；cmd 照常消费——脸仍对着）/ 缺席（真缺席=进程停，
        喂拍不再消费 cmd=停止供数，应答项=0）/ 陌生人（换实例=换 seed 重生 f_应答——仅重生成
        R_resp，ξ 自噪流不变）/ 在场（恢复主档）。"""
        if 动作 == "still_face":
            self.模式 = "still_face"
        elif 动作 == "缺席":
            self.模式 = "缺席"
        elif 动作 == "陌生人":
            if 新种子 is None:
                raise ValueError("陌生人开关须给 新种子（换实例=换 seed）")
            rng = np.random.default_rng(int(新种子))
            self.R_resp = _双极(rng, self.cmd维度) / math.sqrt(self.cmd维度)
            self.模式 = "陌生人@%d" % int(新种子)
        elif 动作 == "在场":
            self.模式 = "在场"
        else:
            raise ValueError("未知开关动作 %r（须 ∈ {still_face, 缺席, 陌生人, 在场}）" % (动作,))
        self.开关史.append((self.tick, 动作))
        return dict(tick=self.tick, 动作=动作, 模式=self.模式,
                    κ_eff=self._κ_eff(), R_resp指纹=self.R_resp指纹())

    def _κ_eff(self) -> float:
        """still-face⇒0（应答项消失）；缺席⇒0（供数停，应答项结构性为 0）；其余=κ 主档。"""
        return 0.0 if self.模式 in ("still_face", "缺席") else self.κ

    def R_resp指纹(self) -> str:
        """f_应答 函数身份指纹（陌生人开关换 seed ⇒ 指纹必变；C9 [4] 可检）。"""
        import hashlib
        return hashlib.sha256(np.ascontiguousarray(self.R_resp).tobytes()).hexdigest()[:16]

    # —— 逐 tick 喂入（10ms 节拍；影子化——tick 由调用方驱动） ——
    def 喂拍(self, C_own: float, cmd: Optional[np.ndarray] = None) -> dict:
        """单 caregiver tick：SENSE_C = C_own + κ_eff·f_应答(CMD(t−δ)) + ξ。
        CMD(t−δ)：cmd 史定容 deque（maxlen=δtick+1）；t<δtick（史未满）⇒ 应答项=0（无历史不伪装）；
        cmd=None（供数停/缺席）⇒ 应答项=0。缺席模式下 cmd 实参不消费（真缺席=停止供数语义）。
        计算自足：本方法不读任何系统侧对象。"""
        c = float(C_own)
        f值 = 0.0
        if self.模式 != "缺席" and cmd is not None:
            v = np.asarray(cmd, dtype=np.float32).ravel()
            if v.size != self.cmd维度:
                raise ValueError("cmd 维度 %d ≠ 构造 cmd维度 %d" % (v.size, self.cmd维度))
            self._cmd史.append(v.copy())
        else:
            self._cmd史.append(None)
        if self.tick >= self.δtick:
            cmd延迟 = self._cmd史[0]                 # t−δtick 处的 CMD（史满后队首=t−δ）
            if cmd延迟 is not None:
                f值 = float(math.tanh(float(np.dot(cmd延迟, self.R_resp))))
        ξ = float(self._ξrng.normal(0.0, self.σ_ξ))
        κ_eff = self._κ_eff()
        应答项 = κ_eff * f值
        sense = c + 应答项 + ξ
        self.tick += 1
        rec = dict(tick=self.tick, C_own=c, f=f值, 应答项=应答项, ξ=ξ, SENSE_C=sense,
                   κ_eff=κ_eff, δ毫秒=self.δ毫秒, δtick=self.δtick, 模式=self.模式,
                   延迟留白=None)                    # 系统侧不可读占位（95 L1 定时包追加冻结接口）
        self._记录.append(rec)
        return rec

    # —— 消费面（personality 的 P_depend 维供数；personality 本体一行不改） ——
    def get_流(self) -> dict:
        """物化 SENSE_C 流（一次性 numpy 化；只读——不改变内部状态）。
        输出=SENSE_C/C_own/应答项/f延迟/ξ 五条流 + tick 数组 + 头部元数据（预注册常量+开关史）。"""
        n = len(self._记录)
        流 = dict(
            tick=np.arange(1, n + 1, dtype=np.int64),
            SENSE_C=np.asarray([r["SENSE_C"] for r in self._记录], dtype=np.float64),
            C_own=np.asarray([r["C_own"] for r in self._记录], dtype=np.float64),
            应答项=np.asarray([r["应答项"] for r in self._记录], dtype=np.float64),
            f延迟=np.asarray([r["f"] for r in self._记录], dtype=np.float64),
            ξ=np.asarray([r["ξ"] for r in self._记录], dtype=np.float64),
            头=dict(种子=self.种子, κ=self.κ, δ毫秒=self.δ毫秒, δtick=self.δtick,
                    σ_ξ=self.σ_ξ, 节拍毫秒=照顾者节拍毫秒, cmd维度=self.cmd维度,
                    模式终态=self.模式, 开关史=list(self.开关史),
                    R_resp指纹终态=self.R_resp指纹(),
                    SENSE_C式="C_own + κ_eff·f_应答(CMD(t−δ)) + ξ（#79 预注册）"))
        return 流

    def 分量锚(self) -> dict:
        """P_depend 影子编码分量锚（C9 [9] 消费面；personality 代数 A_c=R_pers[组]⊛锚_c 的锚半部）。
        由 CARE_SEED rng 在 R_resp/ξ 流之后继续顺序派生（预注册顺序，不挑）；零拟合。"""
        if not hasattr(self, "_锚缓存"):
            rng = np.random.default_rng(self.种子 + 2)       # 域内顺序派生（R_resp=种子流、ξ=+1、锚=+2）
            self._锚缓存 = dict(应答相关=_双极(rng, D_锚), 应答占比=_双极(rng, D_锚))
        return dict(self._锚缓存)
