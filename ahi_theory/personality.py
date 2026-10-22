# -*- coding: utf-8 -*-
"""ahi_theory/personality.py —— M14 · 人格层（影子模式）：Π 统计剖面 + P:=encode(Π) + 人格谱系

预登记：verify_all._CHANGELOG 2026-09-30『M14 人格层预登记』（第 14 次会议五专家两轮收敛，
meeting14/00_人格专项会议纪要.md）。规格出处：纪要 §1 四层栈 / §2 裁决表 / §3 维度对齐表 /
§4 PERS 判据组 / §5 参数表 / §6 施工序（M14a）。

影子模式纪律：
  · 纯新增；**默认不加载**（无模块级副作用、无任何内核接线——消费方显式 import 才实例化）；
  · **不 import brain/world/system/three_loop/mbp**（影子纪律红线；ledger 仅在构造带账本时
    惰性 try-import，用于「变更清单扩充=结构事件」的预授权路径，ledger.py 源文件零改动）；
  · **零拟合红线**：全文件无 lstsq / linalg.inv·pinv / solve / scipy.optimize / 参数搜索构造
    （C4 审计器 tokenize 级可检，meeting14/PERS_selftest [2] 自证）；统计工具仅
    scipy.stats.spearmanr（秩相关，非拟合）；
  · 人格≠活性证据（纪要 §2#7）：本模块只做读出聚合与身份承载，不产出任何活性断言，
    严禁以「人格仍在更新」反证 R2 活性；
  · 诚实边界：PERS 引擎级真值测量（长时程发育数据）待 V2 引擎供数后另行预注册接入
    （与 K1–K4 同律，不虚设）；依赖度/distress 维待 M9 照顾者域供数（P_depend 组
    预留、诚实置零，不造数——纪要 §3 维度对齐表）。

四层栈（纪要 §1，本模块承载 Π/P/持久三层；x 机制层=hdc_core 影子已交付，由台架喂流）：
  经历流 ──▶ x 机制层 ──▶ Π 统计层【主判据对象】──▶ P HDC 码层 ──▶ 行为施测口径
  P := encode(Π)（E3）：分组 {explore, depend, novelty, valence}，P_k = sign(Σ_{c∈组} π_c·A_c)；
  组空 ⇒ 零向量（诚实 N/A）；A_c = R_pers[组k] ⊛ 分量锚（双极逐元素乘）。
  R_pers[1..4] 与分量锚由 PERS_SEED=2048 单 rng 顺序派生（独立模块独立 rng——不动
  hdc_core 的 MASTER_SEED=1024 消耗序，两 seed 域永不相交）。

阈值统一（纪要 §2#2）：机会校正 0.7 标尺——连续量 Spearman≥0.7；本模块只承载体裁机制量
（剖面秩相关/置换检验/角距），PERS1–P5 判据式与阈值预注册于 meeting14/PERS_selftest.py
（引擎级真值判据待 V2 供数，不虚设）。

滞后隔离与两阶段制（纪要 §2#3，E3）：维护拍 t 的观测进 pending 延迟提交队列，
t+ISOL_MAINT 才并入当前窗累积器 ⇒ 当前读出不含受当前 P 影响的近期行为；
阶段 A 立档（读出-only ≥3 窗）；阶段 B 的 G(a) 破并列反作用通道**接口预注册、默认关闭**
（消融臂保留，本影子版不实现反作用本体）。

身份语义（纪要 §2#5/#6，E4）：链式 hash H_i = sha256(H_{i-1} ‖ canonical(读出_i) ‖ seq_i ‖ 事件标记_i)；
身份域**排除** seed/设备ID/图式hash/B7校准量/参数实时值 ⇒ 换 seed 重放同经历 ⇒ 身份 hash
逐位不变；检查点段机器可检不变量 `段.seq ≤ 谱系尾.seq ∧ 段.身份键 == 谱系尾链hash`，
违例 fail-fast 拒载（不静默修补）；检查点回滚 = 谱系尾追加「分叉事件」（回滚也是经历，
不破坏追加纪律）。

账本纪律（纪要 §2#1，E1）：账本登记自由度，不登记自由度的解——参数**当前值全是状态**
（不入账本）；**生成结构**（预注册初值/窗长/隔离深/维度上限/身份域排除表）是常数，
构造时以 kind="参数变更" 入账本（键前缀 "人格_"），账本为 None 时跳过。
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import deque

import numpy as np
from scipy.stats import spearmanr

__all__ = ["人格剖面器", "人格谱系", "canonical", "角距", "剖面秩相关", "置换检验",
           "PERS_SEED", "W_PERS_MAINT", "ISOL_MAINT"]

# ── §0 预注册常量（纪要 §5 参数表；变更 = kind="参数变更" 账本结构事件） ─────────────
PERS_SEED = 2048              # 人格编码单 rng 种子（R_pers[1..4] 与编码锚顺序派生；
                              # 不动 hdc_core 的 MASTER_SEED=1024 消耗序）
W_PERS_MAINT = 820            # 聚合窗（维护拍）=16400 R1 拍≈16384 规格（维护每 20 R1 拍）
ISOL_MAINT = 7                # 滞后隔离 7 维护拍=128 R1 拍（当前窗聚合只累计至 [t−W, t−ISOL]：
                              # 实现为延迟提交缓冲——维护拍 t 的观测进 pending，t+ISOL 才并入）
立档窗数 = 3                  # 阶段 A 读出-only ≥3 窗；反作用通道接口预注册、默认关闭、消融臂保留
PI维度上限 = 32               # Π 读出维度上限（有界分量）
快照KB上限 = 64               # 谱系单快照 ≤64KB（canonical 序列化字节数断言）
身份域排除 = ("seed", "设备ID", "图式hash", "B7校准量", "参数实时值")   # 纪要 §2#5 排除表
维护节拍 = 20                 # 1 维护拍 = 20 R1 拍（纪要 §5 时钟域；台架侧节流比，只作注释承载）
D_P = 1024                    # HDC 码层维度（与 VSA D=1024 同律；结构先验不扫参）

# Π 分量固定顺序（θ 角向量排布序）与声明域（物化窗口 的有界性契约，PERS_selftest [4] 可检）
Π分量顺序 = ("惊讶写入率", "e_均值", "探索比", "动作熵", "重复率",
             "tag_均值", "tag_偏度", "井占据熵", "主井占比")
Π分量域 = {"惊讶写入率": (0.0, 1.0), "e_均值": (0.0, 1.0), "探索比": (0.0, 1.0),
           "动作熵": (0.0, 1.0), "重复率": (0.0, 1.0),
           "tag_均值": (-1.0, 1.0), "tag_偏度": (-1.0, 1.0),
           "井占据熵": (0.0, 1.0), "主井占比": (0.0, 1.0)}

# P:=encode(Π) 分组（纪要 §2 裁决 #E3 / 命令规格）：组空 ⇒ 零向量（诚实 N/A——
# depend 无照顾者流分量、distress 待 M9 供数，影子期不造数）
编码分组 = {"explore": ("探索比", "动作熵", "重复率"),
            "depend": (),
            "novelty": ("e_均值", "惊讶写入率"),
            "valence": ("tag_均值", "tag_偏度")}
组角色序 = ("explore", "depend", "novelty", "valence")   # ↔ R_pers[1..4]


# ── §1 模块级工具（canonical / 角距 / 剖面秩相关 / 置换检验） ────────────────────────
def _规整(x):
    """canonical 前的递归规整：float 固定精度（1e-12 相对形态）保证逐位可复现；
    NaN/±Inf 文字化（json 无此字面量，落读出禁用——防御性保序）；tuple→list 同形。"""
    if isinstance(x, bool):
        return x
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            return repr(x)
        return float(f"{x:.12e}")
    if isinstance(x, (int, str)):
        return x
    if isinstance(x, dict):
        return {str(k): _规整(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_规整(v) for v in x]
    return str(x)


def canonical(读出dict) -> str:
    """逐位可复现的 canonical 序列化（sort_keys + float 固定精度 + 紧凑分隔符）。
    谱系链式 hash 的字节源——同一读出在任何平台/任何时刻产出同一字节串。"""
    return json.dumps(_规整(读出dict), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def _香农熵比(直方: dict) -> float:
    """归一香农熵 H/ln(支撑数)∈[0,1]；零/单支撑 ⇒ 0.0（诚实：无分布可散布）。"""
    总 = sum(直方.values())
    if 总 <= 0:
        return 0.0
    支撑 = len([v for v in 直方.values() if v > 0])
    if 支撑 <= 1:
        return 0.0
    h = 0.0
    for v in 直方.values():
        if v > 0:
            p = v / 总
            h -= p * math.log(p)
    return h / math.log(支撑)


def 角距(θ1, θ2) -> float:
    """角向量距离 arccos(clip cos)∈[0,π]；任一零向量 ⇒ π/2（机会方向，诚实不可比对）。"""
    a = np.asarray(θ1, dtype=np.float64)
    b = np.asarray(θ2, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return math.pi / 2.0
    c = float(np.dot(a, b) / (na * nb))
    return math.acos(max(-1.0, min(1.0, c)))


def 剖面秩相关(Π1: dict, Π2: dict) -> float:
    """两窗 Π 的 Spearman 秩相关（机会校正 0.7 标尺的连续量口径）：
    按共同可用分量取值对；共同分量<2 ⇒ nan（不可判）；逐位相等 ⇒ 1.0
    （scipy 对常数输入返回 nan——完全稳定是合法结局，诚实定义而非吞掉）。"""
    共同 = [c for c in Π1.get("可用", ()) if c in Π2.get("可用", ())]
    if len(共同) < 2:
        return float("nan")
    a = np.array([float(Π1[c]) for c in 共同])
    b = np.array([float(Π2[c]) for c in 共同])
    if np.array_equal(a, b):
        return 1.0
    ρ, _p = spearmanr(a, b)
    return float(ρ)


def 置换检验(组甲, 组乙, 距离函数=None, 重复: int = 2000, 种子: int = PERS_SEED + 1) -> dict:
    """组间/组内均距差的标签置换检验（机会校正零假设；判据 p<0.05）。

    组甲/组乙：Π 窗 dict 列表；距离函数：缺省 = θ 角距。
    统计量 T = mean(跨组距) − mean(组内距)；零假设 = 两组同分布（标签可置换）⇒
    p = (1 + #{T_perm ≥ T_obs}) / (重复+1)。预注册种子（不挑）。"""
    n = len(组甲) + len(组乙)
    if len(组甲) < 1 or len(组乙) < 1:
        return dict(统计量=float("nan"), p值=1.0, 重复=int(重复), 备注="组空")
    全部 = list(组甲) + list(组乙)
    距 = (距离函数 if 距离函数 is not None
          else (lambda p, q: 角距(p["θ"], q["θ"])))
    DM = [[0.0 if i == j else 距(全部[i], 全部[j]) for j in range(n)] for i in range(n)]
    对 = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def 统计量(标签):
        跨 = [DM[i][j] for i, j in 对 if 标签[i] != 标签[j]]
        内 = [DM[i][j] for i, j in 对 if 标签[i] == 标签[j]]
        tc = sum(跨) / len(跨) if 跨 else 0.0
        nc = sum(内) / len(内) if 内 else 0.0
        return tc - nc

    标0 = [0] * len(组甲) + [1] * len(组乙)
    obs = 统计量(标0)
    rng = np.random.default_rng(种子)
    计 = 0
    for _ in range(int(重复)):
        if 统计量(list(rng.permutation(标0))) >= obs:
            计 += 1
    return dict(统计量=float(obs), p值=(计 + 1) / (int(重复) + 1), 重复=int(重复))


def _双极(rng: np.random.Generator, n: int) -> np.ndarray:
    """双极 ±1 向量（dtype=float32；⊛=逐元素乘即 XNOR 保持律，与 hdc_core._双极 同构）。"""
    return np.sign(rng.standard_normal(n)).astype(np.float32)


# ── §2 Π 统计剖面（主判据对象）+ P:=encode(Π) 码层 ──────────────────────────────────
class 人格剖面器:
    """人格 = 剖面（纪要 §1）：由经历流沉淀的慢统计剖面，**涌现/习得**——
    本类不含任何硬编码性格标签；零经历 ⇒ 无剖面、编码全零（反硬编码，[3] 可检）。

    O(1) 纪律：观测R1 / 维护R2 / 步进推进 只动标量与小 dict（逐维护拍 a直方 ≤
    每维护拍 R1 拍数个键、井直方 ≤ 井数键——O(1) 摊销、无数组分配）；物化窗口
    每 W_PERS_MAINT 维护拍一次（摊销 O(Π)，≤PI维度上限 维分量）。
    """

    def __init__(self, 账本=None, 种子: int = PERS_SEED):
        self.账本 = 账本
        self.种子 = int(种子)
        rng = np.random.default_rng(self.种子)          # 单 rng 顺序派生（预注册顺序，不挑）
        # R_pers[1..4] 角色键（explore/depend/novelty/valence ↔ 纪要 §1 E3 码层四维）
        self.R_pers = np.stack([_双极(rng, D_P) for _ in range(4)])
        # 编码锚 A_c = R_pers[组k] ⊛ 分量锚（分量锚从同一 rng 继续 _双极 派生）；
        # depend 组空 ⇒ 无锚（诚实 N/A，不占 rng 派生额）
        self._分量锚 = {名: _双极(rng, D_P) for 名 in Π分量顺序}
        self.A_锚 = {}
        for gi, 组名 in enumerate(组角色序):
            for 名 in 编码分组[组名]:
                self.A_锚[名] = self.R_pers[gi] * self._分量锚[名]
        # 延迟提交队列（ISOL_MAINT 深度语义；纪要 §2#3 滞后聚合）
        self._pending: deque = deque()
        self._维护序 = 0
        # 当前维护拍 R1 侧缓冲（标量/小 dict——观测R1 O(1) 累计，维护R2 收口打包）
        self._b_惊 = 0
        self._b_e和 = 0.0
        self._b_e计 = 0
        self._b_拍 = 0
        self._b_a直方: dict = {}
        self._b_非零 = 0
        self._b_邻重 = 0
        self._b_邻对 = 0
        self._前a = None
        # 当前窗累积器（已过隔离期的提交；全标量/小 dict）
        self._w = self._空累积器()
        # 窗历史（Π dict 列表）与阶段（阶段 A 读出-only；反作用通道默认关闭）
        self.窗历史: list = []
        self.stage = "A"
        self.反作用通道 = None                  # 阶段 B 接口位（预注册；None=关闭）
        self._参数预注册入账()

    # —— 构造辅助 ——
    @staticmethod
    def _空累积器() -> dict:
        return dict(观测数=0, 拍=0, 惊=0, e和=0.0, e计=0, 非零=0, 邻重=0, 邻对=0,
                    tag和=0.0, tag平方和=0.0, tag立方和=0.0, tag计=0,
                    a直方={}, 井直方={})

    def _参数预注册入账(self) -> None:
        """预注册常量以 kind="参数变更" 入账本（键前缀 "人格_"）；账本 None ⇒ 跳过。
        变更清单扩充走 ledger 预授权路径：「扩充清单本身 = 结构事件」（ledger.py 文档
        预授权；运行时模块对象扩充，ledger.py 源文件一行不改）。"""
        if self.账本 is None:
            return
        键值 = (
            ("人格_PERS_SEED", PERS_SEED,
             "M14 #67 预注册初值：人格编码单 rng 种子（R_pers[1..4]+编码锚顺序派生；"
             "独立于 hdc_core MASTER_SEED=1024 消耗序）"),
            ("人格_W_PERS_MAINT", W_PERS_MAINT,
             "M14 #67 预注册初值：聚合窗 820 维护拍（=16400 R1 拍≈16384 规格，维护每 20 R1 拍）"),
            ("人格_ISOL_MAINT", ISOL_MAINT,
             "M14 #67 预注册初值：滞后隔离 7 维护拍=128 R1 拍"
             "（延迟提交缓冲，当前读出不含受当前 P 影响的近期行为）"),
            ("人格_立档窗数", 立档窗数,
             "M14 #67 预注册初值：阶段 A 立档 ≥3 窗（读出-only；反作用通道默认关闭、消融臂保留）"),
            ("人格_PI维度上限", PI维度上限,
             "M14 #67 预注册初值：Π 读出维度上限 32 有界分量"),
            ("人格_快照KB上限", 快照KB上限,
             "M14 #67 预注册初值：谱系单快照 ≤64KB（canonical 字节断言）"),
            ("人格_身份域排除", list(身份域排除),
             "M14 #67 预注册：身份域排除表（seed/设备ID/图式hash/B7校准量/参数实时值不入读出"
             " ⇒ 换 seed 重放 ⇒ 身份 hash 逐位不变）"),
        )
        try:
            from ahi_theory import ledger as _ledger
        except Exception:
            _ledger = None
        if _ledger is not None and hasattr(_ledger, "变更清单"):
            新键 = [k for k, _, _ in 键值 if k not in _ledger.变更清单]
            if 新键:
                self.账本.登记("结构事件",
                               payload={"事件": "变更清单扩充", "键族": "人格_", "新增键": 新键},
                               理由="M14 #67 预注册：人格参数键族入清单"
                                    "（ledger 预授权：扩充清单本身=结构事件；运行时扩充，"
                                    "ledger.py 源文件零改动）")
                _ledger.变更清单 = _ledger.变更清单 | frozenset(新键)
        for 键, 值, 理由 in 键值:
            self.账本.登记("参数变更", payload={"键": 键, "值": 值},
                           理由=理由 + "（先验不调；变更须再次结构事件）")

    # —— R1 侧观测（每 R1 拍；O(1)）——
    def 观测R1(self, e_t, a星序号, 帧新鲜: bool = True) -> None:
        """惊讶写入计数（e_t 非 None 且 >0）、e 累和、a星序号直方图、相邻重复动作计数。
        只进当前维护拍缓冲（收口见 维护R2 → pending → 步进推进 并窗）。"""
        self._b_拍 += 1
        if (e_t is not None) and (e_t > 0.0):
            self._b_惊 += 1
        if e_t is not None:
            self._b_e和 += float(e_t)
            self._b_e计 += 1
        序号 = int(a星序号)
        self._b_a直方[序号] = self._b_a直方.get(序号, 0) + 1
        if 序号 != 0:
            self._b_非零 += 1
        if self._前a is not None:
            self._b_邻对 += 1
            if self._前a == 序号:
                self._b_邻重 += 1
        self._前a = 序号

    # —— R2 侧观测（每维护拍收口；O(1)）——
    def 维护R2(self, k星, tag) -> None:
        """井占据直方、tag 累和/平方和/立方和/计数（tag clip [-1,1]）。
        本维护拍全部观测（R1 缓冲 + R2 量）打包进 pending 延迟提交队列（隔离期 ISOL_MAINT）。"""
        self._维护序 += 1
        self._pending.append(dict(
            序=self._维护序,
            拍=self._b_拍, 惊=self._b_惊, e和=self._b_e和, e计=self._b_e计,
            非零=self._b_非零, 邻重=self._b_邻重, 邻对=self._b_邻对,
            a直方=self._b_a直方,
            k星=int(k星), tag=float(np.clip(float(tag), -1.0, 1.0))))
        self._b_惊 = 0
        self._b_e和 = 0.0
        self._b_e计 = 0
        self._b_拍 = 0
        self._b_a直方 = {}
        self._b_非零 = 0
        self._b_邻重 = 0
        self._b_邻对 = 0

    # —— 延迟提交（每维护拍调用一次；O(1) 摊销）——
    def 步进推进(self) -> None:
        """把隔离期满（序 ≤ 当前维护序 − ISOL_MAINT）的观测并入当前窗累积器。
        ⇒ 当前窗聚合只累计至 [t−W, t−ISOL]：读出不含受当前 P 影响的近期行为。"""
        界 = self._维护序 - ISOL_MAINT
        while self._pending and self._pending[0]["序"] <= 界:
            self._并入窗(self._pending.popleft())

    def _并入窗(self, o: dict) -> None:
        w = self._w
        w["观测数"] += 1
        w["拍"] += o["拍"]
        w["惊"] += o["惊"]
        w["e和"] += o["e和"]
        w["e计"] += o["e计"]
        w["非零"] += o["非零"]
        w["邻重"] += o["邻重"]
        w["邻对"] += o["邻对"]
        for k, v in o["a直方"].items():
            w["a直方"][k] = w["a直方"].get(k, 0) + v
        w["井直方"][o["k星"]] = w["井直方"].get(o["k星"], 0) + 1
        w["tag和"] += o["tag"]
        w["tag平方和"] += o["tag"] * o["tag"]
        w["tag立方和"] += o["tag"] * o["tag"] * o["tag"]
        w["tag计"] += 1

    # —— 窗物化（满 W_PERS_MAINT 维护拍；摊销 O(Π)）——
    def 物化窗口(self):
        """满 W_PERS_MAINT 维护拍 ⇒ 累积器转 Π 窗（全有界分量），重置累积器，
        窗历史追加，返回 Π；未满 ⇒ None。"""
        if self._w["观测数"] < W_PERS_MAINT:
            return None
        w = self._w
        π: dict = {}
        π["惊讶写入率"] = (w["惊"] / w["拍"]) if w["拍"] else 0.0
        π["e_均值"] = min(1.0, max(0.0, (w["e和"] / w["e计"]) / 2.0)) if w["e计"] else 0.0
        π["探索比"] = (w["非零"] / w["拍"]) if w["拍"] else 0.0
        π["动作熵"] = _香农熵比(w["a直方"])
        π["重复率"] = (w["邻重"] / w["邻对"]) if w["邻对"] else 0.0
        n = w["tag计"]
        if n:
            m = w["tag和"] / n
            m2 = w["tag平方和"] / n - m * m
            m3 = w["tag立方和"] / n - 3.0 * m * (w["tag平方和"] / n) + 2.0 * m ** 3
            偏 = (m3 / (m2 ** 1.5)) if m2 > 1e-18 else 0.0
        else:
            m, 偏 = 0.0, 0.0
        π["tag_均值"] = max(-1.0, min(1.0, m))
        π["tag_偏度"] = max(-1.0, min(1.0, 偏))
        π["井占据熵"] = _香农熵比(w["井直方"])
        井总 = sum(w["井直方"].values())
        π["主井占比"] = (max(w["井直方"].values()) / 井总) if 井总 else 0.0
        # 可用分量掩码：依赖度/distress 影子期 N/A——不造数（P_depend 组空、诚实置零）
        π["可用"] = list(Π分量顺序)
        π["θ"] = tuple(float(π[名]) for 名 in Π分量顺序)
        π["窗序"] = len(self.窗历史)
        π["维护拍数"] = int(w["观测数"])
        self.窗历史.append(π)
        self._w = self._空累积器()
        return π

    # —— 读出 ——
    def 剖面(self):
        """最近窗 Π；零经历（无窗）⇒ None（诚实：无人格态可检，反硬编码）。"""
        return self.窗历史[-1] if self.窗历史 else None

    def 角向量θ(self):
        """可用分量按固定顺序（Π分量顺序）排向量，L2 归一；无窗 ⇒ None。"""
        π = self.剖面()
        if π is None:
            return None
        v = np.array([float(π[名]) for 名 in Π分量顺序], dtype=np.float64)
        范 = float(np.linalg.norm(v))
        return v / 范 if 范 > 0.0 else v

    def 编码P(self, 窗=None) -> dict:
        """P := encode(Π)：P_k = sign(Σ_{c∈组} π_c·A_c)（D_P 维双极）；
        组空 ⇒ 零向量（depend——诚实 N/A）；零经历 ⇒ 全零（反硬编码）。
        窗=None 取最近窗；可传任意 Π dict（同一套预注册锚，供跨层一致检验）。"""
        π = self.剖面() if 窗 is None else 窗
        P = {}
        for gi, 组名 in enumerate(组角色序):
            if π is None:
                P["P_" + 组名] = np.zeros(D_P, dtype=np.float32)
                continue
            z = np.zeros(D_P, dtype=np.float64)
            for 名 in 编码分组[组名]:
                z = z + float(π[名]) * self.A_锚[名].astype(np.float64)
            P["P_" + 组名] = np.sign(z).astype(np.float32)
        return P


# ── §3 人格谱系（持久层身份链，E4） ─────────────────────────────────────────────────
class 人格谱系:
    """追加-only 链式 hash 身份链：H0=bytes(32)；
    H_i = sha256(H_{i-1} ‖ canonical(读出_i) ‖ seq_i ‖ 事件标记_i)。

    身份域排除（纪要 §2#5）：读出 dict 只承载 L4 机制读出（Π 机制分量/窗序/标记），
    seed/设备ID/图式hash/B7校准量/参数实时值 一律不得入读出 ⇒ 换 seed 重放同经历
    ⇒ 身份 hash 逐位不变（PERS_selftest [9] 可检）。
    检查点回滚 = 尾部追加「分叉事件」（回滚也是经历，不破坏追加纪律）。"""

    def __init__(self):
        self._链 = bytes(32)
        self.条目: list = []
        self.seq = 0

    def 追加(self, 读出dict, 事件标记: str = "窗") -> dict:
        文 = canonical(读出dict).encode("utf-8")
        if len(文) > 快照KB上限 * 1024:
            raise AssertionError("谱系快照 %d 字节 > %dKB 上限（预注册）"
                                 % (len(文), 快照KB上限))
        self._链 = hashlib.sha256(
            self._链 + 文 + str(self.seq).encode("ascii")
            + 事件标记.encode("utf-8")).digest()
        条 = {"seq": self.seq, "读出": 读出dict, "标记": 事件标记, "H": self._链.hex()}
        self.条目.append(条)
        self.seq += 1
        return 条

    def 校验(self) -> bool:
        """从头重算全链逐位比对 ⇒ bool（篡改历史任一条 ⇒ 必检出）。"""
        h = bytes(32)
        for 条 in self.条目:
            文 = canonical(条["读出"]).encode("utf-8")
            h = hashlib.sha256(h + 文 + str(条["seq"]).encode("ascii")
                               + 条["标记"].encode("utf-8")).digest()
            if h.hex() != 条["H"]:
                return False
        return True

    def 检查段载入(self, 段dict) -> None:
        """检查点段机器可检不变量（纪要 §2#5/#6）：段.seq ≤ 谱系尾.seq ∧ 段.身份键==尾链 hash。
        违例 fail-fast 拒载（先于任何内核状态恢复），不静默修补。"""
        尾 = self.seq - 1
        段seq = 段dict.get("seq")
        if not isinstance(段seq, int) or 段seq > 尾:
            raise ValueError("人格孤儿：段.seq=%r 不满足 ≤ 谱系尾.seq=%d（追加纪律违例，拒载）"
                             % (段seq, 尾))
        if str(段dict.get("身份键", "")) != self._链.hex():
            raise ValueError("人格孤儿：段.身份键 != 谱系尾链 hash（身份断裂，拒载不静默修补）")

    def 分叉事件(self, 理由: str) -> dict:
        """检查点回滚等结构分叉：追加空读出、标记="分叉:"+理由。"""
        return self.追加({}, "分叉:" + str(理由))

    def 身份hash(self) -> str:
        """当前链 H hex——换 seed 重放不变性的测量对象（seed 不在身份域）。"""
        return self._链.hex()
