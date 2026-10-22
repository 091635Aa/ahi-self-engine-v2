# -*- coding: utf-8 -*-
"""ahi_theory.ledger —— M4 账本（环形 4096 · 账本纪律 · C4 审计器 · K3 验账 · 结构事件 API）。

★★【M4 登记 · 2026-09-28】本文件为 V2 施工序列 M4 交付物（**纯新增**，依赖=无——
  仅标准库，零包内导入，不触碰 brain.py/mbp.py 等既有文件 ⇒ 锚点红线不受影响）。
  预登记：`ahi_theory/verify_all.py` `_CHANGELOG` 2026-09-28
  『M4 实装登记（预登记）』条目（先于改码登记）；同日『所有者裁决登记（零代码）』
  条目载明：施工顺序=先 M4 后 M3，O7 脊髓态=实施对齐。
  规格出处：`meeting11/95_架构定稿.md` §5-M4（行 466）+ 行 70/125/158/188/294/299/301/386/445–452。
  回退方式：删除本文件即回（零侵入、零既有路径引用）。

五件交付（与预登记一一对应）：
  ① 环形 4096 账本 —— `环形账本`：容量 4096、溢出=丢最旧+溢出计数（与 three_loop.有界队列
     同律）；写入率遥测伴生（`写入率(窗口秒)`，只覆盖容留窗，诚实边界见 docstring）。
  ② 账本纪律 —— 全部先验常数变更只能以账本条目（结构事件+理由）发生：
     kind="参数变更" 必须 payload["键"]∈`变更清单` 且理由非空；清单外/无理由 ⇒
     拒绝本体 + 强制登记"违规"条目 + 抛 `常数清单违规`。
  ③ C4 审计器 —— `C4审计`：grep 零依赖、纯 stdlib，对指定文件清单静态扫描
     lstsq / inv·pinv / solve / scipy.optimize / EMA 手调混合 / 参数搜索构造，
     要求出现次数=0（机器可检）。默认审计面 = **V2 新施工面**（本轮仅 ledger.py；
     M3/M5–M7 新文件加入时须以结构事件登记扩充——V1 内核 brain/mbp/modality 的
     闭式解是既有合法物，**不在**本判据面内）。兼职 WCET 静态审计：`WCET默认区段`
     对 three_loop.py R0 反射环类体扫描禁构（numpy/IO/容器增长/动态执行，定稿
     "R0 tick 路径可逐行静态审计"的机器化）。
  ④ K3 验账 —— `K3验账`：k_c=0 ⇒ 账本增量（总写入差分）必须为 0。本轮交付
     账本侧记录+断言；K3 判据的**引擎级测量**待 M6 供数后另行预注册（不虚设）。
  ⑤ 结构事件登记 API —— `环形账本.登记(kind, payload, 理由)`，kind 开放枚举；
     未来生产者例：K:8→4 超预算、未发育查询退化、offline 暂停、ACh 失配、
     M3 越级抑制/kill 演练事件。

诚实边界：
  · C4 为静态 grep 级审计——只能检出**具名构造**（lstsq/inv/pinv/solve/optimize/
    EMA 标识/参数搜索惯用名），不能证明语义无罪；其价值在"机器可检的必要条件"，
    非充分条件（登记于 CRITERIA 语义，不冒充形式化证明）。
  · 写入率遥测基于容留条目的单调钟时间戳：溢出丢弃的条目不参与窗口统计
    （总写入/溢出计数不受此限，二者才是完备口径）。
  · 审计为 **tokenize 代码级**（STRING/COMMENT 剔除）：模式定义行与 docstring 字样
    天然不参与匹配，无自指误报；代价是 f-string 内插与跨行拆写不被扫描（登记性漏报）。
  · 线程安全：threading.Lock 单锁；deque(maxlen) 承担环形语义。
"""
from __future__ import annotations

import io
import re
import threading
import time
import tokenize
from collections import deque

# ══════════════════════════════════════════════════════════════════════════
# 预注册常数（变更须走账本条目：kind="参数变更" 且 键∈变更清单）
# ══════════════════════════════════════════════════════════════════════════
#: 账本容量（定稿 §5-M4：环形 4096）
账本容量 = 4096

#: 变更清单（预注册于 _CHANGELOG 2026-09-28 M4 条目；**扩充清单本身 = 结构事件**）
变更清单 = frozenset({
    "γ_H", "γ_S", "λ_M", "η尺度", "η₀", "γ_e",           # E3 先验（学习率/动量学/尺度）
    "θ_w", "ε_f", "C", "压平份额阈值",                     # 稳态律/门控/压平
    "D", "|IM|", "W_o", "R_reg", "δ_dead", "β带",          # 维度/记忆/窗/ Registrar/死区/带
})

#: 已知 kind（**开放枚举**——任意字符串可登记；仅"参数变更"受清单纪律约束，
#: "违规"为审计/纪律副产物保留 kind）
保留kind = frozenset({"参数变更", "违规"})


class 常数清单违规(ValueError):
    """参数变更违反账本纪律（键不在变更清单 / 缺少理由）。"""


# ══════════════════════════════════════════════════════════════════════════
# ① 环形账本 + ② 账本纪律 + ⑤ 结构事件 API
# ══════════════════════════════════════════════════════════════════════════
class 环形账本:
    """线程安全环形账本（容量 4096，溢出=丢最旧+计数）。

    条目 schema（E4_Round3 §2#12 精神：字段机器可查，语义 E5 出）：
        {"seq": 单调写入序号（含被溢出丢弃者——K3 验账口径）,
         "t":   单调钟时刻（写入率遥测用）,
         "kind":条目种类（开放枚举）,
         "payload": 任意可序列化载荷（参数变更须含 "键" 字段）,
         "理由": 非空字符串（参数变更强制）}
    """

    def __init__(self, 容量: int = 账本容量, 时钟=None):
        self.容量 = int(容量)
        self._q: deque = deque(maxlen=self.容量)
        self._锁 = threading.Lock()
        self._时钟 = 时钟 if 时钟 is not None else time.monotonic
        self.总写入 = 0            # 单调计数（含被溢出丢弃者）——K3 增量口径
        self.溢出 = 0              # 丢弃计数（丢最旧的条数）

    # ---- 登记（⑤ 结构事件 API 与 ② 纪律守卫的统一入口）----
    def 登记(self, kind: str, payload=None, 理由: str = "", 强制: bool = False) -> dict:
        """登记一条账本条目并返回它。

        kind="参数变更"（账本纪律核心）：必须 理由非空 且 payload["键"]∈变更清单；
        违反 ⇒ 先强制登记一条"违规"条目（留痕），再抛 `常数清单违规`，
        参数变更本体**不入账**。强制=True 跳过纪律检查（仅供违规登记自用）。
        其余 kind（结构事件/演练/windup_event/…开放枚举）不设限。
        """
        if kind == "参数变更" and not 强制:
            键 = payload.get("键") if isinstance(payload, dict) else None
            if not str(理由).strip():
                self.登记("违规",
                          payload={"拒绝的参数变更": payload, "键": 键, "原因": "缺少理由"},
                          理由="账本纪律拒绝（参数变更必须携带理由）", 强制=True)
                raise 常数清单违规("参数变更缺少理由（账本纪律：变更=结构事件+理由）")
            if 键 not in 变更清单:
                self.登记("违规",
                          payload={"拒绝的参数变更": payload, "键": 键, "原因": "键不在变更清单"},
                          理由="账本纪律拒绝（清单外先验常数不得变更）", 强制=True)
                raise 常数清单违规("参数变更键 %r 不在变更清单（账本纪律：清单外拒绝）" % (键,))
        with self._锁:
            now = self._时钟()
            if len(self._q) >= self.容量:      # 环形满 ⇒ 丢最旧（deque popleft 语义）
                self.溢出 += 1
            条目 = {"seq": self.总写入, "t": now, "kind": kind,
                    "payload": payload, "理由": 理由}
            self._q.append(条目)
            self.总写入 += 1
        return 条目

    # ---- 读侧 ----
    def 容留数(self) -> int:
        with self._锁:
            return len(self._q)

    def 最近(self, n: int = 1) -> list:
        """最近 n 条（旧→新）。"""
        with self._锁:
            return list(self._q)[-int(n):]

    def 导出(self) -> list:
        """容留条目全量快照（浅拷贝列表，旧→新）。"""
        with self._锁:
            return list(self._q)

    def 写入率(self, 窗口秒: float = 60.0) -> int:
        """最近 `窗口秒` 内的登记条数（**只覆盖容留窗**——溢出丢弃者不计入，
        完备口径看 总写入/溢出）。"""
        now = self._时钟()
        下界 = now - float(窗口秒)
        with self._锁:
            return sum(1 for e in self._q if e["t"] >= 下界)

    def 统计(self) -> dict:
        """账本状态摘要（容量/容留/总写入/溢出/kind 分布）。"""
        with self._锁:
            分布: dict = {}
            for e in self._q:
                分布[e["kind"]] = 分布.get(e["kind"], 0) + 1
            return dict(容量=self.容量, 容留=len(self._q), 总写入=self.总写入,
                        溢出=self.溢出, kind分布=分布)


# ══════════════════════════════════════════════════════════════════════════
# ③ C4 审计器（grep 零依赖，纯 stdlib）
# ══════════════════════════════════════════════════════════════════════════
#: C4 禁构模式表（具名，机器可检；**代码 token 级扫描**——STRING/COMMENT 被
#: tokenize 剔除，模式定义行与 docstring 字样天然不参与匹配，无自指误报）
C4模式表 = (
    ("lstsq",          r"\blstsq\s*\("),
    ("inv_pinv",       r"linalg\.(?:inv|pinv)\s*\(|\.pinv\s*\("),
    ("solve",          r"linalg\.solve\s*\("),
    ("scipy_optimize", r"scipy\.optimize|from\s+scipy\s+import\s+optimize"),
    ("EMA手调混合",    r"\bema\w*\s*=[^=\n]*\b0\.\d|0\.\d+\s*\*\s*\w+\s*\+\s*\(\s*1\s*-\s*0\.\d+\s*\)"),
    ("参数搜索构造",   r"optuna|param_grid|ParameterGrid|itertools\.product"),
)

#: WCET 默认区段（定稿：R0 tick 路径"纯标量、无 numpy、无内存分配、无 IO——可逐行
#: 静态审计"的机器化。区段 = three_loop.py 的 R0反射环类体）
WCET默认区段 = {
    "three_loop.py": dict(
        起始锚="class R0反射环", 结束锚="class 钳制代理",
        禁构=(
            ("numpy",       r"np\.|numpy"),
            ("IO",          r"open\s*\(|print\s*\(|json\."),
            ("容器增长",    r"\.append\s*\(|\.extend\s*\("),
            ("动态执行",    r"\bexec\s*\(|\beval\s*\("),
        ),
    ),
}

#: C4 默认审计面 = V2 新施工面（本轮仅 M4 本体；M3/M5–M7 新文件加入时须以
#: 结构事件登记扩充。V1 内核 brain/mbp/modality 的闭式解为既有合法物，不在此列）
C4默认文件清单 = ("ledger.py", "supervisor.py")


def _代码行(路径: str) -> dict:
    """tokenize 级代码提取：返回 {行号: 该行代码 token 连接文本}。

    只保留 NAME/OP/NUMBER/KEYWORD（STRING/COMMENT/NL/INDENT 等剔除）——
    审计**代码**而非文字：docstring/注释里的模式字样（含本文件自身的
    模式表定义行）天然不参与匹配，无自指误报。
    连接规则：词 token（NAME/KEYWORD/NUMBER）之间补空格（保住 `class R0反射环`
    与 `from scipy import optimize` 的可匹配性），OP 紧贴（保住 `np.linalg.inv(`
    的可匹配性）。
    tokenize 失败 ⇒ 直接抛（审计器不许静默）。
    诚实边界：f-string 内插代码与跨行拆写的调用不被扫描（登记性漏报）。
    """
    词类 = {tokenize.NAME, tokenize.NUMBER}   # 关键字在 tokenize 中即 NAME 类型
    行文: dict = {}
    with open(路径, "r", encoding="utf-8") as f:
        保留 = 词类 | {tokenize.OP}
        for tok in tokenize.generate_tokens(f.readline):
            if tok.type not in 保留:
                continue
            前 = 行文.get(tok.start[0], "")
            if 前 and 前[-1].isalnum() and tok.type in 词类 and 前[-1] != "_":
                文 = 前 + " " + tok.string       # 词-词相接补空格
            elif 前 and 前[-1] == "_" and tok.type in 词类:
                文 = 前 + " " + tok.string       # 下划线结尾的词后再接词也补空格
            else:
                文 = 前 + tok.string
            行文[tok.start[0]] = 文
    return 行文


def C4审计(目录: str, 文件清单=C4默认文件清单, 区段规则: dict = None) -> dict:
    """C4 静态审计：对 `目录` 下 `文件清单` 逐文件扫描 `C4模式表` 禁构。

    返回 dict：
        文件结果 = {文件名: {模式名: [命中行号…]}}
        总命中   = Σ 命中数（要求 = 0）
        区段结果 = {文件名: {禁构名: [命中行号…]}}（区段规则=None 时用 WCET默认区段）
    扫描均为代码 token 级（见 _代码行）；任何文件读/解析失败 ⇒ 直接抛（不许静默）。
    """
    区段规则 = WCET默认区段 if 区段规则 is None else 区段规则
    文件结果: dict = {}
    总命中 = 0
    for 名 in 文件清单:
        行文 = _代码行(目录 + "/" + 名)
        命中: dict = {}
        for 模式名, 模式 in C4模式表:
            rx = re.compile(模式)
            行号 = sorted(i for i, 文 in 行文.items() if rx.search(文))
            if 行号:
                命中[模式名] = 行号
        文件结果[名] = 命中
        总命中 += sum(len(v) for v in 命中.values())
    # —— WCET 区段兼职 ——
    区段结果: dict = {}
    for 名, 规则 in 区段规则.items():
        行文 = _代码行(目录 + "/" + 名)
        起 = 终 = None
        for i in sorted(行文):
            if 起 is None and 规则["起始锚"] in 行文[i]:
                起 = i
            elif 起 is not None and 规则["结束锚"] in 行文[i]:
                终 = i
                break
        if 起 is None or 终 is None:
            区段结果[名] = dict(错误="区段锚未找到", 起始锚=规则["起始锚"], 结束锚=规则["结束锚"])
            continue
        段命中: dict = {}
        for 禁构名, 模式 in 规则["禁构"]:
            rx = re.compile(模式)
            行号 = sorted(i for i in 行文 if 起 < i < 终 and rx.search(行文[i]))
            if 行号:
                段命中[禁构名] = 行号
        区段结果[名] = dict(区段行=(起, 终), 命中=段命中,
                            总命中=sum(len(v) for v in 段命中.values()))
    return dict(文件结果=文件结果, 总命中=总命中, 区段结果=区段结果)


# ══════════════════════════════════════════════════════════════════════════
# ④ K3 验账（账本侧；引擎级测量待 M6 供数后另行预注册）
# ══════════════════════════════════════════════════════════════════════════
def K3验账(账本: 环形账本, 基准总写入: int, k_c: int, 断言: bool = False) -> dict:
    """K3 验账：k_c=0 ⇒ 账本增量（总写入差分）必须为 0。

    k_c = 允许的集中式参数调节通道数（V2 终态 = 0：一切变更走账本结构事件）。
    k_c>0 时不设限（过渡期）。断言=True 且不通过 ⇒ raise AssertionError。
    增量口径用 `总写入`（单调、含溢出丢弃者）——与容留数无关。
    """
    增量 = 账本.总写入 - int(基准总写入)
    通过 = (int(k_c) > 0) or (增量 == 0)
    结果 = dict(k_c=int(k_c), 基准总写入=int(基准总写入),
                当前总写入=账本.总写入, 增量=增量, 通过=bool(通过))
    if 断言 and not 通过:
        raise AssertionError("K3 验账失败：k_c=0 但账本增量=%d ≠ 0" % 增量)
    return 结果
