# -*- coding: utf-8 -*-
"""ahi_theory/gpu_path.py —— G0a · GPU 迁移探针（影子模式）

预登记：ahi_theory/verify_all.py `_CHANGELOG` #82（2026-10-01，先于改码）。
规格出处：meeting16/00_GPU迁移路线.md（2026-10-01 所有者裁决：GPU=目标推理口径，CPU=降级方案；
热点算子天然是 GEMM 形——迁移不是重写，是换后端）。

影子纪律（#82 冻结）：
  · 纯新增；不 import brain/world/system/three_loop/mbp（影子期零侵入）；
    只 import numpy/torch/ahi_theory.hdc_core（仅取预注册常量与真实算子形状，不取运行态状态）。
  · 零拟合红线：无 lstsq/linalg.inv·pinv/solve/scipy.optimize/参数搜索构造（C4 tokenize 可检）。
  · 探针只测速与验证确定性策略，**不产出任何判据判定值**（判据/阈值零改动）。
  · 全部 float32 口径（与现有 numpy 实现一致）；显存不足的算子自动降形并如实注明。
  · GPU 浮点差异是预期内的：CPU↔GPU 差异数值直接作为迁移路线 §2 确定性策略的
    容差预注册依据（如 max|Δ|≤1e-5 相对），差异大不是 FAIL、是标定数据。

确定性策略实测项（路线 §2）：
  ① GPU 口径内自一致（同设备同算法双跑逐位）；
  ② CPU↔GPU 数值差异界（连续量绝对/相对 + sign 弛豫终态不一致占比）；
  ③ torch.use_deterministic_algorithms(True) 双跑复测；
  ④ cudnn.benchmark=False + 单流（默认流）执行。

热点算子（路线 §3 清单，全部真实算子形状、零判定值）：
  1) W_selfHop 弛豫（hdc_core step8）：D=1024 随机 W(f32) @ S̃，×5 连续 sign 弛豫；
  2) M 读 GEMM（hdc_core _M读 主项）：稠密基 (1024,1024) @ 键矩阵 (1024,10)（Q=10 候选）+ 衰减标量乘；
  3) lookahead 批量（selfthink step6 扩展形态）：9 候选 × 弛豫 5 次——循环实现与批量 GEMM 实现对比；
  4) K5 批量模拟：40 实现 × 131072 点 fGn 生成 + DFA 的 GEMM 主项代理
     （分窗 Vandermonde 设计批量 XᵀX/Xᵀy——**代理非 DFA 全链**：无积分/无波动拟合/无统计量，
     不产出任何 α 读数）。

CPU 口径=单线程 BLAS：进程级 env 在本模块顶部（numpy 导入前）设置；
若宿主在导入本模块前已导入 numpy 则不生效，输出中如实标注该状态。
"""
from __future__ import annotations

import os

# —— CPU 单线程 BLAS 口径 + cublas 确定性工作区（必须在 numpy/torch 导入前设置） ——
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # use_deterministic_algorithms 前置

import time
from typing import Optional

import numpy as np
import torch

from ahi_theory.hdc_core import D_VSA, C_弛豫, MASTER_SEED   # 预注册常量（形状/种子口径）

__all__ = ["GPU探针"]

_预热 = 10          # 预注册：预热 10 次后计时
_单算子预算ms = 6000.0   # 自适应：单算子计时总预算（ms），迭代数向下取整至预算内


class GPU探针:
    """G0a GPU 迁移探针：环境盘点 / 热点算子原型测速 / 确定性策略实证（零判定值）。"""

    def __init__(self, 设备: str = "cuda"):
        self.设备 = 设备
        self._可用 = bool(torch.cuda.is_available())
        if self._可用:
            self._dev = torch.device(设备)
        else:
            self._dev = torch.device("cpu")

    # —— 内部辅助 ——
    @staticmethod
    def _双极(rng: np.random.Generator, n) -> np.ndarray:
        """双极 ±1 向量（float32；n 可为 int 或 shape 元组；与 hdc_core._双极 同口径）。"""
        return np.sign(rng.standard_normal(n)).astype(np.float32)

    def _显存峰值mib(self) -> float:
        if not self._可用:
            return 0.0
        return round(torch.cuda.max_memory_allocated(self._dev) / 1048576.0, 1)

    def _降形(self, 需要字节: int, 系数: float = 0.6) -> tuple[bool, float]:
        """显存不足⇒True（调用方降形并注明）；返回 (是否降形, 可用显存MiB)。"""
        if not self._可用:
            return False, 0.0
        空闲, _总 = torch.cuda.mem_get_info(self._dev)
        return bool(需要字节 > 空闲 * 系数), round(空闲 / 1048576.0, 1)

    @staticmethod
    def _定迭代(上限: int, 预估单次ms: float) -> int:
        """迭代数自适应：单算子计时总时长 ≤ _单算子预算ms（下限 5）。"""
        if 预估单次ms <= 0:
            return 上限
        return int(max(5, min(上限, _单算子预算ms / 预估单次ms)))

    # —— [1] 环境（只读盘点，无判定值） ——
    def 环境(self) -> dict:
        d: dict = {
            "torch版本": torch.__version__,
            "torch编译CUDA": torch.version.cuda,
            "cudnn版本": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cuda可用": self._可用,
            "numpy版本": np.__version__,
            "cpu单线程BLAS_env": {k: os.environ.get(k) for k in
                                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cpu口径生效说明": "进程级 env 于本模块导入时（numpy 导入前）设置；"
                              "若宿主此前已导入 numpy 则 env 不生效，以本 dict 为准如实登记",
        }
        if self._可用:
            p = torch.cuda.get_device_properties(0)
            空闲, 总 = torch.cuda.mem_get_info(0)
            d.update({
                "设备名": p.name,
                "计算能力": f"{p.major}.{p.minor}",
                "SM数": int(p.multi_processor_count),
                "显存总量MiB": round(总 / 1048576.0, 1),
                "显存当前空闲MiB": round(空闲 / 1048576.0, 1),
                "显存当前占用MiB": round((总 - 空闲) / 1048576.0, 1),
                "当前流": str(torch.cuda.current_stream(0)),
                "cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
                "确定性算法开关(当前)": bool(torch.are_deterministic_algorithms_enabled()),
            })
        return d

    # —— [2] 热点算子测速（原型形态；CPU 单线程 BLAS vs torch-cuda） ——
    def 算子测速(self, 迭代: int = 100) -> dict:
        rng = np.random.default_rng(MASTER_SEED)          # 预注册 master 种子（形状构造）
        out: dict = {"口径": "float32；CPU=单线程 BLAS；GPU=torch-cuda（预热10、synchronize 后计时，均值/p99）",
                     "零判定值声明": "本节只产出耗时/加速比/显存峰值，无任何判据读数"}
        算子 = (
            ("1_W_selfHop弛豫", self._op1构造, self._op1测速),
            ("2_M读GEMM", self._op2构造, self._op2测速),
            ("3_lookahead批量", self._op3构造, self._op3测速),
            ("4_K5批量模拟_代理", self._op4构造, self._op4测速),
        )
        for 名, 构造, 测速 in 算子:
            t0 = time.perf_counter()
            out[名] = 测速(*构造(rng), 迭代上限=int(迭代))
            out[名]["构造耗时秒"] = round(time.perf_counter() - t0, 2)
        return out

    # —— 算子 1：W_selfHop 弛豫（hdc_core step8 形态） ——
    def _op1构造(self, rng):
        W = rng.standard_normal((D_VSA, D_VSA)).astype(np.float32)
        S = self._双极(rng, D_VSA)
        return W, S

    def _op1测速(self, W, S, 迭代上限: int) -> dict:
        明细: dict = {"形态": f"({D_VSA},{D_VSA})f32 @ ({D_VSA},)f32，×{C_弛豫} 连续 sign 弛豫（hdc_core step8）"}

        def cpu一次() -> np.ndarray:
            v = S
            for _ in range(C_弛豫):
                v = np.sign(W @ v)
            return v

        # CPU 样本
        for _ in range(_预热):
            cpu一次()
        t0 = time.perf_counter_ns()
        cpu一次()
        预估 = (time.perf_counter_ns() - t0) / 1e6
        n = self._定迭代(迭代上限, 预估)
        样本 = []
        for _ in range(n):
            t0 = time.perf_counter_ns()
            cpu一次()
            样本.append((time.perf_counter_ns() - t0) / 1e6)
        明细["CPU"] = {"迭代": n, "均值ms": round(float(np.mean(样本)), 4),
                       "p99ms": round(float(np.percentile(样本, 99)), 4)}
        # GPU 样本
        if self._可用:
            Wg = torch.from_numpy(W).to(self._dev)
            Sg = torch.from_numpy(S).to(self._dev)

            def gpu一次() -> torch.Tensor:
                v = Sg
                for _ in range(C_弛豫):
                    v = torch.sign(Wg @ v)
                return v

            torch.cuda.synchronize(self._dev)
            torch.cuda.reset_peak_memory_stats(self._dev)
            for _ in range(_预热):
                gpu一次()
            torch.cuda.synchronize(self._dev)
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            样本g = []
            for _ in range(n):
                ev0.record()
                gpu一次()
                ev1.record()
                torch.cuda.synchronize(self._dev)
                样本g.append(ev0.elapsed_time(ev1))
            明细["GPU"] = {"迭代": n, "均值ms": round(float(np.mean(样本g)), 4),
                           "p99ms": round(float(np.percentile(样本g, 99)), 4)}
            明细["加速比_CPU/GPU"] = round(明细["CPU"]["均值ms"] / max(明细["GPU"]["均值ms"], 1e-9), 1)
            明细["显存峰值MiB"] = self._显存峰值mib()
        else:
            明细["GPU"] = None
            明细["加速比_CPU/GPU"] = None
            明细["显存峰值MiB"] = None
        return 明细

    # —— 算子 2：M 读 GEMM（hdc_core _M读 主项形态） ——
    def _op2构造(self, rng):
        M = rng.standard_normal((D_VSA, D_VSA)).astype(np.float32)
        键 = rng.standard_normal((D_VSA, 10)).astype(np.float32)     # Q=10 候选（1 主预测+9 动作）
        折 = np.float32(0.999)                                        # 衰减标量（λ_M 口径的代理值）
        return M, 键, 折

    def _op2测速(self, M, 键, 折, 迭代上限: int) -> dict:
        明细: dict = {"形态": f"稠密基({D_VSA},{D_VSA})f32 @ 键矩阵({D_VSA},10)f32 + 衰减标量乘"
                             "（hdc_core _M读 主项；秩1缓冲小侧项不计入本主项代理）"}

        def cpu一次() -> np.ndarray:
            return (M @ 键) * 折

        for _ in range(_预热):
            cpu一次()
        t0 = time.perf_counter_ns()
        cpu一次()
        预估 = (time.perf_counter_ns() - t0) / 1e6
        n = self._定迭代(迭代上限, 预估)
        样本 = []
        for _ in range(n):
            t0 = time.perf_counter_ns()
            cpu一次()
            样本.append((time.perf_counter_ns() - t0) / 1e6)
        明细["CPU"] = {"迭代": n, "均值ms": round(float(np.mean(样本)), 4),
                       "p99ms": round(float(np.percentile(样本, 99)), 4)}
        if self._可用:
            Mg = torch.from_numpy(M).to(self._dev)
            Kg = torch.from_numpy(键).to(self._dev)
            折g = torch.tensor(float(折), device=self._dev)

            def gpu一次() -> torch.Tensor:
                return (Mg @ Kg) * 折g

            torch.cuda.synchronize(self._dev)
            torch.cuda.reset_peak_memory_stats(self._dev)
            for _ in range(_预热):
                gpu一次()
            torch.cuda.synchronize(self._dev)
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            样本g = []
            for _ in range(n):
                ev0.record()
                gpu一次()
                ev1.record()
                torch.cuda.synchronize(self._dev)
                样本g.append(ev0.elapsed_time(ev1))
            明细["GPU"] = {"迭代": n, "均值ms": round(float(np.mean(样本g)), 4),
                           "p99ms": round(float(np.percentile(样本g, 99)), 4)}
            明细["加速比_CPU/GPU"] = round(明细["CPU"]["均值ms"] / max(明细["GPU"]["均值ms"], 1e-9), 1)
            明细["显存峰值MiB"] = self._显存峰值mib()
        else:
            明细["GPU"] = None
            明细["加速比_CPU/GPU"] = None
            明细["显存峰值MiB"] = None
        return 明细

    # —— 算子 3：lookahead 批量（selfthink step6 扩展形态：9 候选×弛豫 5） ——
    def _op3构造(self, rng):
        W = rng.standard_normal((D_VSA, D_VSA)).astype(np.float32)
        批 = self._双极(rng, (9, D_VSA))                              # (9,1024) 候选起始态
        return W, 批

    def _op3测速(self, W, 批, 迭代上限: int) -> dict:
        明细: dict = {"形态": f"9 候选 × {C_弛豫} 次 ({D_VSA},{D_VSA})f32 弛豫"
                             "（selfthink step6 扩展形态；循环实现与批量 GEMM 实现对比）"}

        def cpu循环一次() -> np.ndarray:
            out = np.empty_like(批)
            for c in range(批.shape[0]):
                v = 批[c]
                for _ in range(C_弛豫):
                    v = np.sign(W @ v)
                out[c] = v
            return out

        def cpu批量一次() -> np.ndarray:
            v = 批[:, :, None]                                        # (9,1024,1)
            for _ in range(C_弛豫):
                v = np.sign(np.matmul(W[None, :, :], v))              # 广播批量 GEMM
            return v[:, :, 0]

        # CPU 循环
        for _ in range(_预热):
            cpu循环一次()
        t0 = time.perf_counter_ns()
        cpu循环一次()
        预估 = (time.perf_counter_ns() - t0) / 1e6
        n = self._定迭代(迭代上限, 预估)
        样本 = []
        for _ in range(n):
            t0 = time.perf_counter_ns()
            cpu循环一次()
            样本.append((time.perf_counter_ns() - t0) / 1e6)
        明细["CPU_循环"] = {"迭代": n, "均值ms": round(float(np.mean(样本)), 4),
                            "p99ms": round(float(np.percentile(样本, 99)), 4)}
        # CPU 批量
        for _ in range(_预热):
            cpu批量一次()
        样本 = []
        for _ in range(n):
            t0 = time.perf_counter_ns()
            cpu批量一次()
            样本.append((time.perf_counter_ns() - t0) / 1e6)
        明细["CPU_批量GEMM"] = {"迭代": n, "均值ms": round(float(np.mean(样本)), 4),
                                "p99ms": round(float(np.percentile(样本, 99)), 4)}
        if self._可用:
            # 显存降形检查：批量 (9,1024,1024) f32 ≈ 37.7 MiB
            降, 空闲 = self._降形(9 * D_VSA * D_VSA * 4 * 2)
            明细["显存降形"] = False
            批数 = 9
            if 降:
                批数 = 4
                明细["显存降形"] = True
                明细["降形注明"] = f"可用显存不足 ⇒ 9 候选降形为 {批数} 候选（可用 {空闲} MiB）"
            批g = torch.from_numpy(批[:批数].copy()).to(self._dev)
            Wg = torch.from_numpy(W).to(self._dev)

            def gpu循环一次() -> torch.Tensor:
                out = torch.empty_like(批g)
                for c in range(批g.shape[0]):
                    v = 批g[c]
                    for _ in range(C_弛豫):
                        v = torch.sign(Wg @ v)
                    out[c] = v
                return out

            def gpu批量一次() -> torch.Tensor:
                v = 批g[:, :, None]
                for _ in range(C_弛豫):
                    v = torch.sign(torch.matmul(Wg.unsqueeze(0), v))
                return v[:, :, 0]

            torch.cuda.synchronize(self._dev)
            torch.cuda.reset_peak_memory_stats(self._dev)
            for _ in range(_预热):
                gpu循环一次()
                gpu批量一次()
            torch.cuda.synchronize(self._dev)
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            样本l, 样本b = [], []
            for _ in range(n):
                ev0.record()
                gpu循环一次()
                ev1.record()
                torch.cuda.synchronize(self._dev)
                样本l.append(ev0.elapsed_time(ev1))
            for _ in range(n):
                ev0.record()
                gpu批量一次()
                ev1.record()
                torch.cuda.synchronize(self._dev)
                样本b.append(ev0.elapsed_time(ev1))
            明细["GPU_循环"] = {"迭代": n, "均值ms": round(float(np.mean(样本l)), 4),
                                "p99ms": round(float(np.percentile(样本l, 99)), 4)}
            明细["GPU_批量GEMM"] = {"迭代": n, "均值ms": round(float(np.mean(样本b)), 4),
                                    "p99ms": round(float(np.percentile(样本b, 99)), 4)}
            明细["加速比_循环CPU/批量GPU"] = round(
                明细["CPU_循环"]["均值ms"] / max(明细["GPU_批量GEMM"]["均值ms"], 1e-9), 1)
            明细["加速比_批量GPU/循环GPU"] = round(
                明细["GPU_循环"]["均值ms"] / max(明细["GPU_批量GEMM"]["均值ms"], 1e-9), 1)
            明细["显存峰值MiB"] = self._显存峰值mib()
        else:
            明细["GPU_循环"] = None
            明细["GPU_批量GEMM"] = None
            明细["加速比_循环CPU/批量GPU"] = None
            明细["加速比_批量GPU/循环GPU"] = None
            明细["显存峰值MiB"] = None
        return 明细

    # —— 算子 4：K5 批量模拟（40 实现 × 131072 点；GEMM 主项代理） ——
    def _op4构造(self, rng):
        B, T = 40, 131072
        需 = B * T * 4 + B * 128 * 1024 * 3 * 4
        降, 空闲 = self._降形(需)
        if 降 and self._可用:
            B = 20
        X = np.power(np.arange(1024, dtype=np.float32)[:, None] / np.float32(1023.0),
                     np.arange(3, dtype=np.float32)[None, :])        # (1024,3) Vandermonde 设计
        return rng, B, T, X

    def _op4测速(self, rng, B, T, X, 迭代上限: int) -> dict:
        窗, p = 1024, 3
        窗数 = T // 窗                                               # 128
        明细: dict = {
            "形态": f"B={B} 实现 × T={T} 点 fGn 生成 + 分窗({窗}) Vandermonde(p={p}) 批量 XᵀX/Xᵀy GEMM"
                    "——**GEMM 主项代理，非 DFA 全链**（无积分/无逐窗波动拟合/无任何统计量与 α 读数）",
        }
        if B != 40:
            明细["显存降形"] = True
            明细["降形注明"] = f"显存不足 ⇒ 实现数 40 降形为 {B}"

        def cpu一次():
            噪 = rng.standard_normal((B, T)).astype(np.float32)       # 白噪声生成（fGn 谱整形不计入代理）
            系 = 噪.reshape(B, 窗数, 窗, 1)
            Xb = np.broadcast_to(X[None, None], (B, 窗数, 窗, p))
            XtX = np.matmul(Xb.transpose(0, 1, 3, 2), Xb)             # (B,窗数,3,3)
            Xty = np.matmul(Xb.transpose(0, 1, 3, 2), 系)             # (B,窗数,3,1)
            return XtX, Xty

        for _ in range(_预热):
            cpu一次()
        t0 = time.perf_counter_ns()
        cpu一次()
        预估 = (time.perf_counter_ns() - t0) / 1e6
        n = self._定迭代(迭代上限, 预估)
        样本 = []
        for _ in range(n):
            t0 = time.perf_counter_ns()
            cpu一次()
            样本.append((time.perf_counter_ns() - t0) / 1e6)
        明细["CPU"] = {"迭代": n, "均值ms": round(float(np.mean(样本)), 4),
                       "p99ms": round(float(np.percentile(样本, 99)), 4)}
        # 生成与 GEMM 分项（CPU）
        t0 = time.perf_counter_ns()
        rng.standard_normal((B, T)).astype(np.float32)
        明细["CPU_仅生成ms"] = round((time.perf_counter_ns() - t0) / 1e6, 4)
        if self._可用:
            Xg = torch.from_numpy(X).to(self._dev)                    # (窗,p)

            def gpu一次():
                噪 = torch.randn((B, T), device=self._dev, dtype=torch.float32)
                系 = 噪.view(B, 窗数, 窗, 1)                          # (B,窗数,窗,1)
                Xe = Xg.view(1, 1, 窗, p).expand(B, 窗数, 窗, p)      # (B,窗数,窗,p)=X 广播
                XeT = Xe.transpose(2, 3)                              # (B,窗数,p,窗)=Xᵀ
                XtX = torch.matmul(XeT, Xe)                           # (B,窗数,3,3)
                Xty = torch.matmul(XeT, 系)                           # (B,窗数,3,1)
                return XtX, Xty

            torch.cuda.synchronize(self._dev)
            torch.cuda.reset_peak_memory_stats(self._dev)
            for _ in range(_预热):
                gpu一次()
            torch.cuda.synchronize(self._dev)
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            样本g = []
            for _ in range(n):
                ev0.record()
                gpu一次()
                ev1.record()
                torch.cuda.synchronize(self._dev)
                样本g.append(ev0.elapsed_time(ev1))
            明细["GPU"] = {"迭代": n, "均值ms": round(float(np.mean(样本g)), 4),
                           "p99ms": round(float(np.percentile(样本g, 99)), 4)}
            明细["加速比_CPU/GPU"] = round(明细["CPU"]["均值ms"] / max(明细["GPU"]["均值ms"], 1e-9), 1)
            明细["显存峰值MiB"] = self._显存峰值mib()
        else:
            明细["GPU"] = None
            明细["加速比_CPU/GPU"] = None
            明细["显存峰值MiB"] = None
        return 明细

    # —— [3] 确定性验证（逐项布尔+数值，如实报告；零判定值） ——
    def 确定性验证(self) -> dict:
        rng = np.random.default_rng(MASTER_SEED + 1)                  # 独立派生（预注册）
        out: dict = {"口径": "float32；差异如实报告——GPU 浮点差异是预期内的，"
                            "数值进迁移路线 §2 容差预注册依据（CPU↔GPU 差异大不是 FAIL）"}
        W = rng.standard_normal((D_VSA, D_VSA)).astype(np.float32)
        S = self._双极(rng, D_VSA)
        M = rng.standard_normal((D_VSA, D_VSA)).astype(np.float32)
        键 = rng.standard_normal((D_VSA, 10)).astype(np.float32)
        批 = self._双极(rng, (9, D_VSA))

        def 弛豫链_np(Wn, Sn, 轮=2):
            """×轮 次完整 5 步弛豫链（终态+首步连续量）。"""
            v = Sn
            z1 = Wn @ Sn
            for _ in range(C_弛豫):
                v = np.sign(Wn @ v)
            return z1, v

        if not self._可用:
            out["GPU可用"] = False
            out["说明"] = "CUDA 不可用 ⇒ 确定性验证不适用，如实登记（不虚设）"
            return out

        Wg = torch.from_numpy(W).to(self._dev)
        Sg = torch.from_numpy(S).to(self._dev)
        Mg = torch.from_numpy(M).to(self._dev)
        Kg = torch.from_numpy(键).to(self._dev)
        批g = torch.from_numpy(批).to(self._dev)

        def op_w(g):                                                  # 弛豫链（GPU）
            v = Sg
            for _ in range(C_弛豫):
                v = torch.sign(Wg @ v)
            return v

        def op_m(g):
            return (Mg @ Kg)

        def op_批(g):
            v = 批g[:, :, None]
            for _ in range(C_弛豫):
                v = torch.sign(torch.matmul(Wg.unsqueeze(0), v))
            return v[:, :, 0]

        算子表 = (("1_W_selfHop弛豫", op_w), ("2_M读GEMM", op_m), ("3_lookahead批量", op_批))

        # ① GPU 双跑逐位一致（同输入同算子跑两次）
        项1: dict = {}
        for 名, fn in 算子表:
            a, b = fn(None), fn(None)
            torch.cuda.synchronize(self._dev)
            差 = (a - b).abs().max().item() if a.numel() else 0.0
            逐位 = bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))
            项1[名] = {"双跑逐位一致": 逐位, "max绝对差": float(差)}
        out["①_GPU双跑逐位一致"] = 项1

        # ② CPU↔GPU 差异（同输入 numpy vs torch-cuda；绝对+相对+sign 终态不一致占比）
        项2: dict = {}
        z1c, 链c = 弛豫链_np(W, S)
        链g = op_w(None)
        差z = float((torch.from_numpy(z1c).to(self._dev) - (Wg @ Sg)).abs().max().item())
        项2["1_W_selfHop弛豫"] = {
            "连续量W@S_max绝对差": 差z,
            "相对量": round(差z / max(float(np.abs(z1c).max()), 1e-12), 9),
            "sign终态max绝对差": float((torch.from_numpy(链c).to(self._dev) - 链g).abs().max().item()),
            "sign终态不一致元素数": int((torch.from_numpy(链c).to(self._dev) != 链g).sum().item()),
            "sign终态不一致占比": round(float((torch.from_numpy(链c).to(self._dev) != 链g)
                                              .float().mean().item()), 9),
        }
        mc = M @ 键
        mg = op_m(None)
        差m = float((torch.from_numpy(mc).to(self._dev) - mg).abs().max().item())
        项2["2_M读GEMM"] = {"max绝对差": 差m,
                            "相对量": round(差m / max(float(np.abs(mc).max()), 1e-12), 9)}
        vb = 批[:, :, None]
        for _ in range(C_弛豫):
            vb = np.sign(np.matmul(W[None, :, :], vb))
        批c = vb[:, :, 0]
        批gr = op_批(None)
        不一致 = int((torch.from_numpy(批c).to(self._dev) != 批gr).sum().item())
        项2["3_lookahead批量"] = {"sign终态不一致元素数": 不一致,
                                  "sign终态不一致占比": round(不一致 / 批c.size, 9),
                                  "元素总数": int(批c.size)}
        out["②_CPU与GPU差异"] = 项2

        # ③ torch.use_deterministic_algorithms(True) 双跑复测
        旧 = torch.are_deterministic_algorithms_enabled()
        项3: dict = {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
        try:
            torch.use_deterministic_algorithms(True)
            项3["开关成功"] = True
            for 名, fn in 算子表:
                a, b = fn(None), fn(None)
                torch.cuda.synchronize(self._dev)
                差 = float((a - b).abs().max().item())
                项3[名] = {"双跑逐位一致": bool(torch.equal(a.view(torch.int32), b.view(torch.int32))),
                           "max绝对差": 差}
        except Exception as exc:                                      # 不支持算子的异常如实登记
            项3["开关成功"] = False
            项3["异常"] = repr(exc)
        finally:
            torch.use_deterministic_algorithms(旧)
        out["③_确定性算法开启双跑"] = 项3

        # ④ cudnn.benchmark=False + 单流确认（探针退出前恢复原值，零副作用）
        旧bench = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        项4 = {
            "cudnn.benchmark原值": bool(旧bench),
            "cudnn.benchmark现值": bool(torch.backends.cudnn.benchmark),
            "当前流": str(torch.cuda.current_stream(self._dev)),
            "默认流": str(torch.cuda.default_stream(self._dev)),
            "单流确认": str(torch.cuda.current_stream(self._dev)) == str(torch.cuda.default_stream(self._dev)),
            "探针退出恢复原值": None,
        }
        torch.backends.cudnn.benchmark = 旧bench
        项4["探针退出恢复原值"] = bool(torch.backends.cudnn.benchmark == 旧bench)
        out["④_cudnn与单流"] = 项4
        return out
