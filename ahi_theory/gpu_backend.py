# -*- coding: utf-8 -*-
"""ahi_theory/gpu_backend.py —— G1 · backend 抽象层（影子模式，#148 预登记）

规格出处：meeting16/00_GPU迁移路线.md §2/§3/§6（G1=backend 抽象层实装；
预登记=a hi_theory/verify_all.py `_CHANGELOG` #148，2026-10-03，先于改码）。

影子纪律（#148 冻结）：
  · 纯新增；不 import brain/world/system/three_loop/mbp（影子期零侵入）；
    引擎红线文件（brain.py 等）零改动；缺省不接入任何管线。
  · numpy 后端=引擎表达式逐字镜像（判据口径不变）；cuda 后端=镜像进出
    （mirror-in/out，无状态驻留——驻留优化属 G2/G3）。
  · 零拟合红线：无 lstsq/linalg.inv·pinv/solve/scipy.optimize/参数搜索构造。
  · 零判定值产出：本层只提供算子，不产出任何判据读数；判据/阈值零改动。
  · R0 反射环禁迁（无 numpy 纪律层，保持纯标量）——本层不含 R0。

热点算子（路线 §3 引擎热路径四算子）：
  1) 弛豫：step8 同款 S̃←sign(W_selfHop@S̃)×C_弛豫；
  2) M读：_M读 全形（稠密基衰减 + 秩1缓冲小侧 GEMM；实例 dict 只读）；
  3) lookahead批量：候选批量弛豫（numpy 循环=引擎形；cuda 批量 GEMM）；
  4) 编码：编码u 同款（HDC_LEVEL 开关；L4 阈值取预注册常量）。

确定性策略（路线 §2，cuda 后端构造时设置并登记）：
  CUBLAS_WORKSPACE_CONFIG=:4096:8 + torch.use_deterministic_algorithms(True)
  + cudnn.benchmark=False + 单流（默认流）。
  GPU 浮点差异预期内：CPU↔GPU 差异界=标定数据（不作 FAIL）；
  硬判据只有「GPU 口径内自一致」（同输入双跑逐位）。
"""
from __future__ import annotations

import os

# —— cublas 确定性工作区（必须 torch 导入前设置；CPU BLAS 线程 env 同 gpu_path 口径） ——
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

from ahi_theory.hdc_core import C_弛豫, HDC_LEVEL, L4阈值, LAMBDA_M

try:
    import torch
    _TORCH_OK = True
except Exception:                                              # 无 torch 环境如实降级
    torch = None
    _TORCH_OK = False

__all__ = ["G1后端", "TORCH可用"]


def TORCH可用() -> bool:
    """torch 可导入且 CUDA 可用（缺一即 False；bitcheck 据此如实 SKIPPED）。"""
    return bool(_TORCH_OK and torch.cuda.is_available())


class G1后端:
    """G1 backend 抽象层：热点算子的 numpy（逐位镜像）/ cuda（并行实现）双后端。

    numpy 后端（缺省）：与 hdc_core 引擎表达式逐字一致——用于证明抽象层镜像无偏差。
    cuda 后端：torch 实现同算子；构造时设置确定性策略；镜像进出无状态驻留。
    本类不持有引擎状态、不改写任何传入数组（只读）；产出 numpy 数组（引擎可直用）。
    """

    def __init__(self, 后端: str = "numpy"):
        assert 后端 in ("numpy", "cuda"), "后端 ∈ {numpy, cuda}"
        self.后端 = 后端
        self.登记: dict = {"后端": 后端}
        if 后端 == "cuda":
            assert TORCH可用(), "cuda 后端要求 torch+CUDA 可用（TORCH可用()）"
            # —— 确定性策略（路线 §2；幂等设置） ——
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.cuda.set_stream(torch.cuda.default_stream(0))   # 单流=默认流
            self._dev = torch.device("cuda")
            self.登记.update({
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "use_deterministic_algorithms": bool(
                    torch.are_deterministic_algorithms_enabled()),
                "cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
                "当前流": str(torch.cuda.current_stream(0)),
                "torch版本": torch.__version__,
                "设备名": torch.cuda.get_device_properties(0).name,
            })

    # —— 内部：镜像进出 ——
    def _进(self, *数组: np.ndarray):
        return tuple(torch.from_numpy(a).to(self._dev) for a in 数组)

    def _出(self, t) -> np.ndarray:
        return t.detach().cpu().numpy()

    # —— [1] 弛豫：step8 同款 sign(W@S̃)×C ——
    def 弛豫(self, W: np.ndarray, S: np.ndarray, C: int = C_弛豫) -> np.ndarray:
        """S̃←sign(W@S̃) 连续 C 次（hdc_core step8 表达式逐字镜像；C 缺省=C_弛豫）。
        引擎形（步进R1 内）为 v=np.sign(W@v)——本函数同式；numpy 后端逐位一致。"""
        if self.后端 == "numpy":
            v = S
            for _ in range(C):
                v = np.sign(W @ v)
            return v
        Wg, vg = self._进(W, S)
        for _ in range(C):
            vg = torch.sign(Wg @ vg)
        torch.cuda.synchronize(self._dev)
        return self._出(vg)

    # —— [2] M读：_M读 全形（稠密衰减+秩1缓冲小侧；实例 dict 只读） ——
    def M读(self, 实例: dict, 拍数: int, 键矩阵: np.ndarray,
            λ: float = LAMBDA_M) -> np.ndarray:
        """M@键矩阵（键矩阵 (D,Q)）：稠密物化基（衰减至拍数）+ 秩1缓冲 GEMM 求和。
        hdc_core._M读 表达式逐字镜像（#132 实例 dict 口径：["稠密"]/["n"]/["X"]/["K"]/
        ["η"]/["t"]/["物化拍"]）；实例只读不写。"""
        if self.后端 == "numpy":
            稠密, n = 实例["稠密"], 实例["n"]
            out = 稠密 @ 键矩阵
            折 = λ ** (拍数 - 实例["物化拍"])
            if 折 != 1.0:
                out = out * 折
            if n:
                w = 实例["η"][:n] * λ ** (拍数 - 实例["t"][:n])
                c = 实例["K"][:n] @ 键矩阵
                out += 实例["X"][:n].T @ (w[:, None] * c)
            return out
        稠密g, 键g = self._进(实例["稠密"], 键矩阵)
        outg = 稠密g @ 键g
        折 = float(λ ** (拍数 - 实例["物化拍"]))
        if 折 != 1.0:
            outg = outg * 折
        n = 实例["n"]
        if n:
            # η/t 引擎为 float64；torch 不混型 ⇒ GPU 路径 w 降型 f32（CPU↔GPU 差异界覆盖；
            # 引擎 numpy 路径 out(=f32) += f64 小侧为 same_kind 就地转型，终态同为 f32）
            wg = self._进((实例["η"][:n] * λ ** (拍数 - 实例["t"][:n])).astype(np.float32))[0]
            Kg, = self._进(实例["K"][:n])
            Xg, = self._进(实例["X"][:n])
            cg = Kg @ 键g
            outg = outg + Xg.t() @ (wg[:, None] * cg)
        torch.cuda.synchronize(self._dev)
        return self._出(outg)

    # —— [3] lookahead批量：候选批量弛豫 ——
    def lookahead批量(self, W: np.ndarray, 批: np.ndarray,
                      C: int = C_弛豫) -> np.ndarray:
        """批 (m,D) 每行经 sign(W@v)×C 弛豫。numpy=引擎循环形（selfthink step6 同款，
        逐位一致）；cuda=批量 GEMM（W 广播，与引擎形数值差异属 CPU↔GPU 差异界范畴）。"""
        if self.后端 == "numpy":
            out = np.empty_like(批)
            for c in range(批.shape[0]):
                v = 批[c]
                for _ in range(C):
                    v = np.sign(W @ v)
                out[c] = v
            return out
        Wg, 批g = self._进(W, 批)
        v = 批g[:, :, None]
        for _ in range(C):
            v = torch.sign(torch.matmul(Wg.unsqueeze(0), v))
        torch.cuda.synchronize(self._dev)
        return self._出(v[:, :, 0])

    # —— [4] 编码：编码u 同款（HDC_LEVEL 开关） ——
    def 编码(self, Φ: np.ndarray, u: np.ndarray,
             level: int = HDC_LEVEL, c向量: np.ndarray = None) -> np.ndarray:
        """x=sign(Φ@u)（level=1，缺省逐位一致）；level=4 ⇒ 四分块多级编码
        x̃=sign(Φ_L@u−c_vec·‖u‖₂)（#130 口径；c_vec=预注册 L4 阈值向量）。"""
        if level == 1:
            if self.后端 == "numpy":
                return np.sign(Φ @ u.astype(np.float32))
            Φg, ug = self._进(Φ, np.asarray(u, dtype=np.float32))
            torch.cuda.synchronize(self._dev)
            return self._出(torch.sign(Φg @ ug))
        assert c向量 is not None, "level=4 需 c向量（预注册 L4 阈值）"
        norm = float(np.linalg.norm(u))
        if self.后端 == "numpy":
            return np.sign(Φ @ u.astype(np.float32) - c向量 * norm)
        Φg, ug = self._进(Φ, np.asarray(u, dtype=np.float32))
        cg, = self._进(c向量.astype(np.float32))
        torch.cuda.synchronize(self._dev)
        return self._出(torch.sign(Φg @ ug - cg * norm))
