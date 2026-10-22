# -*- coding: utf-8 -*-
"""ahi_theory/gpu_resident.py —— G3 驻留算子层（#202 预登记实装）

★ 与 G1（gpu_backend.G1后端，mirror-in/out 无驻留）的关系：继承其确定性策略与
   numpy 逐位镜像口径；新增「运行期不变矩阵驻留」能力——登记W(W) 一次上传，
   弛豫驻留/lookahead驻留 每调用仅传输向量/批（矩阵零往返）。
★ 适用边界：仅运行期不变张量（W_selfHop 类）；每拍变更的 M 实例驻留需增量同步
   协议，属下一层（#202 明示不含）。
★ 影子纪律（#148 冻结沿用）：纯新增、缺省零侵入、零判定值产出、零拟合。
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

from ahi_theory.hdc_core import C_弛豫, D_VSA, HDC_LEVEL, L4阈值, LAMBDA_M
from ahi_theory.gpu_backend import G1后端, TORCH可用, torch

__all__ = ["G3驻留后端"]


class G3驻留后端(G1后端):
    """驻留算子层：运行期不变矩阵一次上传驻留，逐调用只传向量（#202）。"""

    def __init__(self, 后端: str = "numpy"):
        super().__init__(后端)
        self._驻留: dict = {}                                # 名称→device 张量
        self._M驻留: dict = {}                               # 名称→稠密驻留
        self._M版本: dict = {}                               # 名称→物化拍
        self._M缓冲版本: dict = {}                           # 名称→秩1缓冲 n
        self._M缓冲: dict = {}                               # 名称→(Xg, Kg)
        self._M实例引用: dict = {}                           # 名称→实例 dict 引用
        if 后端 == "cuda":
            self.登记["驻留矩阵"] = {}

    # —— 登记（一次上传，运行期不变） ——
    def 登记W(self, W: np.ndarray, 名称: str = "W_selfHop") -> None:
        assert 名称 not in self._驻留, "驻留名已存在（运行期不变语义，禁覆盖）"
        self._W脏: dict = getattr(self, "_W脏", {})
        self._W脏[名称] = False
        self._W宿主: dict = getattr(self, "_W宿主", {})
        self._W宿主[名称] = W                                # 宿主权威引用（#206）
        if self.后端 == "cuda":
            self._驻留[名称] = torch.from_numpy(
                np.ascontiguousarray(W)).to(self._dev)
            self.登记["驻留矩阵"][名称] = list(W.shape)
        else:
            self._驻留[名称] = W                              # numpy 后端零拷贝引用

    # —— [1s] W 增量同步协议（#206：宿主权威+脏标记全量重传） ——
    def 标脏W(self, 名称: str = "W_selfHop") -> None:
        """宿主 W 变更后由调用方标记（维护拍/井写入路径）；下次驻留调用前自动重传。"""
        self._W脏[名称] = True

    def W同步(self, 名称: str = "W_selfHop") -> None:
        """宿主→device 全量重传（8MB，引擎接线在维护拍后调用；摊销成本见 #206）。"""
        if self.后端 == "cuda" and self._W脏.get(名称):
            self._驻留[名称] = torch.from_numpy(
                np.ascontiguousarray(self._W宿主[名称])).to(self._dev)
            self._W脏[名称] = False

    # —— [1r] 弛豫驻留：矩阵驻留，仅向量进出 ——
    def 弛豫驻留(self, S: np.ndarray, 名称: str = "W_selfHop",
                 C: int = C_弛豫) -> np.ndarray:
        W = self._驻留[名称]
        if self.后端 == "numpy":
            v = S
            for _ in range(C):
                v = np.sign(W @ v)
            return v
        if self._W脏.get(名称):                               # #206 脏自动重传
            self.W同步(名称)
            W = self._驻留[名称]
        vg = torch.from_numpy(np.ascontiguousarray(S)).to(self._dev)
        for _ in range(C):
            vg = torch.sign(W @ vg)
        torch.cuda.synchronize(self._dev)
        return vg.cpu().numpy()

    # —— [3r] lookahead 批量驻留 ——
    def lookahead驻留(self, 批: np.ndarray, 名称: str = "W_selfHop",
                      C: int = C_弛豫) -> np.ndarray:
        W = self._驻留[名称]
        if self.后端 == "numpy":
            out = np.empty_like(批)
            for c in range(批.shape[0]):
                v = 批[c]
                for _ in range(C):
                    v = np.sign(W @ v)
                out[c] = v
            return out
        批g = torch.from_numpy(np.ascontiguousarray(批)).to(self._dev)
        v = 批g[:, :, None]
        for _ in range(C):
            v = torch.sign(torch.matmul(W.unsqueeze(0), v))
        torch.cuda.synchronize(self._dev)
        return v[:, :, 0].cpu().numpy()

    # —— [2r] M 读驻留（#204：版本化增量同步协议） ——
    def 登记M(self, 名称: str, 实例: dict) -> None:
        """稠密基一次驻留（内容仅随物化拍变更，版本号=物化拍）。"""
        assert 名称 not in self._M驻留, "M 驻留名已存在"
        self._M驻留[名称] = None                            # 惰性首次上传
        self._M版本[名称] = None
        self._M缓冲版本[名称] = None                        # (n,) 秩1缓冲版本
        self._M实例引用[名称] = 实例
        if self.后端 == "numpy":
            self._M驻留[名称] = 实例["稠密"]

    def M读驻留(self, 名称: str, 拍数: int, 键矩阵: np.ndarray,
                λ: float = LAMBDA_M) -> np.ndarray:
        """M@键矩阵（实例只读）：稠密驻留（物化拍版本）+ 秩1缓冲增量上传 + w 宿主算。"""
        实例 = self._M实例引用[名称]
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
        # ① 稠密：物化拍版本检查（内容仅随物化变更）
        if self._M版本[名称] != 实例["物化拍"]:
            self._M驻留[名称] = torch.from_numpy(
                np.ascontiguousarray(实例["稠密"])).to(self._dev)
            self._M版本[名称] = 实例["物化拍"]
            self._M缓冲版本[名称] = None                    # 物化重置缓冲
        稠密g = self._M驻留[名称]
        # ② 秩1缓冲：n 增→增量上传新行；n 减→全量重传（缓冲按实例隔离）
        n = 实例["n"]
        cached = self._M缓冲版本[名称]
        if cached is None or n < cached:
            if n:
                Xg = torch.from_numpy(
                    np.ascontiguousarray(实例["X"][:n])).to(self._dev)
                Kg = torch.from_numpy(
                    np.ascontiguousarray(实例["K"][:n])).to(self._dev)
                self._M缓冲[名称] = (Xg, Kg)
            else:
                self._M缓冲[名称] = None
        elif n > cached:
            newX = torch.from_numpy(
                np.ascontiguousarray(实例["X"][cached:n])).to(self._dev)
            newK = torch.from_numpy(
                np.ascontiguousarray(实例["K"][cached:n])).to(self._dev)
            if cached == 0 or self._M缓冲.get(名称) is None:
                self._M缓冲[名称] = (newX, newK)
            else:
                Xg, Kg = self._M缓冲[名称]
                self._M缓冲[名称] = (torch.cat([Xg, newX], 0),
                                     torch.cat([Kg, newK], 0))
        self._M缓冲版本[名称] = n
        # ③ w 宿主算（依赖拍数衰减）+ 键矩阵每拍上传
        键g = torch.from_numpy(np.ascontiguousarray(键矩阵)).to(self._dev)
        outg = 稠密g @ 键g
        折 = float(λ ** (拍数 - 实例["物化拍"]))
        if 折 != 1.0:
            outg = outg * 折
        if n:
            w = (实例["η"][:n] * λ ** (拍数 - 实例["t"][:n])).astype(np.float32)
            wg = torch.from_numpy(w).to(self._dev)
            Xg, Kg = self._M缓冲[名称]
            cg = Kg @ 键g
            outg = outg + Xg.t() @ (wg[:, None] * cg)
        torch.cuda.synchronize(self._dev)
        return outg.cpu().numpy()

    def M读驻留对象(self, 实例: dict, 拍数: int, 键矩阵: np.ndarray,
                    λ: float = LAMBDA_M) -> np.ndarray:
        """#208 stage A：按实例对象惰性自动登记的 M 读驻留（引擎 _M读 钩子入口）。"""
        名称 = "M@%d" % id(实例)
        if 名称 not in self._M实例引用:
            self._M驻留[名称] = None
            self._M版本[名称] = None
            self._M缓冲版本[名称] = None
            self._M缓冲[名称] = None
            self._M实例引用[名称] = 实例
        return self.M读驻留(名称, 拍数, 键矩阵, λ)
