# -*- coding: utf-8 -*-
"""ahi_theory.runtime3 —— 兼容别名（M2 三时钟域壳**规范名 = `ahi_theory.three_loop`**）。

★★【M2 收尾登记 · 2026-09-27】三环壳实现已原位迁移至规范名 `ahi_theory/three_loop.py`
  （任务书 §2 指定名；逐行同物，仅改名）。本文件保留为**兼容别名**：
  · 日期/原因：2026-09-27，M2 收尾规范化（命名与 95_架构定稿 §5-M2 / 任务书对齐）；
  · 授权依据：meeting11/E4_Round3_签署.md §1 白名单 W1 + meeting11/E3_Round3_签署.md
    §1.1 让渡①（调度权上移三环壳）；闸1 预注册见 `verify_all.py` `_CHANGELOG`
    2026-09-27 M2 收尾条目；
  · 回退方式：两文件名互换即回旧名；公开 API 逐位同物（meeting11/M2_selftest.py 经
    `from ahi_theory.runtime3 import 三环壳, R0反射环, 演练登记表` 继续工作，复跑 PASS 为证）。
  本文件为**纯转发**：不含任何实现，永不隐性扩充（E4 四道闸·闸4 封闭性）。
"""
from ahi_theory.three_loop import *          # noqa: F401,F403 —— 规范名全量转发
from ahi_theory.three_loop import (          # noqa: F401 —— 显式名单（含下划线私名，供旧引用）
    域周期秒, 拍预算秒, 队列深, 龄期上限默认秒, R0看门狗R1死秒, R1降级判定拍,
    R2心跳粒度秒, 收紧比默认, 滞回比默认, R0安全通道标量集, 离线窗兜底秒,
    演练登记表, 演练日志字段, 演练类型枚举,
    域钟, 有界队列, R0反射环, 钳制代理, R2慢思考环, World域, 三环壳,
    _分位数, _摘要, _线性设备, _世界只读设备,
)
