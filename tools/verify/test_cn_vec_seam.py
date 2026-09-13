# -*- coding: utf-8 -*-
"""CN 计数 accel 接缝自检（numpy 过渡版；2026-09-14 落地，用户批准）。

断言：
 ① 等价性：accel 向量化路径（utils.accel.cn_counts_vec）与逐行原路径**逐位一致**
    （覆盖：随机重复 / 全零 / 单行 / L=1 / Ls≠Ld / 大 id / 全同值 / 空列 / 单侧空）；
 ② 开关路由：accel.on=True → 走向量化（cn_counts_vec 被调用）；False → 不调用（回原路径）；
 ③ 输出契约：float32、(B,L,2)、CPU tensor（device="cpu" 下）。

用法（仓库根目录）：python tools/verify/test_cn_vec_seam.py
实现说明：本脚本手动注入 `accel.on`（不经 configure，不触碰内核/环境解析）；结束后恢复现场。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.NeighborInteractEncoder import NeighborCooccurrenceEncoder  # noqa: E402
from utils import accel  # noqa: E402

_SENTINEL = object()
failures = []


def _set_on(value: bool):
    accel.on = value


def _restore_on(prev):
    if prev is _SENTINEL:
        accel.__dict__.pop("on", None)
    else:
        accel.on = prev


def make_cases(rng):
    yield "random-repeats", rng.integers(0, 50, size=(100, 20)).astype(np.int64), rng.integers(0, 80, size=(100, 20)).astype(np.int64)
    yield "all-zero", np.zeros((7, 9), dtype=np.int64), np.zeros((7, 9), dtype=np.int64)
    yield "single-row", rng.integers(1, 5, size=(1, 12)).astype(np.int64), rng.integers(0, 3, size=(1, 12)).astype(np.int64)
    yield "L1", rng.integers(0, 3, size=(16, 1)).astype(np.int64), rng.integers(0, 3, size=(16, 1)).astype(np.int64)
    yield "Ls!=Ld", rng.integers(0, 40, size=(64, 20)).astype(np.int64), rng.integers(0, 40, size=(64, 13)).astype(np.int64)
    yield "big-ids", rng.integers(1, 10 ** 7, size=(32, 30)).astype(np.int64), rng.integers(1, 10 ** 7, size=(32, 30)).astype(np.int64)
    yield "all-same", np.full((10, 15), 3, dtype=np.int64), np.full((10, 15), 3, dtype=np.int64)
    yield "empty-cols", np.zeros((5, 0), dtype=np.int64), np.zeros((5, 0), dtype=np.int64)
    yield "one-side-empty", np.zeros((5, 0), dtype=np.int64), rng.integers(0, 4, size=(5, 4)).astype(np.int64)


def main():
    rng = np.random.default_rng(42)
    enc = NeighborCooccurrenceEncoder(neighbor_co_occurrence_feat_dim=50, device="cpu")
    prev = accel.__dict__.get("on", _SENTINEL)

    try:
        # ---- ① 等价性 ----
        print("[①] 逐位等价（off vs on vs 直接调用）")
        for name, s, d in make_cases(rng):
            _set_on(False)
            o_off = enc.count_nodes_appearances(
                src_padded_nodes_neighbor_ids=s, dst_padded_nodes_neighbor_ids=d
            )
            _set_on(True)
            o_on = enc.count_nodes_appearances(
                src_padded_nodes_neighbor_ids=s, dst_padded_nodes_neighbor_ids=d
            )
            v1, v2 = accel.cn_counts_vec(s, d)
            ok_switch = torch.equal(o_off[0], o_on[0]) and torch.equal(o_off[1], o_on[1])
            ok_fn = torch.equal(torch.from_numpy(v1), o_off[0]) and torch.equal(
                torch.from_numpy(v2), o_off[1]
            )
            ok_dtype = o_on[0].dtype == torch.float32 and o_on[1].dtype == torch.float32
            shapes = (tuple(o_off[0].shape), tuple(o_off[1].shape))
            print(f"    {name:14s} switch={ok_switch} fn={ok_fn} dtype={ok_dtype} shapes={shapes}")
            if not (ok_switch and ok_fn and ok_dtype):
                failures.append(f"equiv:{name}")

        # vec 独有：B=0 安全返回（原路径对空批次会崩，不在对比范围）
        v0 = accel.cn_counts_vec(np.zeros((0, 5), dtype=np.int64), np.zeros((0, 5), dtype=np.int64))
        ok_b0 = v0[0].shape == (0, 5, 2) and v0[1].shape == (0, 5, 2)
        print(f"    B=0 vec-safe  = {ok_b0}")
        if not ok_b0:
            failures.append("B=0")

        # ---- ② 开关路由 ----
        print("[②] 开关路由（spy cn_counts_vec）")
        called = {"n": 0}
        orig_fn = accel.cn_counts_vec

        def spy(a, b):
            called["n"] += 1
            return orig_fn(a, b)

        accel.cn_counts_vec = spy
        try:
            s = rng.integers(1, 9, size=(8, 8)).astype(np.int64)
            d = rng.integers(1, 9, size=(8, 8)).astype(np.int64)
            _set_on(False)
            enc.count_nodes_appearances(src_padded_nodes_neighbor_ids=s, dst_padded_nodes_neighbor_ids=d)
            n_off = called["n"]
            _set_on(True)
            enc.count_nodes_appearances(src_padded_nodes_neighbor_ids=s, dst_padded_nodes_neighbor_ids=d)
            n_on = called["n"]
        finally:
            accel.cn_counts_vec = orig_fn
        print(f"    accel=off 时调用次数 = {n_off}（期望 0）；accel=on 时 = {n_on}（期望 1）")
        if not (n_off == 0 and n_on == 1):
            failures.append("routing")
    finally:
        _restore_on(prev)

    if failures:
        print(f"[FAIL] {len(failures)} 项未过: {failures}")
        sys.exit(1)
    print("[ALL PASS] CN 向量化接缝：逐位等价 + 开关路由 + 契约 全过")


if __name__ == "__main__":
    main()
