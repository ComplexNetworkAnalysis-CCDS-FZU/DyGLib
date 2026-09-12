# -*- coding: utf-8 -*-
"""探针：定位 signdyg_accel.bte_sign_effect 的 dtype 绑定契约（一次性诊断用）。

用途：M4 接缝排障——确定哪些入参 dtype 被 Rust 绑定接受/拒绝，
      以决定接缝是否需要显式 dtype 归一化（只做保值转换，不改变数值语义）。
"""
from __future__ import annotations

import numpy as np

import signdyg_accel as m


def attempt(label, **kw):
    try:
        out = m.bte_sign_effect(zero_padding=True, **kw)
        print(f"[OK  ] {label}: shapes={[tuple(o.shape) for o in out]}, dtypes={[o.dtype for o in out]}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")


base_ids = np.array([[1, 3, 0], [2, 4, 0]], dtype=np.int64)
base_signs = np.array([[1, 1, 0], [1, -1, 0]], dtype=np.int8)
src64 = np.array([1, 2], dtype=np.int64)
dst64 = np.array([2, 1], dtype=np.int64)
q64 = np.array([10.0, 11.0], dtype=np.float64)
times32 = np.array([[1.0, 2.0, 0.0], [1.5, 2.5, 0.0]], dtype=np.float32)
times64 = times32.astype(np.float64)


def call(src, dst, s_ids, d_ids, s_sg, d_sg, q, **extra):
    return dict(
        src_nodes=src, dst_nodes=dst,
        src_padded_ids=s_ids, dst_padded_ids=d_ids,
        src_padded_signs=s_sg, dst_padded_signs=d_sg,
        query_times=q, **extra,
    )


attempt("nodes64/ids64/signs8/no-times", **call(src64, dst64, base_ids, base_ids, base_signs, base_signs, q64))
attempt("nodes32", **call(src64.astype(np.int32), dst64.astype(np.int32), base_ids, base_ids, base_signs, base_signs, q64))
attempt("ids32", **call(src64, dst64, base_ids.astype(np.int32), base_ids.astype(np.int32), base_signs, base_signs, q64))
attempt("signs64", **call(src64, dst64, base_ids, base_ids, base_signs.astype(np.int64), base_signs.astype(np.int64), q64))
attempt("times32+decay", **call(src64, dst64, base_ids, base_ids, base_signs, base_signs, q64,
                                src_padded_times=times32, dst_padded_times=times32,
                                time_decay_lambda=0.5, time_decay_gap_mode="staleness", time_scaling_factor=1e-6))
attempt("times64+decay", **call(src64, dst64, base_ids, base_ids, base_signs, base_signs, q64,
                                src_padded_times=times64, dst_padded_times=times64,
                                time_decay_lambda=0.5, time_decay_gap_mode="staleness", time_scaling_factor=1e-6))
attempt("no-query-times", **call(src64, dst64, base_ids, base_ids, base_signs, base_signs, None))
