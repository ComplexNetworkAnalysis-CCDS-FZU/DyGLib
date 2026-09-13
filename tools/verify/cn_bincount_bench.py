# -*- coding: utf-8 -*-
"""CN 计数 · bincount 重构版 vs 现行 numpy 向量化版（2026-09-14，用户要求"跑跑看结果"）。

对照：
  A) `utils.accel.cn_counts_vec` —— 现行落库实现（行复合键全局 unique + searchsorted 跨行匹配）；
  B) `cn_bincount_vec()` —— 本基准内候选：行复合键**直方图**（np.bincount，O(N) 一趟）
     + 四次直方图查表（h[ks]/h[kd]），无排序、无 searchsorted；代价 = 两张 B*stride 直方图。
  （C 参考：逐行原实现，仅默认配置跑少量迭代做比例锚点。）

扫描维度：词表 n_nodes（直方图域大小敏感）/ batch B / 序列宽 L / 分布；每个配置先逐位校验、
后计时（中位数 + 最小）。纯 CPU，本机可跑；结果 JSON：cn_bincount_bench_result.json。

用法（仓库根目录）：python tools/verify/cn_bincount_bench.py [--iters 30] [--warmup 5]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.accel import cn_counts_vec as cn_counts_vec_shipped  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "cn_microbench", ROOT / "tools" / "verify" / "cn_microbench.py"
)
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)


# ---------------------------------------------------------------- 候选 B：bincount 重构
def cn_bincount_vec(src_padded_ids: np.ndarray, dst_padded_ids: np.ndarray):
    """与 cn_counts_vec 同语义的直方图版（行复合键 + bincount 查表；仅评估）。"""
    src = np.ascontiguousarray(src_padded_ids, dtype=np.int64)
    dst = np.ascontiguousarray(dst_padded_ids, dtype=np.int64)
    if src.ndim != 2 or dst.ndim != 2:
        raise ValueError(f"[bench] 要求 2D 输入，收到 {src.shape} / {dst.shape}")
    if src.shape[0] != dst.shape[0]:
        raise ValueError(f"[bench] 要求同 batch，收到 {src.shape} / {dst.shape}")
    B, Ls = src.shape
    Ld = dst.shape[1]
    stride = int(max(src.max(initial=0), dst.max(initial=0))) + 1
    rows_s = np.repeat(np.arange(B, dtype=np.int64), Ls)
    rows_d = np.repeat(np.arange(B, dtype=np.int64), Ld)
    ks = rows_s * stride + src.reshape(-1)
    kd = rows_d * stride + dst.reshape(-1)
    n = B * stride
    h_s = np.bincount(ks, minlength=n)
    h_d = np.bincount(kd, minlength=n)
    s_in_s, s_in_d = h_s[ks], h_d[ks]
    d_in_s, d_in_d = h_s[kd], h_d[kd]
    src_app = np.stack([s_in_s, s_in_d], axis=1).reshape(B, Ls, 2)
    dst_app = np.stack([d_in_s, d_in_d], axis=1).reshape(B, Ld, 2)
    src_app[src == 0] = 0
    dst_app[dst == 0] = 0
    return src_app.astype(np.float32), dst_app.astype(np.float32)


# ---------------------------------------------------------------- 计时
def time_fn(fn, args, warmup, iters):
    for _ in range(warmup):
        fn(*args)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn(*args)
        samples.append(time.perf_counter_ns() - t0)
    return {"median_ns": statistics.median(samples), "min_ns": min(samples), "mean_ns": sum(samples) / len(samples)}


def edge_checks():
    rng = np.random.default_rng(0)
    cases = {
        "random": (rng.integers(0, 50, (100, 20)).astype(np.int64), rng.integers(0, 80, (100, 20)).astype(np.int64)),
        "all-zero": (np.zeros((7, 9), dtype=np.int64), np.zeros((7, 9), dtype=np.int64)),
        "L1": (rng.integers(0, 3, (16, 1)).astype(np.int64), rng.integers(0, 3, (16, 1)).astype(np.int64)),
        "Ls!=Ld": (rng.integers(0, 40, (64, 20)).astype(np.int64), rng.integers(0, 40, (64, 13)).astype(np.int64)),
        "big-ids": (rng.integers(1, 10 ** 7, (32, 30)).astype(np.int64), rng.integers(1, 10 ** 7, (32, 30)).astype(np.int64)),
        "empty-cols": (np.zeros((5, 0), dtype=np.int64), np.zeros((5, 0), dtype=np.int64)),
        "one-side-empty": (np.zeros((5, 0), dtype=np.int64), rng.integers(0, 4, (5, 4)).astype(np.int64)),
        "B0": (np.zeros((0, 5), dtype=np.int64), np.zeros((0, 5), dtype=np.int64)),
    }
    bad = []
    for name, (s, d) in cases.items():
        # bincount 版的内存 = 2 * B*stride * 8B（域=最大 id）；大 id 场景会爆内存——
        # 属已知缺陷（真实数据 n_nodes≤O(1e4) 不触发），此处跳过并标注。
        stride = int(max(s.max(initial=0), d.max(initial=0))) + 1
        domain_mb = s.shape[0] * stride * 8 / 1e6
        if domain_mb > 500:
            print(f"    [edge] {name:15s} SKIP（直方图域 {domain_mb:.0f}MB 过大 → bincount 缺陷；shipped 无此问题）")
            continue
        a = cn_counts_vec_shipped(s, d)
        b = cn_bincount_vec(s, d)
        ok = (
            a[0].dtype == b[0].dtype == np.float32
            and a[0].shape == b[0].shape
            and a[1].shape == b[1].shape
            and np.array_equal(a[0], b[0])
            and np.array_equal(a[1], b[1])
        )
        print(f"    [edge] {name:15s} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(name)
    return bad


# ---------------------------------------------------------------- 主流程
def run_config(tag, B, L, n_nodes, dist, rng, warmup, iters, do_orig=False):
    src, dst = mb.gen_batch(B, L, n_nodes, dist, rng)
    a_src, a_dst = cn_counts_vec_shipped(src, dst)
    b_src, b_dst = cn_bincount_vec(src, dst)
    exact = bool(np.array_equal(a_src, b_src) and np.array_equal(a_dst, b_dst))

    t_ship = time_fn(cn_counts_vec_shipped, (src, dst), warmup, iters)
    t_bin = time_fn(cn_bincount_vec, (src, dst), warmup, iters)

    orig_med = None
    if do_orig:
        _ = mb.count_nodes_appearances_instrumented(src, dst, "cpu", mb.Agg())
        t_orig = time_fn(
            lambda s, d: mb.count_nodes_appearances_instrumented(s, d, "cpu", mb.Agg()),
            (src, dst),
            1,
            min(iters, 5),
        )
        orig_med = t_orig["median_ns"]

    stride = int(max(src.max(initial=0), dst.max(initial=0))) + 1
    row = {
        "tag": tag, "B": B, "L": L, "n_nodes": n_nodes, "dist": dist,
        "shipped_median_us": t_ship["median_ns"] / 1e3,
        "shipped_min_us": t_ship["min_ns"] / 1e3,
        "bincount_median_us": t_bin["median_ns"] / 1e3,
        "bincount_min_us": t_bin["min_ns"] / 1e3,
        "speedup_bincount_vs_shipped": t_ship["median_ns"] / t_bin["median_ns"],
        "orig_median_us": None if orig_med is None else orig_med / 1e3,
        "exact_bitwise": exact,
        "hist_domain_mb": 2 * B * stride * 8 / 1e6,
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", type=str, default=str(ROOT / "tools" / "verify" / "cn_bincount_bench_result.json"))
    args = ap.parse_args()

    print("=" * 100)
    print("[边界/一致性自检]（shipped vs bincount）")
    bad = edge_checks()
    if bad:
        print(f"[FAIL] 边界不一致: {bad}")
        sys.exit(1)

    rng = np.random.default_rng(42)
    configs = [
        ("BA 默认", 200, 40, 3783, "zipf", True),
        ("BA 默认·uniform", 200, 40, 3783, "uniform", False),
        ("词表 1k", 200, 40, 1000, "zipf", False),
        ("词表 8k", 200, 40, 8000, "zipf", False),
        ("词表 20k", 200, 40, 20000, "zipf", False),
        ("B=18（尾批）", 18, 40, 3783, "zipf", False),
        ("B=64", 64, 40, 3783, "zipf", False),
        ("B=512", 512, 40, 3783, "zipf", False),
        ("L=16", 200, 16, 3783, "zipf", False),
        ("L=100", 200, 100, 3783, "zipf", False),
    ]

    rows = []
    for tag, B, L, nn, dist, do_orig in configs:
        row = run_config(tag, B, L, nn, dist, rng, args.warmup, args.iters, do_orig)
        rows.append(row)
        print(
            f"  {tag:16s} B={B:<4d} L={L:<4d} nn={nn:<6d} "
            f"shipped={row['shipped_median_us']:8.1f}us  bincount={row['bincount_median_us']:8.1f}us  "
            f"ratio={row['speedup_bincount_vs_shipped']:5.2f}  exact={row['exact_bitwise']}  hist={row['hist_domain_mb']:.1f}MB"
            + (f"  orig={row['orig_median_us']:.0f}us" if row["orig_median_us"] else "")
        )

    print("=" * 100)
    print(f"{'config':18s} {'shipped(us)':>12s} {'bincount(us)':>13s} {'bin/ship':>9s}   verdict")
    for r in rows:
        verdict = "bincount 更快" if r["speedup_bincount_vs_shipped"] > 1.05 else ("≈ 平" if r["speedup_bincount_vs_shipped"] > 0.95 else "bincount 更慢")
        print(f"{r['tag']:18s} {r['shipped_median_us']:12.1f} {r['bincount_median_us']:13.1f} {r['speedup_bincount_vs_shipped']:9.2f}   {verdict}")

    all_exact = all(r["exact_bitwise"] for r in rows)
    ratios = [r["speedup_bincount_vs_shipped"] for r in rows]
    print(f"逐位一致（全部配置）: {all_exact}；bincount/shipped 比率 min={min(ratios):.2f} max={max(ratios):.2f} median={statistics.median(ratios):.2f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "all_exact": all_exact}, f, ensure_ascii=False, indent=2, default=str)
    print(f"JSON -> {args.out}")


if __name__ == "__main__":
    main()
