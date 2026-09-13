#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CN（共现编码）细粒度分级评估 —— K3 前置，纯 CPU，可本机运行。

背景（服务器测试阶段 profiler，仅包裹 test forward）：
  - "Common Neighbor Encoding" 每次调用 mean≈130-140ms（BA），占被计时模块总和 ~80%；
  - 该计时**不受 CNAS/BTE 模块开关影响**（全关 vanilla run 仍 137.7ms）=> 无条件计算；
  - K1/K2 已把 History Sampling / Balance Theory 降下来，CN 是下一个大头。

本脚本：
  1) 本地逐位复刻 `models/NeighborInteractEncoder.py :: count_nodes_appearances`
     的逐行实现，在 ~12 个子步骤打点，给出"每次调用"的耗时分解；
  2) 提供"全向量化原型"（仅评估、不落库）作上限对照 + 逐位一致性校验；
  3) 单独测量 CN 命中后的 encode 层（Linear 1→50→ReLU→50→sum）成本。

本机限制（无 CUDA）：
  - `.to(self.device)` 涉及 4 次/行（800 次/调用）的行级 H2D 小传输，本机不可测，
    需服务器小样本标定；
  - CPU 绝对速度与服务器不同 => 看"份额/子步骤结构"与"向量化倍数"，绝对值仅参考。

用法（在仓库根目录执行）：
    python tools/verify/cn_microbench.py                      # 默认 B=200, L=40, zipf
    python tools/verify/cn_microbench.py --sweep              # L 扫描 16~100
    python tools/verify/cn_microbench.py --dist uniform --iters 50
    python tools/verify/cn_microbench.py --out tools/verify/cn_microbench_result.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict

import numpy as np
import torch

# ---------------------------------------------------------------- 计时工具
def _tic():
    return time.perf_counter_ns()


class Agg:
    """按子步骤累计耗时（ns）。"""

    def __init__(self):
        self.t = defaultdict(float)
        self.n = defaultdict(int)

    def add(self, key, ns):
        self.t[key] += float(ns)
        self.n[key] += 1

    def total(self):
        return sum(self.t.values())


# ---------------------------------------------------------------- 原实现复刻（逐位语义一致）
def count_nodes_appearances_instrumented(
    src_padded_nodes_neighbor_ids: np.ndarray,
    dst_padded_nodes_neighbor_ids: np.ndarray,
    device: str,
    agg: Agg,
):
    """复刻 NeighborInteractEncoder.count_nodes_appearances，并打点。

    与源码一致的操作顺序：
      每行循环内: np.unique(src) → counts 还原 → dict(src) →
                 np.unique(dst) → counts 还原 → dict(dst) →
                 to_float_torch(src, dst_map) → stack 行(src) →
                 to_float_torch(dst, src_map) → stack 行(dst)
      循环后: 外层 stack ×2 → mask 清零 ×2
    """
    src_app_list, dst_app_list = [], []
    B = len(src_padded_nodes_neighbor_ids)

    for i in range(B):
        src_row = src_padded_nodes_neighbor_ids[i]
        dst_row = dst_padded_nodes_neighbor_ids[i]

        t0 = _tic()
        src_unique_keys, src_inverse_indices, src_counts = np.unique(
            src_row, return_inverse=True, return_counts=True
        )
        agg.add("np.unique(src)", _tic() - t0)

        t0 = _tic()
        src_counts_in_src = (
            torch.from_numpy(src_counts[src_inverse_indices]).float().to(device)
        )
        agg.add("counts还原(src-in-src)", _tic() - t0)

        t0 = _tic()
        src_mapping_dict = dict(zip(src_unique_keys, src_counts))
        agg.add("dict构建(src)", _tic() - t0)

        t0 = _tic()
        dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(
            dst_row, return_inverse=True, return_counts=True
        )
        agg.add("np.unique(dst)", _tic() - t0)

        t0 = _tic()
        dst_counts_in_dst = (
            torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(device)
        )
        agg.add("counts还原(dst-in-dst)", _tic() - t0)

        t0 = _tic()
        dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))
        agg.add("dict构建(dst)", _tic() - t0)

        # ---- to_float_torch(src_row.copy(), lambda: dst_mapping_dict.get(...))
        t0 = _tic()
        arr = src_row.copy()
        agg.add("copy(src行)", _tic() - t0)
        t0 = _tic()
        ten = torch.from_numpy(arr)
        ten = ten.apply_(lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0))
        agg.add("apply_逐元素映射(src→dst,含dict.get)", _tic() - t0)
        t0 = _tic()
        src_counts_in_dst = ten.float().to(device)
        agg.add("float+to(src→dst)", _tic() - t0)

        t0 = _tic()
        src_app_list.append(torch.stack([src_counts_in_src, src_counts_in_dst], dim=1))
        agg.add("stack单行(src)", _tic() - t0)

        # ---- to_float_torch(dst_row.copy(), lambda: src_mapping_dict.get(...))
        t0 = _tic()
        arr = dst_row.copy()
        agg.add("copy(dst行)", _tic() - t0)
        t0 = _tic()
        ten = torch.from_numpy(arr)
        ten = ten.apply_(lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0))
        agg.add("apply_逐元素映射(dst→src,含dict.get)", _tic() - t0)
        t0 = _tic()
        dst_counts_in_src = ten.float().to(device)
        agg.add("float+to(dst→src)", _tic() - t0)

        t0 = _tic()
        dst_app_list.append(torch.stack([dst_counts_in_src, dst_counts_in_dst], dim=1))
        agg.add("stack单行(dst)", _tic() - t0)

    t0 = _tic()
    src_padded_nodes_appearances = torch.stack(src_app_list, dim=0)
    dst_padded_nodes_appearances = torch.stack(dst_app_list, dim=0)
    agg.add("stack外层×2", _tic() - t0)

    t0 = _tic()
    src_padded_nodes_appearances[
        torch.from_numpy(src_padded_nodes_neighbor_ids == 0)
    ] = 0.0
    dst_padded_nodes_appearances[
        torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)
    ] = 0.0
    agg.add("mask清零×2", _tic() - t0)

    return src_padded_nodes_appearances, dst_padded_nodes_appearances


# ---------------------------------------------------------------- 全向量化原型（仅评估）
def count_nodes_appearances_vectorized(
    src: np.ndarray, dst: np.ndarray, node_stride: int, device: str, agg: Agg | None = None
):
    """全批次向量化（无逐行 python 循环、无逐元素回调、单次批量传输）。

    语义与逐行版逐位一致：每行的 [src内计数, src在dst中的计数] / [dst在src中的计数, dst内计数]。
    实现：行复合键 = row*stride + id，全局 unique 取计数，searchsorted 做跨行匹配。
    """
    B, L = src.shape
    N = B * L
    rows = np.repeat(np.arange(B, dtype=np.int64), L)
    ks = rows * node_stride + src.reshape(-1)
    kd = rows * node_stride + dst.reshape(-1)

    def _add(key, t0):
        if agg is not None:
            agg.add(key, _tic() - t0)

    t0 = _tic()
    us, inv_s, cnt_s = np.unique(ks, return_inverse=True, return_counts=True)
    ud, inv_d, cnt_d = np.unique(kd, return_inverse=True, return_counts=True)
    _add("vec.np.unique(全局×2)", t0)

    t0 = _tic()
    s_in_s = cnt_s[inv_s]
    d_in_d = cnt_d[inv_d]
    _add("vec.同行计数还原×2", t0)

    t0 = _tic()
    pos = np.searchsorted(ud, ks)
    pos_c = np.minimum(pos, ud.size - 1)
    hit = ud[pos_c] == ks
    s_in_d = np.where(hit, cnt_d[pos_c], 0)
    pos2 = np.searchsorted(us, kd)
    pos2_c = np.minimum(pos2, us.size - 1)
    hit2 = us[pos2_c] == kd
    d_in_s = np.where(hit2, cnt_s[pos2_c], 0)
    _add("vec.searchsorted匹配×2", t0)

    t0 = _tic()
    src_app = np.stack([s_in_s, s_in_d], axis=1).reshape(B, L, 2)
    dst_app = np.stack([d_in_s, d_in_d], axis=1).reshape(B, L, 2)
    _add("vec.np.stack×2", t0)

    t0 = _tic()
    src_app[src == 0] = 0
    dst_app[dst == 0] = 0
    _add("vec.mask清零×2", t0)

    t0 = _tic()
    src_t = torch.from_numpy(src_app.astype(np.float32)).to(device)
    dst_t = torch.from_numpy(dst_app.astype(np.float32)).to(device)
    _add("vec.float+to(仅2次批量)", t0)

    return src_t, dst_t


# ---------------------------------------------------------------- encode 层（CN 命中后的网络层）
def build_encode_layer(feat_dim: int = 50):
    layer = torch.nn.Sequential(
        torch.nn.Linear(in_features=1, out_features=feat_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=feat_dim, out_features=feat_dim),
    )
    layer.eval()
    return layer


def run_encode_layer(layer, src_app: torch.Tensor, dst_app: torch.Tensor):
    with torch.no_grad():
        src_feat = layer(src_app.unsqueeze(dim=-1)).sum(dim=2)
        dst_feat = layer(dst_app.unsqueeze(dim=-1)).sum(dim=2)
    return src_feat, dst_feat


# ---------------------------------------------------------------- 数据生成
def gen_batch(B, L, n_nodes, dist, rng):
    """生成用于计时的批数据（int64，0 填充，可含重复邻居）。"""
    src = np.zeros((B, L), dtype=np.longlong)
    dst = np.zeros((B, L), dtype=np.longlong)

    if dist == "zipf":
        rank = np.arange(1, n_nodes + 1)
        p = 1.0 / np.power(rank, 1.2)
        p /= p.sum()
        logp = np.log(p)
        sample = lambda size: (rng.choice(n_nodes, size=size, p=p) + 1).astype(np.longlong)
    else:
        sample = lambda size: rng.integers(1, n_nodes + 1, size=size).astype(np.longlong)

    # 真实行长度：1..L（先抽长度，再填充）——最后一行为 L 以保证 padded 宽度确定
    lens = rng.integers(1, L + 1, size=B)
    lens[-1] = L
    for i in range(B):
        li = int(lens[i])
        src[i, :li] = sample(li)
        dst[i, :li] = sample(li)
    return src, dst


# ---------------------------------------------------------------- 单配置评估
def evaluate_once(cfg, src, dst, device, warmup, iters):
    agg = Agg()

    # 预热
    for _ in range(warmup):
        count_nodes_appearances_instrumented(src, dst, device, Agg())

    t0 = _tic()
    for _ in range(iters):
        s_ref, d_ref = count_nodes_appearances_instrumented(src, dst, device, agg)
    wall_orig = (_tic() - t0) / iters

    agg_vec = Agg()
    for _ in range(warmup):
        count_nodes_appearances_vectorized(src, dst, cfg["node_stride"], device, None)
    t0 = _tic()
    for _ in range(iters):
        s_vec, d_vec = count_nodes_appearances_vectorized(
            src, dst, cfg["node_stride"], device, agg_vec
        )
    wall_vec = (_tic() - t0) / iters

    exact = bool(torch.equal(s_ref, s_vec) and torch.equal(d_ref, d_vec))

    # encode 层（CPU）
    layer = build_encode_layer()
    for _ in range(warmup):
        run_encode_layer(layer, s_ref, d_ref)
    t0 = _tic()
    for _ in range(iters):
        run_encode_layer(layer, s_ref, d_ref)
    wall_enc = (_tic() - t0) / iters

    return {
        "orig_ns": wall_orig,
        "vec_ns": wall_vec,
        "enc_ns": wall_enc,
        "exact": exact,
        "agg_orig": {k: v / iters for k, v in agg.t.items()},
        "agg_vec": {k: v / iters for k, v in agg_vec.t.items()},
    }


def report(cfg, res):
    MS = 1e6
    print("=" * 78)
    print(
        f"[配置] B={cfg['batch']}  L={cfg['seq_len']}  n_nodes={cfg['n_nodes']}  "
        f"dist={cfg['dist']}  seed={cfg['seed']}  iters={cfg['iters']} (warmup={cfg['warmup']})  device={cfg['device']}"
    )
    print(
        f"[原实现] 每次调用 {res['orig_ns']/MS:8.2f} ms   "
        f"（打点切片合计 {sum(res['agg_orig'].values())/MS:8.2f} ms）"
    )
    total = res["orig_ns"]
    rows = sorted(res["agg_orig"].items(), key=lambda kv: -kv[1])
    for k, v in rows:
        print(f"    {k:<38s} {v/MS:9.3f} ms   {100.0*v/total:6.2f}%")
    # 固定成本 vs 元素规模成本 汇总
    fixed_keys = [
        "np.unique(src)",
        "np.unique(dst)",
        "counts还原(src-in-src)",
        "counts还原(dst-in-dst)",
        "dict构建(src)",
        "dict构建(dst)",
        "copy(src行)",
        "copy(dst行)",
        "float+to(src→dst)",
        "float+to(dst→src)",
        "stack单行(src)",
        "stack单行(dst)",
        "stack外层×2",
        "mask清零×2",
    ]
    fixed = sum(res["agg_orig"].get(k, 0.0) for k in fixed_keys)
    elem = sum(
        v for k, v in res["agg_orig"].items() if k.startswith("apply_逐元素映射")
    )
    print(
        f"    ---- 汇总: 逐行固定成本类 {fixed/MS:8.2f} ms ({100*fixed/total:5.1f}%) | "
        f"逐元素 apply_ 类 {elem/MS:8.2f} ms ({100*elem/total:5.1f}%)"
    )
    print(
        f"[向量化原型] 每次调用 {res['vec_ns']/MS:8.2f} ms   → 加速 {res['orig_ns']/res['vec_ns']:5.1f}x   "
        f"逐位一致={res['exact']}"
    )
    for k, v in sorted(res["agg_vec"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:<38s} {v/MS:9.3f} ms")
    print(f"[encode 层(CPU, src+dst 各一次)] 每次调用 {res['enc_ns']/MS:8.2f} ms")
    print(
        "[提示] .to(cuda) 行级传输(4×B=800 次/调用)在本机不可测，需服务器标定；绝对值随 CPU 不同而变，看份额即可。"
    )


def main():
    ap = argparse.ArgumentParser(description="CN 共现编码细粒度微基准（本机 CPU）")
    ap.add_argument("--batch", type=int, default=200, help="batch size（BA 主实验=200）")
    ap.add_argument("--seq-len", type=int, default=40, help="padded 序列宽 L（BA: NN=40 → L≤40）")
    ap.add_argument("--n-nodes", type=int, default=3783, help="节点数（BA=3783）")
    ap.add_argument("--dist", choices=["zipf", "uniform"], default="zipf")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--sweep", action="store_true", help="扫描 L ∈ {16,24,32,40,64,100}")
    ap.add_argument("--out", type=str, default=None, help="结果 JSON 输出路径（可选）")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] 本机无 CUDA，回退 cpu")
        args.device = "cpu"

    print("=" * 78)
    print("CN 微基准环境:")
    print(
        f"  Python {sys.version.split()[0]} | torch {torch.__version__} | numpy {np.__version__} "
        f"| threads={torch.get_num_threads()} | {platform.platform()}"
    )
    print(f"  CPU: {platform.processor()}")

    rng = np.random.default_rng(args.seed)
    results = []

    if args.sweep:
        for L in [16, 24, 32, 40, 64, 100]:
            cfg = dict(
                batch=args.batch,
                seq_len=L,
                n_nodes=args.n_nodes,
                dist=args.dist,
                seed=args.seed,
                iters=min(args.iters, 20),
                warmup=3,
                device=args.device,
                node_stride=args.n_nodes + 2,
            )
            src, dst = gen_batch(cfg["batch"], L, cfg["n_nodes"], cfg["dist"], rng)
            res = evaluate_once(cfg, src, dst, args.device, cfg["warmup"], cfg["iters"])
            report(cfg, res)
            results.append(dict(cfg=cfg, res={k: v for k, v in res.items() if k != "agg_orig" and k != "agg_vec"}))
        # 汇总表
        print("=" * 78)
        print("扫描汇总: L | 原实现 ms | 向量化 ms | 加速 | apply_占比 | 逐行固定占比")
        for r in results:
            c, rs = r["cfg"], r["res"]
            print(
                f"  L={c['seq_len']:>3d} | {rs['orig_ns']/1e6:9.2f} | {rs['vec_ns']/1e6:8.3f} | "
                f"{rs['orig_ns']/rs['vec_ns']:6.1f}x | encode={rs['enc_ns']/1e6:7.2f}ms"
            )
    else:
        cfg = dict(
            batch=args.batch,
            seq_len=args.seq_len,
            n_nodes=args.n_nodes,
            dist=args.dist,
            seed=args.seed,
            iters=args.iters,
            warmup=args.warmup,
            device=args.device,
            node_stride=args.n_nodes + 2,
        )
        src, dst = gen_batch(cfg["batch"], cfg["seq_len"], cfg["n_nodes"], cfg["dist"], rng)
        res = evaluate_once(cfg, src, dst, args.device, cfg["warmup"], cfg["iters"])
        report(cfg, res)
        results.append(dict(cfg=cfg, res=res))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
