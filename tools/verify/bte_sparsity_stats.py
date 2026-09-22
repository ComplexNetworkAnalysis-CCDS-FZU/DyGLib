# -*- coding: utf-8 -*-
"""D1–D3：BTE 通道稀疏度度量（模型侧 · 纯 CPU · 无需训练）。

目的（Paper 2026-09-22 问询 `mb-20260922-121123-paper-2c30`）：
  D1 每样本 BTE 非零位置比例（均值/分位）；
  D2 BTE 通道整段为零的样本占比（按数据集）；
  D3 按"BTE 是否全零"分组的性能对照（本工具只产出分组掩码，性能由 D3 脚本消费）。

做法：完全复刻模型侧链路（与 SignDyGFormer.forward 一致），但**不训练、不加载权重**：
  1. `get_link_prediction_data` 取数据（与训练同参数：val/test=0.15，tail 同主实验）；
  2. `get_neighbor_sampler(full_data, ...)`（与训练 eval 相同：recent、LF=主表最优、RAS/CNAS 按 full 配置开）；
  3. 对 test_data 的每条真实边：`history_neighbors_sampling` → pad_sequences 复刻
     （截到最近 NN-1 + pos0 自身 + 右侧零填充，与 models/SignDyGFormer.py::pad_sequences 逐行一致）；
  4. `NeighborCooccurrenceEncoder.count_neighbor_sign_effect`（numpy 路径）得到每位置
     [pos, neg] 证据（含 RAE direct 项）→ 非零位置 = pos>0 或 neg>0。

口径要点：
  - 证据在 **NN 截断后的采样序列** 上算（CN 交集 = 两侧截断后历史的交集）——与模型所见一致；
  - 有效序列长度 L_eff = 1 + min(len(历史), NN-1)；比例分母用 L_eff（不含右侧零填充）；
  - 同时给出"仅间接（共邻居）证据"与"仅 direct（RAE 重复位置）证据"的分解；
  - 输出：控制台表 + results/bte_sparsity/<task>_<dataset>.npz（逐边掩码，供 D3 消费）+ summary CSV。

用法（仓库根目录，torch CPU 即可）：
    python tools/verify/bte_sparsity_stats.py --task linksign
    python tools/verify/bte_sparsity_stats.py --task sign --datasets RedditHyperlinkBody
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils import accel as _accel  # noqa: E402
from utils.DataLoader import get_link_prediction_data  # noqa: E402
from utils.direct_neighbor_sampler import get_neighbor_sampler  # noqa: E402
from models.NeighborInteractEncoder import NeighborCooccurrenceEncoder  # noqa: E402

# 主实验最优配置（run_experiments.py::TASK_DATASET_BEST_PARAMS）
BEST = {
    "sign": {
        "WikiVote": (40, 15),
        "BitcoinAlpha": (40, 15),
        "BitcoinOTC": (40, 15),
        "RedditHyperlinkTitle": (100, 1),
        "RedditHyperlinkBody": (60, 1),
    },
    "linksign": {
        "WikiVote": (15, 10),
        "BitcoinAlpha": (40, 15),
        "BitcoinOTC": (80, 5),
        "RedditHyperlinkTitle": (60, 1),
        "RedditHyperlinkBody": (80, 3),
    },
}
TAIL = {
    "WikiVote": 20000,
    "RedditHyperlinkTitle": 20000,
    "RedditHyperlinkBody": 20000,
}


def truncate_and_pad(
    node_ids: np.ndarray,
    node_interact_times: np.ndarray,
    lists_ids: list,
    lists_times: list,
    lists_signs: list,
    max_input_sequence_length: int,
):
    """复刻 SignDyGFormer.pad_sequences（patch_size=1）：返回 (ids, times, signs, L_eff)。"""
    ms = 0
    for i in range(len(lists_ids)):
        if len(lists_ids[i]) > max_input_sequence_length - 1:
            lists_ids[i] = lists_ids[i][-(max_input_sequence_length - 1):]
            lists_times[i] = lists_times[i][-(max_input_sequence_length - 1):]
            lists_signs[i] = lists_signs[i][-(max_input_sequence_length - 1):]
        ms = max(ms, len(lists_ids[i]))
    ms += 1  # pos0 自身
    B = len(node_ids)
    ids = np.zeros((B, ms), dtype=np.longlong)
    times = np.zeros((B, ms), dtype=np.float32)
    signs = np.zeros((B, ms), dtype=np.int8)
    L_eff = np.zeros(B, dtype=np.int64)
    for i in range(B):
        ids[i, 0] = node_ids[i]
        times[i, 0] = node_interact_times[i]
        signs[i, 0] = 0  # 2026-09-09 修复：pos0 sign 恒 0（防标签泄漏）
        n = len(lists_ids[i])
        L_eff[i] = n + 1
        if n > 0:
            ids[i, 1:n + 1] = lists_ids[i]
            times[i, 1:n + 1] = lists_times[i]
            signs[i, 1:n + 1] = lists_signs[i]
    return ids, times, signs, L_eff


def run_dataset(task: str, ds: str, batch_size: int, out_dir: pathlib.Path, verbose: bool):
    nn_, lf = BEST[task][ds]
    print(f"[{task}/{ds}] NN={nn_} LF={lf} tail={TAIL.get(ds)} 载入数据…", flush=True)
    (_, _, full_data, _, _, test_data, _, _) = get_link_prediction_data(
        dataset_name=ds,
        val_ratio=0.15,
        test_ratio=0.15,
        tail_num=TAIL.get(ds),
    )
    sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy="recent",
        time_scaling_factor=1e-6,
        seed=1,
        common_neighbor_look_forward=lf,
        ras_look_forward=None,
        module_repeat_aware_sampler=True,      # full 配置：RAS 开
        module_common_neighbor_sampler=True,   # full 配置：CNAS 开
    )
    enc = NeighborCooccurrenceEncoder(
        neighbor_co_occurrence_feat_dim=1,
        device="cpu",
        module_repeat_aware_sign_encoder=True,  # full 配置：RAE 开
        module_balance_theory_encoder=True,
        module_common_neighbor_encoder=True,
        time_decay_lambda=None,
    )
    try:
        _accel.on = False  # 强制 numpy 路径（本地无扩展也一致）
    except Exception:
        pass

    src = test_data.src_node_ids
    dst = test_data.dst_node_ids
    ts = test_data.node_interact_times
    sg = test_data.node_interact_sign
    n = len(src)

    ratio = np.full(n, np.nan)
    all_zero = np.zeros(n, dtype=bool)
    ratio_indirect = np.full(n, np.nan)
    direct_only = np.zeros(n, dtype=bool)
    l_eff_arr = np.zeros(n, dtype=np.int64)

    for b0 in range(0, n, batch_size):
        b1 = min(n, b0 + batch_size)
        s_ids, d_ids, t_ids = src[b0:b1], dst[b0:b1], ts[b0:b1]
        (s_list, _, s_times, s_signs, d_list, _, d_times, d_signs) = sampler.history_neighbors_sampling(
            s_ids, d_ids, t_ids
        )
        s_pid, s_pt, s_ps, s_le = truncate_and_pad(s_ids, t_ids, s_list, s_times, s_signs, nn_)
        d_pid, d_pt, d_ps, d_le = truncate_and_pad(d_ids, t_ids, d_list, d_times, d_signs, nn_)
        s_eff, d_eff = s_pt[:, :, None], d_pt[:, :, None]
        s_pe, d_pe = enc.count_neighbor_sign_effect(
            src_nodes=s_ids,
            dst_nodes=d_ids,
            src_padded_nodes_neighbor_ids=s_pid,
            dst_padded_nodes_neighbor_ids=d_pid,
            src_padded_nodes_neighbor_sign=s_ps,
            dst_padded_nodes_neighbor_sign=d_ps,
            node_interact_times=t_ids,
            src_padded_nodes_neighbor_times=s_pt,
            dst_padded_nodes_neighbor_times=d_pt,
        )
        s_pe = s_pe.numpy()
        d_pe = d_pe.numpy()
        for i in range(b1 - b0):
            le_s, le_d = int(s_le[i]), int(d_le[i])
            nz_s = (s_pe[i, :le_s, 0] > 0) | (s_pe[i, :le_s, 1] > 0)
            nz_d = (d_pe[i, :le_d, 0] > 0) | (d_pe[i, :le_d, 1] > 0)
            # direct（RAE）位置：历史 id == 对向节点
            dir_s = np.zeros(le_s, dtype=bool)
            dir_d = np.zeros(le_d, dtype=bool)
            if le_s > 1:
                dir_s[1:] = s_pid[i, 1:le_s] == d_ids[i]
            if le_d > 1:
                dir_d[1:] = d_pid[i, 1:le_d] == s_ids[i]
            nz_ind_s = nz_s & ~dir_s
            nz_ind_d = nz_d & ~dir_d
            le = le_s + le_d
            nz = int(nz_s.sum() + nz_d.sum())
            nz_ind = int(nz_ind_s.sum() + nz_ind_d.sum())
            ratio[b0 + i] = nz / le if le > 0 else 0.0
            ratio_indirect[b0 + i] = nz_ind / le if le > 0 else 0.0
            all_zero[b0 + i] = nz == 0
            direct_only[b0 + i] = (nz > 0) and (nz_ind == 0)
            l_eff_arr[b0 + i] = le
        if verbose and (b0 // batch_size) % 10 == 0:
            print(f"  {b1}/{n}", flush=True)

    real = sg != 0
    def q(a, p):
        return float(np.nanquantile(a, p))
    summary = {
        "task": task,
        "dataset": ds,
        "nn": nn_,
        "lf": lf,
        "n_test": int(n),
        "n_real(sign!=0)": int(real.sum()),
        "D1_mean_nonzero_ratio_real": float(np.nanmean(ratio[real])),
        "D1_p25": q(ratio[real], 0.25),
        "D1_p50": q(ratio[real], 0.50),
        "D1_p75": q(ratio[real], 0.75),
        "D1mean_indirect_only_real": float(np.nanmean(ratio_indirect[real])),
        "D2_allzero_share_real": float(all_zero[real].mean()),
        "D2_allzero_share_all": float(all_zero.mean()),
        "D2_directonly_share_real": float(direct_only[real].mean()),
        "avg_L_eff_real": float(l_eff_arr[real].mean()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{task}_{ds}.npz",
        ratio=ratio,
        all_zero=all_zero,
        direct_only=direct_only,
        L_eff=l_eff_arr,
        sign=sg,
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sign", "linksign", "both"], default="both")
    ap.add_argument("--datasets", nargs="+", default=list(BEST["sign"].keys()))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out-dir", default="results/bte_sparsity")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    tasks = ["sign", "linksign"] if args.task == "both" else [args.task]
    rows = []
    for task in tasks:
        for ds in args.datasets:
            rows.append(run_dataset(task, ds, args.batch_size, ROOT / args.out_dir, not args.quiet))
    print("\n== 汇总（真实边 sign!=0） ==")
    hdr = f"{'task':8} {'dataset':22} {'NN':>3} {'LF':>3} | {'D1 均值':>8} {'p25':>7} {'p50':>7} {'p75':>7} | {'D2 全零%':>9} {'仅RAE%':>7} {'L_eff':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['task']:8} {r['dataset']:22} {r['nn']:>3} {r['lf']:>3} | "
            f"{r['D1_mean_nonzero_ratio_real']:>8.4f} {r['D1_p25']:>7.4f} {r['D1_p50']:>7.4f} {r['D1_p75']:>7.4f} | "
            f"{100*r['D2_allzero_share_real']:>8.2f}% {100*r['D2_directonly_share_real']:>6.2f}% {r['avg_L_eff_real']:>6.1f}"
        )
    import csv
    csv_path = ROOT / args.out_dir / "summary_D1D2.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[saved] {csv_path}")


if __name__ == "__main__":
    main()
