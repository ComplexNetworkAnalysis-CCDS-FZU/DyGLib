# -*- coding: utf-8 -*-
"""cnas_last_cn_stats.py — CNAS 窗口结构离线统计（纯 CPU · 轻量）。

目的：验证「CNAS 加密（k↑）向 recent-N 采样退化」的结构性残余——
  (a) **last-CN 截断**：CNAS 序列 = 各 CN 出现位窗口的并集，序列永远终止于
      「最后一个共同邻居的出现位」；其后（更新）的交互全部被丢弃。
  (b) **覆盖率**：窗口并集占全历史比例（k 的有效性 = min(k, 相邻 CN 间距) 的宏观体现）。

做法（与 bte_signal_check.py 同口径：无向历史 / 真实边 / 测试段）：
  - 对测试段抽样的边 (u, v, t)，取两侧 t 之前的全部历史邻居序列；
  - CN = 两侧历史邻居 id 的交集；记录每个 CN 在序列中的所有出现位；
  - drop_tail = 历史长度 − 1 − last_CN 位（被截掉的最新事件数）；
  - coverage(k) = 按上游 look_forward_sampling 规则（窗口 [max(0,idx−k), idx]，
    且被上一个 CN 截断）的并集覆盖率。
  - 汇总：no-CN 率、drop_tail 均值/中位/p90、coverage@LF(该数据集最优) 与 @20。

用法（仓库根目录）：
    python tools/verify/cnas_last_cn_stats.py                 # 5 数据集 × 2000 边
    python tools/verify/cnas_last_cn_stats.py --edges 1000 --datasets BitcoinAlpha RedditHyperlinkTitle
"""
from __future__ import annotations

import argparse
import bisect
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]

DATASETS = {
    "BitcoinAlpha": "processed_data/BitcoinAlpha/ml_BitcoinAlpha.csv",
    "BitcoinOTC": "processed_data/BitcoinOTC/ml_BitcoinOTC.csv",
    "WikiVote": "processed_data/WikiVote/ml_WikiVote_tail20000.csv",
    "RedditHyperlinkTitle": "processed_data/RedditHyperlinkTitle/ml_RedditHyperlinkTitle_tail20000.csv",
    "RedditHyperlinkBody": "processed_data/RedditHyperlinkBody/ml_RedditHyperlinkBody_tail20000.csv",
}

# linksign 各数据集最优 LF（= 采样窗口半径 k；主实验口径）
LF_BEST = {
    "BitcoinAlpha": 15,
    "BitcoinOTC": 5,
    "RedditHyperlinkTitle": 1,
    "RedditHyperlinkBody": 3,
    "WikiVote": 10,
}


def load_edges(path: pathlib.Path):
    df = pd.read_csv(path, usecols=lambda c: c in {"u", "i", "ts"})
    return (
        df["u"].to_numpy(np.int64),
        df["i"].to_numpy(np.int64),
        df["ts"].to_numpy(np.float64),
    )


def build_node_histories(u, i, ts):
    """无向历史：节点 -> (times, nbrs)（按时间升序）。"""
    hist = {}
    for a, b, t in zip(u, i, ts):
        for node, nbr in ((a, b), (b, a)):
            rec = hist.get(node)
            if rec is None:
                hist[node] = rec = ([], [])
            rec[0].append(t)
            rec[1].append(nbr)
    out = {}
    for node, (tt, nn) in hist.items():
        order = np.argsort(tt, kind="stable")
        out[node] = (np.asarray(tt, np.float64)[order], np.asarray(nn, np.int64)[order])
    return out


def side_stats(nbrs_self, nbrs_other, k, k_ref=20):
    """单侧统计：(has_cn, drop_tail, n_total, cov_k, cov_ref)。

    nbrs_self/nbrs_other：升序历史邻居 id 数组。
    """
    n_total = len(nbrs_self)
    if n_total == 0:
        return None
    pos_map = {}
    for p, w in enumerate(nbrs_self):
        pos_map.setdefault(int(w), []).append(p)
    other = set(int(w) for w in nbrs_other)
    positions = []
    for w in other:
        ps = pos_map.get(w)
        if ps:
            positions.extend(ps)
    if not positions:
        return (0, 0, n_total, 0.0, 0.0)
    positions.sort()
    last = positions[-1]
    drop_tail = n_total - 1 - last

    def coverage(k):
        mask = np.zeros(n_total, bool)
        for idx in positions:
            start = idx - k
            if start < 0:
                start = 0
            j = bisect.bisect_left(positions, idx)  # 第一个 >= idx
            if j > 0:
                prev = positions[j - 1]
                if prev + 1 > start:
                    start = prev + 1
            mask[start : idx + 1] = True
        return mask.sum() / n_total

    return (1, drop_tail, n_total, coverage(k), coverage(k_ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=int, default=2000, help="每数据集抽样边数")
    ap.add_argument("--test-ratio", type=float, default=0.15, help="测试段比例（取最后分位）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS),
        help="数据集子集（默认全部）",
    )
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print(
        f"{'dataset':<22s} {'LF':>3s} {'no-CN%':>7s} {'drop_tail(mean/med/p90)':>24s} "
        f"{'drop%':>7s} {'cov@LF':>7s} {'cov@20':>7s}"
    )
    print("-" * 88)
    for ds in args.datasets:
        path = ROOT / DATASETS[ds]
        u, i, ts = load_edges(path)
        n = len(u)
        lo = int(n * (1 - args.test_ratio))
        idx = np.arange(lo, n)
        if len(idx) > args.edges:
            idx = rng.choice(idx, size=args.edges, replace=False)
        hist = build_node_histories(u, i, ts)

        k = LF_BEST.get(ds, 5)
        no_cn = 0
        drops, tot, covs, cov_refs = [], [], [], []
        for e in idx:
            a, b = int(u[e]), int(i[e])
            t = float(ts[e])
            rec_a = hist.get(a)
            rec_b = hist.get(b)
            if rec_a is None or rec_b is None:
                continue
            cut_a = int(np.searchsorted(rec_a[0], t, side="left"))
            cut_b = int(np.searchsorted(rec_b[0], t, side="left"))
            na = rec_a[1][:cut_a]
            nb = rec_b[1][:cut_b]
            for self_n, other_n in ((na, nb), (nb, na)):
                st = side_stats(self_n, other_n, k)
                if st is None:
                    continue
                has_cn, d, nt, c, cr = st
                if not has_cn:
                    no_cn += 1
                    continue
                drops.append(d)
                tot.append(nt)
                covs.append(c)
                cov_refs.append(cr)

        if not drops:
            print(f"{ds:<22s} {k:>3d}  (无 CN 样本)")
            continue
        drops = np.asarray(drops)
        tot = np.asarray(tot)
        covs = np.asarray(covs)
        cov_refs = np.asarray(cov_refs)
        n_sides = len(drops) + no_cn
        drop_frac = drops / np.maximum(tot, 1)
        print(
            f"{ds:<22s} {k:>3d} {100.0*no_cn/max(n_sides,1):>6.1f}% "
            f"{drops.mean():>8.1f}/{np.median(drops):>6.0f}/{np.percentile(drops, 90):>7.0f}   "
            f"{100*drop_frac.mean():>5.1f}%   {covs.mean():>6.3f}   {cov_refs.mean():>6.3f}"
        )


if __name__ == "__main__":
    main()
