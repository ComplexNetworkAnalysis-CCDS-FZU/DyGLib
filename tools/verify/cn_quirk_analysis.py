"""CN 伪交集怪癖分析（本地只读，2026-09-11）。

背景：`utils/direct_neighbor_sampler.py:86` 使用
    np.intersect1d(src_neighbor, dst_neighbor, assume_unique=True)
当历史含重复 id（重复边所致）时，**不是真交集**：
numpy 实现（assume_unique=True）= 排序拼接后取"与前一个元素相等"的所有元素，
形式化结果：值 v 的 (左多重数 c1, 右多重数 c2) 满足 c1+c2 ≥ 2 即被输出（输出 k-1 份，k=c1+c2）。
→ 凡"单侧重复 ≥2 且另一侧 0 次"的邻居会变成**伪共同邻居**（论文集合语义下不应存在）。

本脚本产出：
1) 玩具探针：复现 numpy 行为（含 [1,1] vs [] → [1] 等反例）
2) 真实数据（processed_data，tail 与实验一致）：
   - 伪 CN 查询占比 / 平均伪 CN 数 / 其中属于对方节点(u/v)的占比（与 RAS 锚点重叠度）
   - 采样输出差异率：伪交集 vs 真交集（其余流程取生产实现 `look_forward_sampling` 原样）

用法（仓库根）：python tools/verify/cn_quirk_analysis.py [--queries 2000]
输出：stdout 表 + tools/verify/cn_quirk_stats.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = Path(__file__).resolve().parent / "cn_quirk_stats.json"

# linksign 生产最佳超参的 k（common_neighbors_look_forward）
DATASETS = {
    "BitcoinAlpha": dict(tail=None, k=15),
    "BitcoinOTC": dict(tail=None, k=5),
    "RedditHyperlinkBody": dict(tail=20000, k=3),
    "RedditHyperlinkTitle": dict(tail=20000, k=1),
    "WikiVote": dict(tail=20000, k=10),
}


def load_edges(name: str, tail):
    path = ROOT / "processed_data" / name / f"ml_{name}.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            rows.append((int(r["u"]), int(r["i"]), float(r["ts"]), int(r["idx"])))
    rows.sort(key=lambda x: x[3])  # 按 idx（=时间序）
    if tail is not None:
        rows = rows[-tail:]
    return rows


def build_histories(rows, n_nodes):
    ids = [[] for _ in range(n_nodes)]
    ts = [[] for _ in range(n_nodes)]
    for (u, v, t, _i) in rows:
        ids[u].append(v); ts[u].append(t)
        ids[v].append(u); ts[v].append(t)
    ids = [np.asarray(a, np.int64) for a in ids]
    ts = [np.asarray(a, np.float64) for a in ts]
    return ids, ts


def quirk_location(s_ids: np.ndarray, d_ids: np.ndarray):
    """生产路径：assume_unique=True（含怪癖）。"""
    common_vals = np.intersect1d(s_ids, d_ids, assume_unique=True)
    aware = {}
    for v in common_vals:
        aware[int(v)] = (np.where(s_ids == v)[0], np.where(d_ids == v)[0])
    return aware


def true_location(s_ids: np.ndarray, d_ids: np.ndarray):
    """修正路径：真集合交。"""
    common_vals = np.intersect1d(s_ids, d_ids)
    aware = {}
    for v in common_vals:
        aware[int(v)] = (np.where(s_ids == v)[0], np.where(d_ids == v)[0])
    return aware


def look_forward_sampling(src_neighbor_ids, dst_neighbor_ids, common_neighbors: dict, k):
    """上游 look_forward_sampling 原样（保证对比只差 intersect 一处）。"""
    src_all_common = np.sort(np.concatenate([v[0] for v in common_neighbors.values()]))
    dst_all_common = np.sort(np.concatenate([v[1] for v in common_neighbors.values()]))

    src_idxs = []
    dst_idxs = []
    for v, (src_pos, dst_pos) in common_neighbors.items():
        for idx in src_pos:
            start = max(0, idx - k)
            left, right = 0, len(src_all_common)
            while left < right:
                mid = (left + right) // 2
                if src_all_common[mid] < idx:
                    left = mid + 1
                else:
                    right = mid
            if left > 0 and src_all_common[left - 1] >= start:
                start = src_all_common[left - 1] + 1
            src_idxs.append(np.arange(start, idx + 1, dtype=np.int32))
        for idx in dst_pos:
            start = max(0, idx - k)
            left, right = 0, len(dst_all_common)
            while left < right:
                mid = (left + right) // 2
                if dst_all_common[mid] < idx:
                    left = mid + 1
                else:
                    right = mid
            if left > 0 and dst_all_common[left - 1] >= start:
                start = dst_all_common[left - 1] + 1
            dst_idxs.append(np.arange(start, idx + 1, dtype=np.int32))
    return src_idxs, dst_idxs


def sample_variant(s_ids, d_ids, k, mode):
    common = quirk_location(s_ids, d_ids) if mode == "quirk" else true_location(s_ids, d_ids)
    if len(common) == 0:
        return np.arange(len(s_ids), dtype=np.int64), np.arange(len(d_ids), dtype=np.int64)
    src_idxs, dst_idxs = look_forward_sampling(s_ids, d_ids, common, k)
    src_idxs = np.concatenate(src_idxs) if src_idxs else np.array([], np.int64)
    src_idxs.sort()
    dst_idxs = np.concatenate(dst_idxs) if dst_idxs else np.array([], np.int64)
    dst_idxs.sort()
    return src_idxs, dst_idxs


def toy_probe():
    cases = [
        ([1, 1], []),
        ([1, 1], [1]),
        ([1, 1], [1, 1]),
        ([1, 2], [2, 3]),
    ]
    print("== 玩具探针：np.intersect1d(..., assume_unique=True) ==")
    for a, b in cases:
        got = np.intersect1d(np.array(a), np.array(b), assume_unique=True).tolist()
        print(f"  a={a}, b={b} -> {got}")
    print("  （注：[1,1] vs [] -> [1]，即单侧重复被当作“交集”）\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=2000)
    args = ap.parse_args()

    print(f"numpy {np.__version__}")
    toy_probe()

    summary = {}
    print(f"{'数据集':<22s} {'查询':>5s} {'伪CN查询%':>9s} {'均伪CN':>7s} {'含u/v%':>7s} {'输出差异%':>9s} {'均Δ位置':>8s}")
    print("-" * 78)
    for name, cfg in DATASETS.items():
        rows = load_edges(name, cfg["tail"])
        n_nodes = max(max(r[0], r[1]) for r in rows) + 1
        h_ids, h_ts = build_histories(rows, n_nodes)

        rng = np.random.RandomState(20260911)
        qidx = rng.choice(len(rows), size=min(args.queries, len(rows)), replace=False)

        n_q = n_sur = n_cp = n_diff = 0
        sum_sur = sum_extra = 0
        for qi in qidx:
            u, v, t = rows[qi][0], rows[qi][1], rows[qi][2]
            iu = int(np.searchsorted(h_ts[u], t))
            iv = int(np.searchsorted(h_ts[v], t))
            s_ids, d_ids = h_ids[u][:iu], h_ids[v][:iv]

            q_common = set(np.intersect1d(s_ids, d_ids, assume_unique=True).tolist())
            t_common = set(np.intersect1d(s_ids, d_ids).tolist())
            surplus = q_common - t_common

            n_q += 1
            if surplus:
                n_sur += 1
                sum_sur += len(surplus)
            if surplus & {u, v}:
                n_cp += 1

            sq = sample_variant(s_ids, d_ids, cfg["k"], "quirk")
            st = sample_variant(s_ids, d_ids, cfg["k"], "true")
            if not (np.array_equal(sq[0], st[0]) and np.array_equal(sq[1], st[1])):
                n_diff += 1
            sum_extra += (len(sq[0]) + len(sq[1])) - (len(st[0]) + len(st[1]))

        row = dict(
            queries=n_q,
            surplus_queries=n_sur,
            surplus_rate=n_sur / n_q,
            mean_surplus=sum_sur / n_q,
            counterpart_queries=n_cp,
            counterpart_rate=n_cp / n_q,
            diff_queries=n_diff,
            diff_rate=n_diff / n_q,
            mean_extra_positions=sum_extra / n_q,
            k=cfg["k"],
            tail=cfg["tail"],
        )
        summary[name] = row
        print(f"{name:<22s} {n_q:5d} {100*row['surplus_rate']:8.1f}% {row['mean_surplus']:7.3f} "
              f"{100*row['counterpart_rate']:6.1f}% {100*row['diff_rate']:8.1f}% {row['mean_extra_positions']:8.3f}")

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] 统计已写入 {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
