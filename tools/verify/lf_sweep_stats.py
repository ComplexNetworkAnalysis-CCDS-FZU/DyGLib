# -*- coding: utf-8 -*-
"""LF 结构分解 vs E1c（2026-09-25，用户问询：E1c 后可否降 LF）。

问题背景：C/R 锚点 = BTE 可编码位；LF 窗 = 锚点局部上下文（非可编码）。
本工具量化：LF ∈ {1,3,5,10,15} 下（各数据集 linksign 最佳 NN；m ∈ {0,10,80}）：
  union=采样窗并集大小；kept=模型侧截断后（cap=NN+m-1）；
  C_kept/R_kept=存活的可编码锚点（共邻居/重复）；ctx_kept=存活上下文位；
  anchor survival=锚点存活率 —— 检验「LF 增宽是否挤占锚点预算」。
输出：results/lf_sweep_stats_20260925.txt
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from utils import accel as _accel  # noqa: E402

try:
    _accel.on = False
except Exception:
    pass

from utils.DataLoader import get_link_prediction_data  # noqa: E402
from utils.direct_neighbor_sampler import get_neighbor_sampler  # noqa: E402

CFG = {
    "WikiVote": 15,
    "RedditHyperlinkTitle": 60,
    "RedditHyperlinkBody": 80,
    "BitcoinAlpha": 40,
    "BitcoinOTC": 80,
}
LF_LIST = (1, 3, 5, 10, 15)
M_LIST = (0, 10, 80)
N_EDGE = 600

lines: list[str] = []
ap = lines.append

for ds, nn in CFG.items():
    t0 = time.time()
    (_, _, full_data, _, _, test_data, _, _) = get_link_prediction_data(
        dataset_name=ds, val_ratio=0.15, test_ratio=0.15, tail_num=20000
    )
    N = len(test_data.src_node_ids)
    idxs = np.unique(np.linspace(0, N - 1, N_EDGE).astype(int))

    ap(f"===== {ds}（linksign 最佳 NN={nn}；测试边 {len(idxs)}）=====")
    for m in M_LIST:
        cap = nn + m - 1 if m > 0 else nn - 1
        cap = max(1, cap)
        ap(f"  -- m={m}（cap={cap}）--")
        ap(f"     LF | union/侧 | kept/侧 | C存活 | R存活 | 上下文位 | 锚点存活率")
        for lf in LF_LIST:
            kw = dict(
                sample_neighbor_strategy="recent",
                time_scaling_factor=1e-6,
                seed=1,
                common_neighbor_look_forward=lf,
                ras_look_forward=None,
                module_repeat_aware_sampler=True,
                recent_block=m,
            )
            s = get_neighbor_sampler(data=full_data, module_common_neighbor_sampler=True, **kw)
            nb = s.undirected_nodes_neighbor
            un, kp, ck, rk, ctx = [], [], [], [], []
            surv = []
            for j in idxs:
                u, v = int(test_data.src_node_ids[j]), int(test_data.dst_node_ids[j])
                t = float(test_data.node_interact_times[j])
                out = s.history_neighbors_sampling(np.array([u]), np.array([v]), np.array([t]))
                for side, (node, counterpart) in enumerate(((u, v), (v, u))):
                    ids_all = nb.ids[node]
                    n = int(np.searchsorted(nb.times[node], t))
                    if n == 0:
                        continue
                    hist = ids_all[:n]
                    other = v if side == 0 else u
                    n_o = int(np.searchsorted(nb.times[other], t))
                    if n_o > 0:
                        common = np.setdiff1d(
                            np.intersect1d(np.unique(hist), np.unique(nb.ids[other][:n_o])),
                            np.array([counterpart]),
                        )
                    else:
                        common = np.array([], dtype=hist.dtype)
                    nC_all = int(np.isin(hist, common).sum()) if len(common) else 0
                    nR_all = int((hist == counterpart).sum())
                    o_ids = out[0 if side == 0 else 4][0]
                    L = len(o_ids)
                    un.append(L)
                    take = min(cap, L)
                    tail_ids = o_ids[L - take :]
                    nC_k = int(np.isin(tail_ids, common).sum()) if len(common) else 0
                    nR_k = int((tail_ids == counterpart).sum())
                    kp.append(take)
                    ck.append(nC_k)
                    rk.append(nR_k)
                    ctx.append(take - nC_k - nR_k)
                    if (nC_all + nR_all) > 0:
                        surv.append((nC_k + nR_k) / (nC_all + nR_all))
            ap(
                f"     {lf:>2} | {np.mean(un):7.1f} | {np.mean(kp):7.1f} | {np.mean(ck):5.2f} | {np.mean(rk):5.2f} | {np.mean(ctx):7.1f} | "
                f"{np.mean(surv) * 100 if surv else float('nan'):5.1f}%（n={len(surv)}）"
            )
    ap(f"  [用时 {time.time() - t0:.1f}s]")
    print(f"{ds} done ({time.time() - t0:.1f}s)")

out = pathlib.Path(__file__).resolve().parents[2] / "results/lf_sweep_stats_20260925.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\n写出 {out}")
