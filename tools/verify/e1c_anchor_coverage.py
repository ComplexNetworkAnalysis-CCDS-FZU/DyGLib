# -*- coding: utf-8 -*-
"""E1c 最近块 vs RAS 重复锚点（R 锚点）相互作用诊断（2026-09-25，用户问询）。

问题：m 是否/多大程度上「覆盖」了重复交互采样锚点（RAS 的 R 锚点）？
口径（真实数据、测试期边、各数据集 linksign 现行最佳配置）：
  1. R 位置覆盖率 @m∈{10,30,80}：R 锚点位置落入 [n-m, n) 的占比（按侧，R-bearing 侧）；
  2. R 回看窗全覆盖率：p - k >= n - m（窗左缘也在块内；未计锚点截断，属乐观口径）；
  3. 全 R 覆盖边占比 @80（两侧所有 R 都进块）；
  4. 块「吸收度」：|U(m=80)| - |W(m=0)| = 块去重后的净增量；吸收度 = 1 - 净增量/min(m, n)；
     完全吸收（净增量=0）侧占比；
  5. 静默节点：n <= m 的侧占比（该侧块=全历史 → CNAS/RAS 结构被完全旁路）；
  6. 模型侧截断后的「有效 R 数」（cap=NN+m-1，保留最近 cap 条）：
     对照 m=0（cap=NN-1）vs m=80（cap=NN+79）——E1c 把截断视界从 ~NN 扩到 ~NN+m。
输出：results/e1c_anchor_coverage_20260925.txt
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

# linksign 现行最佳配置（TASK_DATASET_BEST_PARAMS）
CFG = {
    "WikiVote": (15, 10),
    "RedditHyperlinkTitle": (60, 1),
    "RedditHyperlinkBody": (80, 3),
    "BitcoinAlpha": (40, 15),
    "BitcoinOTC": (80, 5),
}
M_LIST = (3, 10, 30, 80)
N_EDGE = 600

lines: list[str] = []
ap = lines.append


def eff_r_on_side(out_ids, out_times, cap, counterpart):
    L = len(out_ids)
    if L == 0:
        return 0
    tail = slice(L - cap, L) if L > cap else slice(0, L)
    return int(np.sum(out_ids[tail] == counterpart))


for ds, (nn, k) in CFG.items():
    t0 = time.time()
    (_, _, full_data, _, _, test_data, _, _) = get_link_prediction_data(
        dataset_name=ds, val_ratio=0.15, test_ratio=0.15, tail_num=20000
    )
    kw = dict(
        sample_neighbor_strategy="recent",
        time_scaling_factor=1e-6,
        seed=1,
        common_neighbor_look_forward=k,
        ras_look_forward=None,
        module_repeat_aware_sampler=True,
    )
    s0 = get_neighbor_sampler(data=full_data, module_common_neighbor_sampler=True, recent_block=0, **kw)
    s3 = get_neighbor_sampler(data=full_data, module_common_neighbor_sampler=True, recent_block=3, **kw)
    s80 = get_neighbor_sampler(data=full_data, module_common_neighbor_sampler=True, recent_block=80, **kw)
    s10 = get_neighbor_sampler(data=full_data, module_common_neighbor_sampler=True, recent_block=10, **kw)
    s30 = get_neighbor_sampler(data=full_data, module_common_neighbor_sampler=True, recent_block=30, **kw)
    nb = s0.undirected_nodes_neighbor

    N = len(test_data.src_node_ids)
    idxs = np.unique(np.linspace(0, N - 1, N_EDGE).astype(int))

    n_sides = 0
    n_sides_le80 = 0
    r_sides = 0
    r_cover3, r_cover10, r_cover30, r_cover80, r_win_in80 = [], [], [], [], []
    r_len = []
    edges_all_r_covered = 0
    edges_with_r = 0
    fallback_edges = 0
    absorb_sides, fully_absorbed_sides, new_side_vals = [], 0, []
    eff0_list, eff80_list = [], []
    edges_eff_up = edges_eff_same = edges_eff_down = 0

    for j in idxs:
        u, v = int(test_data.src_node_ids[j]), int(test_data.dst_node_ids[j])
        t = float(test_data.node_interact_times[j])
        uu, vv, tt = np.array([u]), np.array([v]), np.array([t])
        o0 = s0.history_neighbors_sampling(uu, vv, tt)
        o3 = s3.history_neighbors_sampling(uu, vv, tt)
        o80 = s80.history_neighbors_sampling(uu, vv, tt)
        o10 = s10.history_neighbors_sampling(uu, vv, tt)
        o30 = s30.history_neighbors_sampling(uu, vv, tt)

        edge_has_r = False
        edge_all_covered = True
        eff0 = eff80 = 0
        both_fallback = True
        for side, (node, counterpart) in enumerate(((u, v), (v, u))):
            ids = nb.ids[node]
            times = nb.times[node]
            n = int(np.searchsorted(times, t))
            if n == 0:
                continue
            n_sides += 1
            if n <= 80:
                n_sides_le80 += 1
            hist_ids = ids[:n]
            R = np.where(hist_ids == counterpart)[0]
            # 分支：m=0 输出长度 == n 视为回退/无差异（块在语义上无操作）
            o0_side = o0[0 if side == 0 else 4][0]
            o80_side = o80[0 if side == 0 else 4][0]
            if len(o0_side) != n:
                both_fallback = False
            # 块吸收度
            denom = min(80, n)
            new = len(o80_side) - len(o0_side)
            new_side_vals.append(new)
            if denom > 0:
                absorb_sides.append(1.0 - min(new, denom) / denom)
            if new == 0:
                fully_absorbed_sides += 1
            # R 覆盖
            if len(R) > 0:
                edge_has_r = True
                r_sides += 1
                r_len.append(len(R))
                tail3 = R >= n - 3
                tail10 = R >= n - 10
                tail30 = R >= n - 30
                tail80 = R >= n - 80
                r_cover3.append(tail3.mean())
                r_cover10.append(tail10.mean())
                r_cover30.append(tail30.mean())
                r_cover80.append(tail80.mean())
                r_win_in80.append(((R - k) >= n - 80).mean())
                if not tail80.all():
                    edge_all_covered = False
            # 有效 R（模型侧截断后）
            cp_u = v if side == 0 else u
            idx_side = 0 if side == 0 else 4
            eff0 += eff_r_on_side(o0[idx_side][0], o0[idx_side + 2][0], max(1, nn - 1), cp_u)
            eff80 += eff_r_on_side(o80[idx_side][0], o80[idx_side + 2][0], nn + 80 - 1, cp_u)
        if edge_has_r:
            edges_with_r += 1
            if edge_all_covered:
                edges_all_r_covered += 1
            eff0_list.append(eff0)
            eff80_list.append(eff80)
            if eff80 > eff0:
                edges_eff_up += 1
            elif eff80 == eff0:
                edges_eff_same += 1
            else:
                edges_eff_down += 1
        if both_fallback and (len(o0[0][0]) == len(nb.ids[u][: np.searchsorted(nb.times[u], t)]) if np.searchsorted(nb.times[u], t) > 0 else True):
            fallback_edges += 1

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    ap(f"===== {ds}（linksign 配置 NN={nn}, LF={k}；测试边 {len(idxs)} 采样）=====")
    ap(f"  侧数(有历史) {n_sides}；n<=80 侧占比 {n_sides_le80 / max(1, n_sides) * 100:.1f}%（块=全历史）")
    ap(f"  R-bearing 侧占比 {r_sides / max(1, n_sides) * 100:.1f}%；R 数/侧 均值 {np.mean(r_len) if r_len else 0:.2f}")
    ap(f"  R 位置覆盖率：@m=3 {np.mean(r_cover3) if r_cover3 else 0:.3f} | @m=10 {np.mean(r_cover10) if r_cover10 else 0:.3f} | @m=30 {np.mean(r_cover30) if r_cover30 else 0:.3f} | @m=80 {np.mean(r_cover80) if r_cover80 else 0:.3f}")
    ap(f"  R 回看窗整窗入块率 @80 {np.mean(r_win_in80) if r_win_in80 else 0:.3f}（乐观口径，未计锚点间截断）")
    ap(f"  全 R 覆盖边占比 @80：{edges_all_r_covered}/{edges_with_r} = {edges_all_r_covered / max(1, edges_with_r) * 100:.1f}%")
    ap(f"  块净增量/侧 均值 {np.mean(new_side_vals) if new_side_vals else 0:.2f}（上限 80）；吸收度均值 {np.mean(absorb_sides) if absorb_sides else 0:.3f}；完全吸收侧占比 {fully_absorbed_sides / max(1, n_sides) * 100:.1f}%")
    ap(f"  回退边占比（两侧 m0=全历史）{fallback_edges / len(idxs) * 100:.1f}%")
    if eff0_list:
        ap(f"  有效 R（截断后）均值：m=0 → {np.mean(eff0_list):.2f}；m=80 → {np.mean(eff80_list):.2f}"
           f"；R 边中 增大 {edges_eff_up} / 不变 {edges_eff_same} / 减少 {edges_eff_down}")
    else:
        ap("  有效 R：无 R-bearing 边")
    ap(f"  [用时 {time.time() - t0:.1f}s]")
    print(f"{ds} done ({time.time() - t0:.1f}s)")

out = pathlib.Path(__file__).resolve().parents[2] / "results/e1c_anchor_coverage_20260925.txt"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\n写出 {out}")
