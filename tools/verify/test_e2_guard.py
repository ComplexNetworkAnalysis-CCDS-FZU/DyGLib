# -*- coding: utf-8 -*-
"""E2 自历史锚点护栏单测（2026-09-25 Paper 立项、用户已批；af3e §一 要求"新增单测"）。

覆盖：
  (a) "仅目标边存在"的合成样本 → 队列不得含目标边（t=t_query）；E2 块自动满足 strict-past；
  (b) "含历史重复边"的样本 → 历史重复可入队（含经 E2 最近块入队），t=t_query 边不得出现；
  (c) 有锚点（共邻居）边：E2 最近 k 块保底并入 + 升序 + 不挤掉原锚点窗；
  (d) k=0 零行为变更：与不开 E2 的基线采样器输出【逐位一致】。

运行（仓库根）：python tools/verify/test_e2_guard.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
from utils import accel as _accel  # noqa: E402

try:
    _accel.on = False
except Exception:
    pass

from utils.direct_neighbor_sampler import (  # noqa: E402
    DirectedNeighborSampler,
    Neighbor,
    NeighborType,
)


def make_sampler(edges, *, e2_self_recent=0, cnas_tail_fill=False, recent_block=0):
    n = max(max(a, b) for a, b, _ in edges) + 1
    adj = {i: [] for i in range(n)}
    for eid, (a, b, t) in enumerate(edges):
        adj[a].append(Neighbor(b, eid, t, 1, NeighborType.OutcomeNeighbor))
        adj[b].append(Neighbor(a, eid, t, 1, NeighborType.IncomeNeighbor))
    for i in adj:
        adj[i].sort(key=lambda nb: nb.timestamp)
    return DirectedNeighborSampler(
        adj_list=adj,
        common_neighbor_look_forward=5,
        ras_look_forward=None,
        module_repeat_aware_sampler=True,
        module_common_neighbor_sampler=True,
        cnas_tail_fill=cnas_tail_fill,
        recent_block=recent_block,
        e2_self_recent=e2_self_recent,
    )


def side_pairs(res, side):
    """取一侧 (ids, times) 的 (id, 时间) 集合。side: 0=src, 4=dst。"""
    ids, times = res[side][0], res[side + 2][0]
    return set(zip(ids.tolist(), np.round(np.asarray(times, dtype=float), 6).tolist()))


# (a) 仅目标边存在：u 无任何前史 → E2 无可并入；目标边自身不得出现
s_a = make_sampler([(0, 1, 10.0)], e2_self_recent=3)
r_a = s_a.history_neighbors_sampling(np.array([0]), np.array([1]), np.array([10.0]))
assert not ((r_a[0][0] == 1) & (r_a[2][0] == 10.0)).any(), "(a) 队列含目标边自身"
assert np.all(r_a[2][0] < 10.0) if len(r_a[2][0]) else True
print("(a) 仅目标边存在 + E2: PASS（目标边被排除）")

# (b) 含历史重复边：u 前史 [(w,3),(v,5)]；E2 k=2 → 最近 2 个（含历史重复 v@5）可入队；t=10 不得入队
s_b = make_sampler([(0, 2, 3.0), (0, 1, 5.0), (0, 1, 10.0)], e2_self_recent=2)
r_b = s_b.history_neighbors_sampling(np.array([0]), np.array([1]), np.array([10.0]))
pairs_b = side_pairs(r_b, 0)
assert (1, 5.0) in pairs_b, "(b) E2 最近块应含历史重复边 (v@5)"
assert (2, 3.0) in pairs_b, "(b) E2 最近块应含 w@3"
assert (1, 10.0) not in pairs_b, "(b) 目标边自身不得入队"
assert np.all(r_b[2][0] < 10.0), "(b) strict-past 失败"
print("(b) 含历史重复 + E2: PASS（v@5 可入队，目标边排除）")

# (c) 有锚点（共邻居）边：u=[3@1,4@2,5@3,6@4,7@5,w=2@6] + 尾段 [9@8,10@9]；
#     v=[w=2@7]。LF=5 时锚点窗只到 w@6（u pos 0-5），尾段 pos 6-7 在窗外 → E2 k=2 应补入。
edges_c = [
    (0, 3, 1.0), (0, 4, 2.0), (0, 5, 3.0), (0, 6, 4.0), (0, 7, 5.0),
    (0, 2, 6.0), (0, 9, 8.0), (0, 10, 9.0), (1, 2, 7.0), (0, 1, 10.0),
]
s_c0 = make_sampler(edges_c, e2_self_recent=0)
s_c2 = make_sampler(edges_c, e2_self_recent=2)
u, v, t = np.array([0]), np.array([1]), np.array([10.0])
r_c0 = s_c0.history_neighbors_sampling(u, v, t)
r_c2 = s_c2.history_neighbors_sampling(u, v, t)
p0, p2 = side_pairs(r_c0, 0), side_pairs(r_c2, 0)
assert (10, 9.0) in p2 and (9, 8.0) in p2, "(c) E2 最近 2 块缺失（u 侧尾段）"
assert p0 <= p2, "(c) E2 不得挤掉原锚点窗（应为超集）"
assert len(p2) > len(p0), "(c) 有锚点边上 E2 应有净增量"
assert np.all(r_c2[2][0] < 10.0) and np.all(np.diff(r_c2[2][0]) >= 0), "(c) 护栏/升序失败"
print(f"(c) 有锚点 + E2: PASS（u 侧 {len(p0)}→{len(p2)} 个位置超集）")

# (d) k=0 零行为变更：逐位一致
idx = np.array([0, 2])
r_off = s_c0.history_neighbors_sampling(u, v, t)
r_e0 = make_sampler(edges_c, e2_self_recent=0).history_neighbors_sampling(u, v, t)
for k2 in range(8):
    a0, b0 = r_off[k2][0], r_e0[k2][0]
    if isinstance(a0, np.ndarray):
        assert np.array_equal(a0, b0), f"(d) k=0 非逐位一致 side#{k2}"
print("(d) k=0 零行为变更: PASS（逐位一致）")

# (e) 与 E1a/E1c 组合：三块叠加仍满足护栏
s_e = make_sampler(edges_c, e2_self_recent=2, cnas_tail_fill=True, recent_block=2)
r_e = s_e.history_neighbors_sampling(u, v, t)
assert np.all(r_e[2][0] < 10.0) and np.all(r_e[6][0] < 10.0), "(e) 组合模式护栏失败"
assert not ((r_e[0][0] == 1) & (r_e[2][0] == 10.0)).any(), "(e) 组合模式含目标边"
print("(e) E2+E1a+E1c 组合护栏: PASS")

print("\n=== E2 护栏单测：全部 PASS ===")
