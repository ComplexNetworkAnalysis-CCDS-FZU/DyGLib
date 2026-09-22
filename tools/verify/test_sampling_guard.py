# -*- coding: utf-8 -*-
"""采样护栏单测（两类合成样本 + 扩展例；2026-09-22 用户批准，Paper 5c66 §三 要求）。

  (i)   "仅目标边存在"的合成样本 → 队列中不得含该边（t=t_query 事件）；
  (ii)  "含历史重复边"的样本 → 历史重复边可入队（R 锚点），t=t_query 边不得出现；
  (iii) 开启 cnas_tail_fill / recent_block 时护栏仍成立（strict-past）；
  (iv)  无锚点回退 = 全历史列表（模型侧 pad 后即"最近 N"语义）。

运行（仓库根）：python tools/verify/test_sampling_guard.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
from utils import accel as _accel  # noqa: E402

# 本地 accel ABI 与当前代码不一致（服务器上正常）：探针/单测走原路径即可。
try:
    _accel.on = False
except Exception:
    pass

from utils.direct_neighbor_sampler import (  # noqa: E402
    DirectedNeighborSampler,
    Neighbor,
    NeighborType,
)


def make_sampler(edges, *, cnas_tail_fill=True, recent_block=0):
    n = max(max(a, b) for a, b, _ in edges) + 1
    adj = {i: [] for i in range(n)}
    for eid, (a, b, t) in enumerate(edges):
        adj[a].append(Neighbor(b, eid, t, 1, NeighborType.OutcomeNeighbor))
        adj[b].append(Neighbor(a, eid, t, 1, NeighborType.IncomeNeighbor))
    # 采样器契约：每个节点的邻居表按交互时间升序（与真实数据构造一致）。
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
    )


# (i) 仅目标边存在
s1 = make_sampler([(0, 1, 10.0)])
r1 = s1.history_neighbors_sampling(np.array([0]), np.array([1]), np.array([10.0]))
for side, counterpart in ((0, 1), (4, 0)):
    ids, times = r1[side][0], r1[side + 2][0]
    assert np.all(times < 10.0) if len(times) else True
    assert not ((ids == counterpart) & (times == 10.0)).any(), "队列含目标边自身"
print(f"(i)  仅目标边: PASS（src={len(r1[0][0])} dst={len(r1[4][0])} 条；目标边被排除）")

# (ii) 含历史重复边
s2 = make_sampler([(0, 1, 5.0), (0, 1, 10.0)])
r2 = s2.history_neighbors_sampling(np.array([0]), np.array([1]), np.array([10.0]))
src_id2, src_t2 = r2[0][0], r2[2][0]
assert not ((src_id2 == 1) & (src_t2 == 10.0)).any(), "队列含目标边自身"
assert ((src_id2 == 1) & (src_t2 == 5.0)).any(), "历史重复边应可入队（R 锚点）"
assert np.all(src_t2 < 10.0), "含非过去事件"
print("(ii) 含历史重复: PASS（t=5 入队，t=10 被排除）")

# (iii) RK 模式（最近块）护栏
s3 = make_sampler([(0, 1, 5.0), (2, 0, 3.0), (2, 1, 4.0), (0, 1, 10.0)], recent_block=3)
r3 = s3.history_neighbors_sampling(np.array([0]), np.array([1]), np.array([10.0]))
assert np.all(r3[2][0] < 10.0) and np.all(r3[6][0] < 10.0), "护栏失败（RK 模式）"
assert not ((r3[0][0] == 1) & (r3[2][0] == 10.0)).any(), "RK 模式含目标边自身"
print("(iii) RK 模式护栏: PASS")

# (iv) 无锚点回退 = 全历史（节选验证）
s4 = make_sampler([(2, 0, 3.0), (3, 1, 4.0), (0, 1, 10.0)])
r4 = s4.history_neighbors_sampling(np.array([0]), np.array([1]), np.array([10.0]))
assert len(r4[0][0]) == 1 and float(r4[2][0][0]) == 3.0, "无 CN 回退应返回完整历史"
assert len(r4[4][0]) == 1 and float(r4[6][0][0]) == 4.0, "无 CN 回退应返回完整历史"
print("(iv) 无锚点回退（全历史）: PASS（模型侧 pad 后=最近 N 语义）")

print("\n=== 采样护栏单测：全部 PASS ===")
