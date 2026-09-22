# -*- coding: utf-8 -*-
"""E1a 空白填补冒烟验证：同一批测试边、on/off 对比采样序列尾部。"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
from utils import accel as _accel  # noqa: E402

# 本地 .pyd 的 ABI 与仓库期望不一致 —— 直接关断（也确保走 numpy 路径，与 E1 服务器配置一致）
try:
    _accel.on = False
except Exception:
    pass

from utils.DataLoader import get_link_prediction_data  # noqa: E402
from utils.direct_neighbor_sampler import get_neighbor_sampler  # noqa: E402

ds = "WikiVote"
(_, _, full_data, _, _, test_data, _, _) = get_link_prediction_data(
    dataset_name=ds, val_ratio=0.15, test_ratio=0.15, tail_num=20000
)
kw = dict(
    sample_neighbor_strategy="recent",
    time_scaling_factor=1e-6,
    seed=1,
    common_neighbor_look_forward=10,
    ras_look_forward=None,
    module_repeat_aware_sampler=True,
    module_common_neighbor_sampler=True,
)
s0 = get_neighbor_sampler(data=full_data, **kw)
s1 = get_neighbor_sampler(data=full_data, cnas_tail_fill=True, **kw)

n = len(test_data.src_node_ids)
idxs = np.linspace(0, n - 1, 40).astype(int)
n_changed = 0
tail_lens = []
gap0, gap1 = [], []
for j in idxs:
    u = np.array([int(test_data.src_node_ids[j])])
    v = np.array([int(test_data.dst_node_ids[j])])
    t = np.array([float(test_data.node_interact_times[j])])
    a = s0.history_neighbors_sampling(u, v, t)
    b = s1.history_neighbors_sampling(u, v, t)
    a_ids, a_ts = a[0][0], a[2][0]
    b_ids, b_ts = b[0][0], b[2][0]
    changed = not (len(a_ids) == len(b_ids) and np.array_equal(a_ids, b_ids))
    if changed:
        n_changed += 1
        assert np.all(b_ts == b_ts), "升序破坏"
        assert np.all(np.diff(b_ts) >= 0), "升序断言失败"
        assert np.all(b_ts < t[0]), "护栏失败: 含非过去事件"
    if len(b_ids) > len(a_ids):
        tail_lens.append(len(b_ids) - len(a_ids))
    if len(a_ts) > 0:
        gap0.append(t[0] - float(a_ts[-1]))
    if len(b_ts) > 0:
        gap1.append(t[0] - float(b_ts[-1]))

print(f"edges={len(idxs)}  changed={n_changed}")
print(f"tail_added(mean)={np.mean(tail_lens) if tail_lens else 0:.2f}  (max={max(tail_lens) if tail_lens else 0})")
print(
    "last-event→query 时间差: "
    f"OFF mean={np.mean(gap0) if gap0 else float('nan'):.1f} max={np.max(gap0) if gap0 else float('nan'):.1f} | "
    f"ON  mean={np.mean(gap1) if gap1 else float('nan'):.1f} max={np.max(gap1) if gap1 else float('nan'):.1f}"
)
