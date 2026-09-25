# -*- coding: utf-8 -*-
"""E2 自历史锚点冒烟（真实数据 WikiVote）：最近 k 块保底包含性 + k=0 零变更 + 护栏 + 上限属性。"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
from utils import accel as _accel  # noqa: E402

try:
    _accel.on = False
except Exception:
    pass

from utils.DataLoader import get_link_prediction_data  # noqa: E402
from utils.direct_neighbor_sampler import get_neighbor_sampler  # noqa: E402
from utils.load_configs import SignPredictArgs  # noqa: E402

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
)
s_base = get_neighbor_sampler(data=full_data, module_common_neighbor_sampler=True, **kw)
s_hist = get_neighbor_sampler(data=full_data, module_common_neighbor_sampler=False, **kw)
k = 8
s_e2 = get_neighbor_sampler(
    data=full_data, module_common_neighbor_sampler=True, e2_self_recent=k, **kw
)

n = len(test_data.src_node_ids)
idxs = np.linspace(0, n - 1, 40).astype(int)
n_missing = 0
n_changed = 0
for j in idxs:
    u = np.array([int(test_data.src_node_ids[j])])
    v = np.array([int(test_data.dst_node_ids[j])])
    t = np.array([float(test_data.node_interact_times[j])])
    a = s_base.history_neighbors_sampling(u, v, t)
    b = s_e2.history_neighbors_sampling(u, v, t)
    h = s_hist.history_neighbors_sampling(u, v, t)  # 全历史（无窗）
    changed = False
    for side in (0, 4):
        b_ids, b_ts = b[side][0], np.asarray(b[side + 2][0], dtype=float)
        h_ids, h_ts = h[side][0], np.asarray(h[side + 2][0], dtype=float)
        a_ids, a_ts = a[side][0], np.asarray(a[side + 2][0], dtype=float)
        # 护栏：strict-past + 升序
        assert np.all(b_ts < t[0]) if len(b_ts) else True, "护栏失败"
        assert np.all(np.diff(b_ts) >= 0) if len(b_ts) > 1 else True, "升序破坏"
        if len(h_ids) == 0:
            continue
        # 最近 k 块保底：全历史最后 min(k, len) 个 (id, 时间) 必须入选
        kk = min(k, len(h_ids))
        want = set(zip(h_ids[-kk:].tolist(), np.round(h_ts[-kk:].astype(float), 6).tolist()))
        got = set(zip(b_ids.tolist(), np.round(b_ts.astype(float), 6).tolist()))
        if not want <= got:
            n_missing += 1
        # 增量统计（与 baseline 相比序列变化）
        if len(b_ids) != len(a_ids) or not np.array_equal(b_ids, a_ids):
            changed = True
    if changed:
        n_changed += 1
print(f"edges={len(idxs)}  最近 k 块缺失侧数={n_missing}  baseline 相比序列改变边数={n_changed}")
assert n_missing == 0, "最近 k 块保底失败"
assert n_changed > 0, "E2 未产生任何增量（可疑）"

# k=0 零行为变更：构造显式 e2=0 与基线对比
s_e0 = get_neighbor_sampler(
    data=full_data, module_common_neighbor_sampler=True, e2_self_recent=0, **kw
)
diffs = 0
for j in idxs[:10]:
    u = np.array([int(test_data.src_node_ids[j])])
    v = np.array([int(test_data.dst_node_ids[j])])
    t = np.array([float(test_data.node_interact_times[j])])
    a = s_base.history_neighbors_sampling(u, v, t)
    b = s_e0.history_neighbors_sampling(u, v, t)
    for s2 in range(8):
        if not np.array_equal(a[s2][0] if isinstance(a[s2][0], np.ndarray) else np.array(a[s2][0]),
                              b[s2][0] if isinstance(b[s2][0], np.ndarray) else np.array(b[s2][0])):
            diffs += 1
print(f"k=0 与基线逐位对比差异数={diffs}")
assert diffs == 0, "k=0 非零行为变更"

# 上限属性：NN + k
ap = SignPredictArgs(num_neighbors=80, e2_self_recent=10)
assert ap.max_input_sequence_length == 90, f"上限属性异常: {ap.max_input_sequence_length}"
print("上限属性 NN+k = 90: PASS")

print("\n=== E2 冒烟通过：最近 k 保底 ✓ / 升序 ✓ / strict-past ✓ / k=0 零变更 ✓ / 上限 NN+k ✓ ===")
