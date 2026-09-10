"""临时脚本：验证 RAS 采样差异在 pad_sequences 截断（last N-1）后是否仍存在。
结论用于解释 E-2 修复后消融 full≡base 的现象。运行完即删。"""
import numpy as np
from utils.DataLoader import get_link_prediction_data
from utils.direct_neighbor_sampler import get_neighbor_sampler

CASES = [
    ("BitcoinAlpha", 15, 40),
    ("RedditHyperlinkTitle", 1, 60),
    ("RedditHyperlinkBody", 3, 80),
]

for ds, LF, N in CASES:
    _, _, full_data, _, _, _, _, _ = get_link_prediction_data(ds, 0.15, 0.15)
    s_off = get_neighbor_sampler(
        data=full_data,
        module_repeat_aware_sampler=False,
        module_common_neighbor_sampler=True,
        common_neighbor_look_forward=LF,
    )
    s_on = get_neighbor_sampler(
        data=full_data,
        module_repeat_aware_sampler=True,
        module_common_neighbor_sampler=True,
        common_neighbor_look_forward=LF,
    )
    n_edges = len(full_data.src_node_ids)
    start = n_edges // 2
    step = max(1, (n_edges - start) // 200)
    raw_diff = trunc_diff = total = 0
    for j in range(start, n_edges, step):
        u = int(full_data.src_node_ids[j])
        v = int(full_data.dst_node_ids[j])
        t = float(full_data.node_interact_times[j])
        o = s_off.history_neighbors_sampling(np.array([u]), np.array([v]), np.array([t]))
        p = s_on.history_neighbors_sampling(np.array([u]), np.array([v]), np.array([t]))
        total += 1
        same_raw = all(np.array_equal(a, b) for a, b in zip(o[0], p[0])) and all(
            np.array_equal(a, b) for a, b in zip(o[4], p[4])
        )
        if not same_raw:
            raw_diff += 1
        ot = [a[-(N - 1):] for a in o[0]]
        pt = [b[-(N - 1):] for b in p[0]]
        od = [a[-(N - 1):] for a in o[4]]
        pd = [b[-(N - 1):] for b in p[4]]
        same_tr = all(np.array_equal(a, b) for a, b in zip(ot, pt)) and all(
            np.array_equal(a, b) for a, b in zip(od, pd)
        )
        if not same_tr:
            trunc_diff += 1
    print(
        f"{ds}: total={total} raw_diff(RAS对采样名单影响)={raw_diff} trunc_diff(截断N-1后仍不同)={trunc_diff}",
        flush=True,
    )
