# -*- coding: utf-8 -*-
"""G1 门控冒烟：features_on == features_off * evidence_mask（同权重切换旗标）。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
from utils import accel as _accel  # noqa: E402

try:
    _accel.on = False
except Exception:
    pass

from models.NeighborInteractEncoder import EncodeType, NeighborCooccurrenceEncoder  # noqa: E402

enc = NeighborCooccurrenceEncoder(
    neighbor_co_occurrence_feat_dim=2,
    device="cpu",
    module_repeat_aware_sign_encoder=True,
    module_balance_theory_encoder=True,
    module_common_neighbor_encoder=True,
)

# 合成样本：u=0, v=1；src 序列 [self=0, w=5(共邻), x=7(非共邻), 1(重复)]；
# dst 序列 [self=1, w=5(共邻), y=9(非共邻), 0(重复)]
src_pid = np.array([[0, 5, 7, 1]], dtype=np.longlong)
dst_pid = np.array([[1, 5, 9, 0]], dtype=np.longlong)
src_ps = np.array([[0, 1, 1, 1]], dtype=np.int8)
dst_ps = np.array([[0, 1, 1, 1]], dtype=np.int8)
src_pt = np.array([[10.0, 5.0, 6.0, 4.0]], dtype=np.float32)
dst_pt = np.array([[10.0, 5.0, 6.0, 4.0]], dtype=np.float32)

kwargs = dict(
    src_nodes=np.array([0]),
    dst_nodes=np.array([1]),
    src_padded_nodes_neighbor_ids=src_pid,
    dst_padded_nodes_neighbor_ids=dst_pid,
    src_padded_nodes_neighbor_sign=src_ps,
    dst_padded_nodes_neighbor_sign=dst_ps,
    node_interact_times=np.array([10.0]),
    src_padded_nodes_neighbor_times=src_pt,
    dst_padded_nodes_neighbor_times=dst_pt,
)

raw_s, raw_d = enc.count_neighbor_sign_effect(**kwargs)
mask_s = (raw_s.abs().sum(dim=-1, keepdim=True) > 0).float()
mask_d = (raw_d.abs().sum(dim=-1, keepdim=True) > 0).float()
print("raw evidence src:", raw_s.numpy().tolist(), "mask:", mask_s.squeeze(-1).numpy().tolist())
print("raw evidence dst:", raw_d.numpy().tolist(), "mask:", mask_d.squeeze(-1).numpy().tolist())
assert mask_s[0, 1].item() == 1.0, "共邻位置应有证据"
assert mask_s[0, 2].item() == 0.0, "非共邻位置应无证据"
assert mask_s[0, 3].item() == 1.0, "重复位置应有证据（RAE）"

feat_off_s, feat_off_d = enc.forward(
    src_ids=np.array([0]),
    dst_ids=np.array([1]),
    src_padded_nodes_neighbor_ids=src_pid,
    dst_padded_nodes_neighbor_ids=dst_pid,
    src_padded_nodes_neighbor_sign=src_ps,
    dst_padded_nodes_neighbor_sign=dst_ps,
    node_interact_times=np.array([10.0]),
    src_padded_nodes_neighbor_times=src_pt,
    dst_padded_nodes_neighbor_times=dst_pt,
    sample_type=EncodeType.InteractSignEffect,
)
enc.module_bte_evidence_gate = True
feat_on_s, feat_on_d = enc.forward(
    src_ids=np.array([0]),
    dst_ids=np.array([1]),
    src_padded_nodes_neighbor_ids=src_pid,
    dst_padded_nodes_neighbor_ids=dst_pid,
    src_padded_nodes_neighbor_sign=src_ps,
    dst_padded_nodes_neighbor_sign=dst_ps,
    node_interact_times=np.array([10.0]),
    src_padded_nodes_neighbor_times=src_pt,
    dst_padded_nodes_neighbor_times=dst_pt,
    sample_type=EncodeType.InteractSignEffect,
)

assert (feat_on_s * (1 - mask_s) == 0).all().item(), "门控后无证据位置未置零（src）"
assert (feat_on_d * (1 - mask_d) == 0).all().item(), "门控后无证据位置未置零（dst）"
sel = mask_s.squeeze(-1) > 0
assert np.allclose(
    feat_on_s[sel].detach().numpy(), feat_off_s[sel].detach().numpy(), atol=1e-6
), "有证据位置被改动"
print("G1 门控冒烟：PASS（无证据位置严格置零；有证据位置与关闭时逐位一致）")
