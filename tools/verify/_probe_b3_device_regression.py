# -*- coding: utf-8 -*-
"""B3 构造回归：device='cuda:0' 下构造（不 .to）必须不崩（修复前在此抛设备不匹配）。"""
import sys

import torch

sys.path.insert(0, ".")
from models.NeighborInteractEncoder import NeighborCooccurrenceEncoder

assert torch.cuda.is_available(), "需要 CUDA 环境"
for flags in (
    {"module_bte_b3_default_marker": True},
    {"module_bte_b4_channel_gate": True},
    {"module_bte_b3_default_marker": True, "module_bte_b4_channel_gate": True},
):
    enc = NeighborCooccurrenceEncoder(
        neighbor_co_occurrence_feat_dim=3,
        device="cuda:0",
        module_balance_theory_encoder=True,
        **flags,
    )
    enc = enc.to("cuda:0")
    if "module_bte_b3_default_marker" in flags:
        assert enc.b3_default_marker.device.type == "cuda"
    if "module_bte_b4_channel_gate" in flags:
        assert enc.b4_gain_pos.device.type == "cuda"
    print("ok", flags)
print("B3/B4 构造回归 PASS")
