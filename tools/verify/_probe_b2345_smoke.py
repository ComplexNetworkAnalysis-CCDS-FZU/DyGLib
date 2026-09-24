# -*- coding: utf-8 -*-
"""B2/B3/B4/B5 冒烟（同权重切换旗标；含 A 族互斥与参数量核对）。

校验点：
  1) 全默认关 = 与 node_sign_effect_mapping 直算逐位一致（零行为变更）；
  2) B2: feat == full * 1/sqrt(max(k,1))（k = 样本非零证据位置数）；
  3) B3: 无证据位置 = marker（init = layer(0,0)），有证据位置与 full 逐位一致；
  4) B4: gains=1 时恒等；gains=1.5 时 == full + 0.5·m_p·f_p + 0.5·m_n·f_n；
  5) B5: feat == full * g，g = min(1, sqrt(n/τ))，n=0 处 = 0；
  6) A 族互斥断言（G1+B5 构造即报错）；
  7) 参数量增量：B2 +0、B3 +F、B4 +2。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from utils import accel as _accel  # noqa: E402

try:
    _accel.on = False
except Exception:
    pass

from models.NeighborInteractEncoder import EncodeType, NeighborCooccurrenceEncoder  # noqa: E402

F = 3


def build(**flags):
    return NeighborCooccurrenceEncoder(
        neighbor_co_occurrence_feat_dim=F,
        device="cpu",
        module_repeat_aware_sign_encoder=True,
        module_balance_theory_encoder=True,
        module_common_neighbor_encoder=True,
        **flags,
    )


# 合成样本（2 条）：样本0 有证据（pos=1,neg=1 及 pos=2），样本1 全零样本
src_pid = np.array(
    [
        [0, 5, 7, 1],
        [0, 8, 3, 2],
    ],
    dtype=np.longlong,
)
dst_pid = np.array(
    [
        [1, 5, 9, 0],
        [1, 4, 6, 9],
    ],
    dtype=np.longlong,
)
src_ps = np.array([[0, 1, 1, 1], [0, 0, 0, 0]], dtype=np.int8)
dst_ps = np.array([[0, 1, 1, 1], [0, 0, 0, 0]], dtype=np.int8)
src_pt = np.array([[10.0, 5.0, 6.0, 4.0], [10.0, 1.0, 2.0, 3.0]], dtype=np.float32)
dst_pt = np.array([[10.0, 5.0, 6.0, 4.0], [10.0, 1.0, 2.0, 3.0]], dtype=np.float32)

kwargs = dict(
    src_nodes=np.array([0, 2]),
    dst_nodes=np.array([1, 3]),
    src_padded_nodes_neighbor_ids=src_pid,
    dst_padded_nodes_neighbor_ids=dst_pid,
    src_padded_nodes_neighbor_sign=src_ps,
    dst_padded_nodes_neighbor_sign=dst_ps,
    node_interact_times=np.array([10.0, 10.0]),
    src_padded_nodes_neighbor_times=src_pt,
    dst_padded_nodes_neighbor_times=dst_pt,
)

enc = build()
raw_s, raw_d = enc.count_neighbor_sign_effect(**kwargs)
assert raw_s.dtype in (torch.float32, torch.float64), f"effect dtype={raw_s.dtype}"


def fwd():
    return enc.forward(
        src_ids=kwargs["src_nodes"],
        dst_ids=kwargs["dst_nodes"],
        src_padded_nodes_neighbor_ids=src_pid,
        dst_padded_nodes_neighbor_ids=dst_pid,
        src_padded_nodes_neighbor_sign=src_ps,
        dst_padded_nodes_neighbor_sign=dst_ps,
        node_interact_times=kwargs["node_interact_times"],
        src_padded_nodes_neighbor_times=src_pt,
        dst_padded_nodes_neighbor_times=dst_pt,
        sample_type=EncodeType.InteractSignEffect,
    )


# 1) 全关 = 直算 mapping
base_s, base_d = fwd()
assert torch.allclose(base_s, enc.node_sign_effect_mapping(raw_s)), "全关 ≠ mapping 直算（src）"
assert torch.allclose(base_d, enc.node_sign_effect_mapping(raw_d)), "全关 ≠ mapping 直算（dst）"
print("[1] 全默认关 == mapping 直算：PASS")

# 2) B2
enc2 = build(module_bte_b2_density_norm=True)
enc2.load_state_dict(enc.state_dict(), strict=False)
r2_s, r2_d = enc2.count_neighbor_sign_effect(**kwargs)
f2_s, f2_d = enc2.forward(
    src_ids=kwargs["src_nodes"], dst_ids=kwargs["dst_nodes"],
    src_padded_nodes_neighbor_ids=src_pid, dst_padded_nodes_neighbor_ids=dst_pid,
    src_padded_nodes_neighbor_sign=src_ps, dst_padded_nodes_neighbor_sign=dst_ps,
    node_interact_times=kwargs["node_interact_times"],
    src_padded_nodes_neighbor_times=src_pt, dst_padded_nodes_neighbor_times=dst_pt,
    sample_type=EncodeType.InteractSignEffect,
)
k = (r2_s.abs().sum(dim=-1) > 0).sum(dim=1, keepdim=True).float()  # [B,1]
exp2_s = enc.node_sign_effect_mapping(r2_s) / torch.sqrt(torch.clamp(k, min=1.0)).unsqueeze(-1)
assert torch.allclose(f2_s, exp2_s, atol=1e-6), "B2 公式不符（src）"
print(f"[2] B2 公式（1/sqrt(k)，k={k.squeeze(-1).tolist()}）：PASS")

# 3) B3
_ini_enc = build(module_bte_b3_default_marker=True)  # 独立实例：验 init（加载权重前）
_ini = _ini_enc.neighbor_sign_effect_layer(torch.zeros(1, 2))[0]
assert torch.allclose(_ini_enc.b3_default_marker.data, _ini, atol=1e-6), "B3 init ≠ layer(0,0)"
enc3 = build(module_bte_b3_default_marker=True)
enc3.load_state_dict(enc.state_dict(), strict=False)
with torch.no_grad():
    # 公式校验与 marker 取值无关：对齐到当前权重的 layer(0,0) 便于外部直算对照
    enc3.b3_default_marker.data.copy_(enc3.neighbor_sign_effect_layer(torch.zeros(1, 2))[0])
r3_s, _ = enc3.count_neighbor_sign_effect(**kwargs)
f3_s, f3_d = enc3.forward(
    src_ids=kwargs["src_nodes"], dst_ids=kwargs["dst_nodes"],
    src_padded_nodes_neighbor_ids=src_pid, dst_padded_nodes_neighbor_ids=dst_pid,
    src_padded_nodes_neighbor_sign=src_ps, dst_padded_nodes_neighbor_sign=dst_ps,
    node_interact_times=kwargs["node_interact_times"],
    src_padded_nodes_neighbor_times=src_pt, dst_padded_nodes_neighbor_times=dst_pt,
    sample_type=EncodeType.InteractSignEffect,
)
has = (r3_s.abs().sum(dim=-1, keepdim=True) > 0).float()
exp3_s = enc3.node_sign_effect_mapping(r3_s) * has + enc3.b3_default_marker.view(1, 1, -1) * (1 - has)
assert torch.allclose(f3_s, exp3_s, atol=1e-6), "B3 公式不符（src）"
print("[3] B3 公式与 init=layer(0,0)：PASS")

# 4) B4
enc4 = build(module_bte_b4_channel_gate=True)
enc4.load_state_dict(enc.state_dict(), strict=False)
r4_s, _ = enc4.count_neighbor_sign_effect(**kwargs)
f4_s, _ = enc4.forward(
    src_ids=kwargs["src_nodes"], dst_ids=kwargs["dst_nodes"],
    src_padded_nodes_neighbor_ids=src_pid, dst_padded_nodes_neighbor_ids=dst_pid,
    src_padded_nodes_neighbor_sign=src_ps, dst_padded_nodes_neighbor_sign=dst_ps,
    node_interact_times=kwargs["node_interact_times"],
    src_padded_nodes_neighbor_times=src_pt, dst_padded_nodes_neighbor_times=dst_pt,
    sample_type=EncodeType.InteractSignEffect,
)
assert torch.allclose(f4_s, enc4.node_sign_effect_mapping(r4_s), atol=1e-6), "B4 gains=1 非恒等"
with torch.no_grad():
    enc4.b4_gain_pos.fill_(1.5)
    enc4.b4_gain_neg.fill_(1.5)
pos, neg = r4_s[..., 0:1], r4_s[..., 1:2]
zero = torch.zeros_like(pos)
f_p = enc4.neighbor_sign_effect_layer(torch.cat([pos, zero], dim=-1))
f_n = enc4.neighbor_sign_effect_layer(torch.cat([zero, neg], dim=-1))
exp4 = (
    enc4.node_sign_effect_mapping(r4_s)
    + 0.5 * (pos > 0).float() * f_p
    + 0.5 * (neg > 0).float() * f_n
)
f4b_s, _ = enc4.forward(
    src_ids=kwargs["src_nodes"], dst_ids=kwargs["dst_nodes"],
    src_padded_nodes_neighbor_ids=src_pid, dst_padded_nodes_neighbor_ids=dst_pid,
    src_padded_nodes_neighbor_sign=src_ps, dst_padded_nodes_neighbor_sign=dst_ps,
    node_interact_times=kwargs["node_interact_times"],
    src_padded_nodes_neighbor_times=src_pt, dst_padded_nodes_neighbor_times=dst_pt,
    sample_type=EncodeType.InteractSignEffect,
)
assert torch.allclose(f4b_s, exp4, atol=1e-6), "B4 gains=1.5 公式不符"
print("[4] B4 恒等/缩放公式：PASS")

# 5) B5
enc5 = build(module_bte_b5_continuous_gate=True)
enc5.load_state_dict(enc.state_dict(), strict=False)
r5_s, _ = enc5.count_neighbor_sign_effect(**kwargs)
n = r5_s.sum(dim=-1).float()
nz = n[n > 0]
tau = nz.median() if nz.numel() > 0 else torch.tensor(1.0)
g = torch.clamp(torch.sqrt(n / torch.clamp(tau, min=1e-6)), max=1.0).unsqueeze(-1)
exp5 = enc5.node_sign_effect_mapping(r5_s) * g
f5_s, _ = enc5.forward(
    src_ids=kwargs["src_nodes"], dst_ids=kwargs["dst_nodes"],
    src_padded_nodes_neighbor_ids=src_pid, dst_padded_nodes_neighbor_ids=dst_pid,
    src_padded_nodes_neighbor_sign=src_ps, dst_padded_nodes_neighbor_sign=dst_ps,
    node_interact_times=kwargs["node_interact_times"],
    src_padded_nodes_neighbor_times=src_pt, dst_padded_nodes_neighbor_times=dst_pt,
    sample_type=EncodeType.InteractSignEffect,
)
assert torch.allclose(f5_s, exp5, atol=1e-6), "B5 公式不符"
assert (f5_s[n == 0] == 0).all(), "B5：n=0 位置未置零"
print(f"[5] B5 公式（τ={tau.item():.2f}；n=0 置零）：PASS")

# 6) A 族互斥
try:
    build(module_bte_evidence_gate=True, module_bte_b5_continuous_gate=True)
    raise SystemExit("[6] FAIL：A 族互斥断言未生效")
except AssertionError:
    print("[6] A 族（G1/B5/B3）互斥断言：PASS")

# 7) 参数量增量
def n_params(m):
    return sum(p.numel() for p in m.parameters())


p0 = n_params(build())
d2 = n_params(build(module_bte_b2_density_norm=True)) - p0
d3 = n_params(build(module_bte_b3_default_marker=True)) - p0
d4 = n_params(build(module_bte_b4_channel_gate=True)) - p0
d5 = n_params(build(module_bte_b5_continuous_gate=True)) - p0
assert (d2, d3, d4, d5) == (0, F, 2, 0), (d2, d3, d4, d5)
print(f"[7] 参数增量 B2={d2} B3={d3}(=F) B4={d4} B5={d5}：PASS")
print("B2/B3/B4/B5 冒烟全部 PASS")
