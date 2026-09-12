# -*- coding: utf-8 -*-
"""M4 加速接缝等价性自检（本地运行；不触服务器）。

验证内容：
  1) CLI 开关：默认启用 / --no-accel 关闭 / --accel 显式 / SIGNDYG_ACCEL 兜底（子进程隔离）
  2) K1 接缝：utils/direct_neighbor_sampler.get_common_neighbors
     原路径 vs 加速路径（Perf 参考实现做桩）——逐位一致
  3) K2 接缝：models.NeighborInteractEncoder.NeighborCooccurrenceEncoder.count_neighbor_sign_effect
     原路径 vs 加速路径（参考实现做桩；含两种 Δt 模式）——逐位一致
  4) （可选）本机已装 signdyg_accel 且 __abi__ 匹配时，再用真实内核复跑 2)/3)

用法（仓库根）：python tools/verify/test_accel_seam.py
退出码：0=全过；1=存在不一致。
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
PERF_REF = Path(r"D:\codes\SignDyG-Perf\reference")

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


# ---------- 1) CLI 开关（子进程隔离；进程内配置不可变，故分开跑） ----------
def cli_checks() -> None:
    code_no = (
        "import sys; sys.argv=['x','--no-accel']; "
        "from utils.load_configs import get_sign_prediction_args as g; "
        "a=g(); print('FIELD', a.accel)"
    )
    r = subprocess.run([sys.executable, "-c", code_no], capture_output=True, text=True, cwd=str(REPO))
    check(
        "CLI --no-accel（关闭 + 原路径打印）",
        r.returncode == 0 and "FIELD False" in r.stdout and "[accel] disabled" in r.stdout,
        (r.stdout + r.stderr).strip().replace("\n", " | ")[:200],
    )

    code_on = (
        "import sys; sys.argv=['x','--accel']; "
        "from utils.load_configs import get_sign_prediction_args as g; "
        "a=g(); print('FIELD', a.accel)"
    )
    r = subprocess.run([sys.executable, "-c", code_on], capture_output=True, text=True, cwd=str(REPO))
    check(
        "CLI --accel（显式启用）",
        r.returncode == 0 and "FIELD True" in r.stdout and "[accel] enabled" in r.stdout,
        (r.stdout + r.stderr).strip().replace("\n", " | ")[:200],
    )

    code_def = (
        "import sys; sys.argv=['x']; "
        "from utils.load_configs import get_sign_prediction_args as g; "
        "a=g(); print('FIELD', a.accel)"
    )
    r = subprocess.run([sys.executable, "-c", code_def], capture_output=True, text=True, cwd=str(REPO))
    if "signdyg_accel" in r.stderr or "契约不符" in r.stderr:
        check("CLI 默认启用（需本机 wheel）", False, "本机未装/不匹配 signdyg_accel——请先构建")
    else:
        check(
            "CLI 默认启用（启用 + 契约打印）",
            r.returncode == 0 and "FIELD True" in r.stdout and "[accel] enabled" in r.stdout,
            (r.stdout + r.stderr).strip().replace("\n", " | ")[:200],
        )

    env_off = {**os.environ, "SIGNDYG_ACCEL": "0"}
    r = subprocess.run(
        [sys.executable, "-c", code_def], capture_output=True, text=True, cwd=str(REPO), env=env_off
    )
    check(
        "ENV SIGNDYG_ACCEL=0（兜底关闭；无 flag）",
        r.returncode == 0 and "FIELD True" in r.stdout and "[accel] disabled" in r.stdout,
        (r.stdout + r.stderr).strip().replace("\n", " | ")[:200],
    )

    r = subprocess.run(
        [sys.executable, "-c", code_on], capture_output=True, text=True, cwd=str(REPO), env=env_off
    )
    check(
        "ENV=0 + --accel（CLI 优先 → 启用）",
        r.returncode == 0 and "FIELD True" in r.stdout and "[accel] enabled" in r.stdout,
        (r.stdout + r.stderr).strip().replace("\n", " | ")[:200],
    )


# ---------- 2) K1 接缝 ----------
def k1_checks(kernel_ns, label: str) -> None:
    import utils.accel as accel_mod
    from utils.direct_neighbor_sampler import DirectedNeighborSampler, Neighbor, NeighborType

    def build():
        edges = [
            (1, 3, 1.0, 1), (1, 3, 2.0, -1),  # 1 侧重复（历史含 3 两次）
            (1, 2, 3.0, 1),                   # 直接历史 u-v → RAS 锚点
            (2, 3, 4.0, 1),                   # 共同邻居 3
            (2, 0, 1.5, -1),
            (3, 0, 2.5, 1),
            (4, 5, 1.0, 1), (5, 4, 2.0, -1),  # 无共同邻居 → 回退分支
            (6, 6, 1.0, 1),                   # 自环
            (7, 1, 2.2, 1),
        ]
        adj = {i: [] for i in range(8)}
        for eid, (u, v, t, s) in enumerate(edges):
            adj[u].append(Neighbor(v, eid, t, s, NeighborType.OutcomeNeighbor))
            adj[v].append(Neighbor(u, eid, t, s, NeighborType.IncomeNeighbor))
        for node in adj:  # 上游假定按时序；玩具图显式排序
            adj[node].sort(key=lambda nb: nb.timestamp)
        return DirectedNeighborSampler(
            adj_list=adj,
            sample_neighbor_strategy="uniform",
            time_scaling_factor=0.0,
            seed=None,
            common_neighbor_look_forward=2,
            module_repeat_aware_sampler=True,
            module_common_neighbor_sampler=True,
        )

    qs = np.array([1, 2, 4, 6, 3])
    qd = np.array([2, 1, 7, 6, 0])
    qt = np.array([9.0, 9.0, 9.0, 9.0, 2.6])

    accel_mod.on = False
    accel_mod.kernel = None
    off = build().get_common_neighbors(src_node_ids=qs, dst_node_ids=qd, node_interact_times=qt)

    accel_mod.on = True
    accel_mod.kernel = kernel_ns
    on = build().get_common_neighbors(src_node_ids=qs, dst_node_ids=qd, node_interact_times=qt)

    ok, detail = True, ""
    for li, (a_list, b_list) in enumerate(zip(off, on)):
        for qi, (a_arr, b_arr) in enumerate(zip(a_list, b_list)):
            if not np.array_equal(a_arr, b_arr):
                ok = False
                detail = f"list#{li} q#{qi}: {a_arr!r} != {b_arr!r}"
                break
        if not ok:
            break
    check(f"K1 接缝等价（{label}）", ok, detail)


# ---------- 3) K2 接缝 ----------
def k2_checks(kernel_ns, label: str) -> None:
    import torch

    import utils.accel as accel_mod
    from models.NeighborInteractEncoder import NeighborCooccurrenceEncoder, TimeDecayGapMode

    src_nodes = np.array([1, 2, 3, 4], dtype=np.int64)
    dst_nodes = np.array([2, 1, 5, 6], dtype=np.int64)
    s_ids = np.array([[1, 3, 5, 7, 0, 0, 0],
                      [2, 1, 4, 0, 0, 0, 0],
                      [3, 2, 2, 8, 0, 0, 0],
                      [4, 9, 0, 0, 0, 0, 0]], dtype=np.int64)
    d_ids = np.array([[2, 3, 6, 9, 0, 0, 0],
                      [1, 5, 7, 0, 0, 0, 0],
                      [5, 2, 8, 0, 0, 0, 0],
                      [6, 1, 1, 3, 0, 0, 0]], dtype=np.int64)
    s_sign = np.array([[1, 1, -1, 1, 0, 0, 0],
                       [1, -1, 1, 0, 0, 0, 0],
                       [1, 1, -1, 1, 0, 0, 0],
                       [1, -1, 0, 0, 0, 0, 0]], dtype=np.int8)
    d_sign = np.array([[1, -1, 1, 1, 0, 0, 0],
                       [1, 1, -1, 0, 0, 0, 0],
                       [1, -1, 1, 0, 0, 0, 0],
                       [1, 1, -1, 1, 0, 0, 0]], dtype=np.int8)
    s_times = np.where(s_ids > 0, 1.0 + (s_ids % 7) * 0.5, 0.0).astype(np.float32)
    d_times = np.where(d_ids > 0, 1.0 + (d_ids % 7) * 0.5, 0.0).astype(np.float32)
    q_times = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)

    common = dict(
        src_nodes=src_nodes, dst_nodes=dst_nodes,
        src_padded_nodes_neighbor_ids=s_ids, dst_padded_nodes_neighbor_ids=d_ids,
        src_padded_nodes_neighbor_sign=s_sign, dst_padded_nodes_neighbor_sign=d_sign,
        node_interact_times=q_times,
        src_padded_nodes_neighbor_times=s_times, dst_padded_nodes_neighbor_times=d_times,
    )

    def run(decay, mode):
        enc = NeighborCooccurrenceEncoder(
            neighbor_co_occurrence_feat_dim=4, device="cpu",
            module_repeat_aware_sign_encoder=True,
            time_decay_lambda=decay, time_decay_gap_mode=mode,
        )
        return enc.count_neighbor_sign_effect(**common)

    cases = [(None, TimeDecayGapMode.STALENESS),
             (0.5, TimeDecayGapMode.STALENESS),
             (0.5, TimeDecayGapMode.GAP)]
    ok, detail = True, ""
    for decay, mode in cases:
        accel_mod.on = False
        accel_mod.kernel = None
        off = run(decay, mode)
        accel_mod.on = True
        accel_mod.kernel = kernel_ns
        on = run(decay, mode)
        if not (torch.equal(off[0], on[0]) and torch.equal(off[1], on[1])):
            ok = False
            detail = f"decay={decay}, mode={mode.value}: 不一致"
            break
    check(f"K2 接缝等价（{label}）", ok, detail)


def main() -> int:
    print(f"REPO = {REPO}")
    cli_checks()

    if not PERF_REF.exists():
        check("参考实现桩（SignDyG-Perf 工作区）", False, f"未找到 {PERF_REF}")
    else:
        sys.path.insert(0, str(PERF_REF))
        import sampling_ref
        import bte_ref

        k1_checks(types.SimpleNamespace(core_sample=sampling_ref.core_sample), "参考实现桩")
        k2_checks(types.SimpleNamespace(bte_sign_effect=bte_ref.bte_sign_effect), "参考实现桩")

    signdyg_kernel = None
    try:
        import signdyg_accel
        abi = getattr(signdyg_accel, "__abi__", None)
        if abi:
            signdyg_kernel = signdyg_accel
        else:
            print("[skip] 本机 signdyg_accel 无 __abi__（旧构建），跳过真实内核比对")
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] 本机导入 signdyg_accel 失败：{exc}")
    if signdyg_kernel is not None:
        k1_checks(signdyg_kernel, f"真实内核 {signdyg_kernel.__abi__}")
        k2_checks(signdyg_kernel, f"真实内核 {signdyg_kernel.__abi__}")

    print("== 结论 ==")
    if FAILED:
        print(f"存在 {len(FAILED)} 项失败：{FAILED}")
        return 1
    print("全部通过（接缝逐位一致）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
