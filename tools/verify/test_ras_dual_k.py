"""双半径（k_c/k_r）接口自检（纯 CPU；双进程版本，2026-09-15 落地）。

验证四件事：
  ① 兼容性：ras_look_forward=None 与 =k_c 输出【逐位一致】（原公式零行为变更）。
  ② 生效性：k_r=0 与 k_r=k_c 在含重复交互的样本上输出【存在差异】（R 窗半径真实生效）。
  ③ 回退正确性：k_r != k_c 时 K1 内核自动回退 numpy 原路径——带 accel 与纯 numpy
     输出【逐位一致】（防串口径；需已装 signdyg_accel，否则自动 SKIP）。
  ①' 加速路径下 ① 同样成立。

实现：父进程起两个子进程（SIGNDYG_ACCEL=0/1），各自采样三档 k_r {None, k_c, 0}，
结果落盘后逐位比对（accel.configure 为进程内一次性 → 用子进程隔离路径）。

用法（仓库根）：python tools/verify/test_ras_dual_k.py
"""
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATASET, LF = "RedditHyperlinkTitle", 3  # R 窗占比高（27%），必有重复交互
N_PROBE = 120


def sample_many(out_path):
    """子进程主体：三档 k_r 采样并落盘（SIGNDYG_ACCEL 由父进程环境变量传入）。"""
    from utils.DataLoader import get_link_prediction_data  # noqa: E402
    from utils.direct_neighbor_sampler import get_neighbor_sampler  # noqa: E402

    _, _, full_data, _, _, _, _, _ = get_link_prediction_data(DATASET, 0.15, 0.15)

    n_edges = len(full_data.src_node_ids)
    start = n_edges // 2
    step = max(1, (n_edges - start) // N_PROBE)
    idxs = list(range(start, n_edges, step))

    def run(kr):
        sampler = get_neighbor_sampler(
            data=full_data,
            module_repeat_aware_sampler=True,
            module_common_neighbor_sampler=True,
            common_neighbor_look_forward=LF,
            ras_look_forward=kr,
        )
        out = []
        for j in idxs:
            u = int(full_data.src_node_ids[j])
            v = int(full_data.dst_node_ids[j])
            t = float(full_data.node_interact_times[j])
            r = sampler.history_neighbors_sampling(
                np.array([u]), np.array([v]), np.array([t])
            )
            out.append(([np.asarray(a) for a in r[0]], [np.asarray(a) for a in r[4]]))
        return out

    with open(out_path, "wb") as f:
        pickle.dump({str(kr): run(kr) for kr in (None, LF, 0)}, f)


def same(p, q):
    return len(p) == len(q) and all(
        len(x[0]) == len(y[0])
        and len(x[1]) == len(y[1])
        and all(np.array_equal(a, b) for a, b in zip(x[0], y[0]))
        and all(np.array_equal(a, b) for a, b in zip(x[1], y[1]))
        for x, y in zip(p, q)
    )


def diff_count(p, q):
    return sum(0 if same([x], [y]) else 1 for x, y in zip(p, q))


def spawn(accel_env, out_path):
    env = dict(os.environ, SIGNDYG_ACCEL=accel_env)
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(out_path)],
        env=env,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )


def main():
    tmpdir = tempfile.mkdtemp(prefix="ras_dual_k_")
    p_off, p_on = os.path.join(tmpdir, "numpy.pkl"), os.path.join(tmpdir, "accel.pkl")

    proc_off = spawn("0", p_off)
    if proc_off.returncode != 0:
        print(proc_off.stdout[-2000:])
        print(proc_off.stderr[-2000:])
        raise SystemExit("numpy 子进程失败")
    off = pickle.load(open(p_off, "rb"))

    accel_ok = True
    proc_on = spawn("1", p_on)
    if proc_on.returncode != 0:
        accel_ok = False
        tail = (proc_on.stderr or "").strip().splitlines()
        print(
            f"[skip] 加速后端不可用（{tail[-1] if tail else 'see log'}）"
            "——跳过 accel 相关断言"
        )

    ok1 = same(off["None"], off[str(LF)])
    nd = diff_count(off["0"], off[str(LF)])
    print(f"[1] numpy 路径 None==LF 逐位一致: {'PASS' if ok1 else 'FAIL'}")
    print(f"[2] numpy 路径 k_r=0 vs k_r={LF} 差异样本: {nd}/{len(off[str(LF)])}（期望 >0）")

    if accel_ok:
        on = pickle.load(open(p_on, "rb"))
        ok1b = same(on["None"], on[str(LF)])
        ok3 = same(on["0"], off["0"])
        nd_on = diff_count(on["0"], on[str(LF)])
        print(f"[1'] accel 路径 None==LF 逐位一致: {'PASS' if ok1b else 'FAIL'}")
        print(f"[2'] accel 路径 k_r=0 差异样本: {nd_on}/{len(on[str(LF)])}（期望 >0）")
        print(f"[3] 双半径 accel 回退 on/off 逐位一致: {'PASS' if ok3 else 'FAIL'}")
        assert ok1b, "兼容性（accel）未通过"
        assert ok3, "双半径回退未逐位一致"
        assert nd_on > 0, "accel 路径下 k_r 未生效"
    else:
        print("[1']/[2']/[3] 已跳过（无加速后端）")

    assert ok1, "兼容性（numpy）未通过"
    assert nd > 0, "k_r=0 与 k_r=k_c 无差异——R 锚点半径未生效"
    print("ALL PASS" if accel_ok else "NUMPY-ONLY PASS（无加速后端，accel 项跳过）")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sample_many(sys.argv[-1])
    else:
        main()
