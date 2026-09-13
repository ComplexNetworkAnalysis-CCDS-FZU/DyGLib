"""可选加速后端接入（M4 集成落地；设计：`D:\\codes\\SignDyG-Perf\\M4_INTEGRATION_DESIGN.md` §2）。

开关（命令行参数优先，其次环境变量；**默认启用**——用户 2026-09-12 口径）：
  - 命令行：训练脚本 `--accel / --no-accel`（经 `load_configs` 在参数解析后调用 `configure()`）；
  - 环境变量 `SIGNDYG_ACCEL`：`"1"`=启用 / `"0"`=关闭 / 未设=默认启用；
  - 启用 = **fail-fast**：未安装 / 契约不符 / 载入失败一律报错退出，**绝不静默回退**；
  - 关闭 = 100% 原始 Python 路径（连内核 import 都不尝试）。

契约串：`signdyg_accel.__abi__ == ABI`（内核语义变更必须同步 bump；不符 = 崩溃）。
实验溯源：结果 JSON 记录 `status`（{"enabled": bool, "abi": str|None}）。

实现说明：解析是**惰性**的（首次访问 on/kernel/status 或显式 `configure()` 时完成）——
这样命令行参数可以在程序启动早期（参数解析后）决定开关，随后于首次使用时严格校验；
已解析后不允许冲突重配（防中途切换语义）。

附加（2026-09-14，用户批准）：`cn_counts_vec()` —— CN 共现计数的 **numpy 全批次向量化**
（过渡实现；受同一 accel 开关控制；K3 Rust 内核到货后由同一接缝替换）。
"""
import os

import numpy as np

ABI = "k1k2-v2-2026-09-11"
ENV_VAR = "SIGNDYG_ACCEL"

_resolved = False


def _env_default():
    raw = os.environ.get(ENV_VAR, "")
    if raw == "0":
        return False
    if raw in ("", "1"):
        return True
    raise RuntimeError(f"[accel] 非法 {ENV_VAR}={raw!r}（仅支持 '0'/'1'）")


def configure(enabled=None):
    """配置加速开关。enabled: True/False 显式指定；None = 回落环境变量（默认启用）。

    进程内首次生效；重复配置仅允许同值（冲突时报错，防止运行中途切换）。
    """
    global _resolved, kernel, on, status
    want = _env_default() if enabled is None else bool(enabled)
    if _resolved:
        if want != on:
            raise RuntimeError("[accel] 开关已确定后不允许冲突重配")
        return
    if want:
        import signdyg_accel  # 未安装 → ModuleNotFoundError（fail-fast：拒绝运行）
        abi = getattr(signdyg_accel, "__abi__", None)
        if abi != ABI:
            raise RuntimeError(
                f"[accel] 契约不符：__abi__={abi!r}，期望 {ABI!r}——请重建/更新 signdyg_accel"
            )
        kernel = signdyg_accel
        on = True
        print(f"[accel] enabled（signdyg_accel __abi__={abi}）")
    else:
        kernel = None
        on = False
        print("[accel] disabled（原始路径）")
    status = {"enabled": on, "abi": (ABI if on else None)}
    _resolved = True


def __getattr__(name):  # 惰性解析：on / kernel / status 在首次访问前不存在
    if name in ("on", "kernel", "status"):
        configure(None)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def cn_counts_vec(src_padded_ids: np.ndarray, dst_padded_ids: np.ndarray):
    """CN 共现计数 · numpy 全批次向量化（accel 路径；2026-09-14 落地，用户批准）。

    语义与 `NeighborInteractEncoder.count_nodes_appearances` 的逐行原路径**逐位一致**
    （自检：tools/verify/test_cn_vec_seam.py；子步骤分解：tools/verify/cn_microbench.py）：
      - 每行: src 邻居输出 [src 内计数, 在 dst 中计数]；dst 邻居输出 [在 src 中计数, dst 内计数]；
      - 0 填充位置输出 0；允许重复 id；输入为 int 型非负 (B, L) padded ids（Ls 与 Ld 可不同）；
      - 实现：行复合键 row*stride + id → 全局 unique + searchsorted 跨行匹配，消除
        原实现的 2B 次 np.unique / 2B 次 torch.apply_ 逐元素回调 / 4B 次行级 .to(device)。
      - 边界：单侧 L=0 按原语义（对侧内部计数照常、跨侧计 0）；B=0 返回空张量
        （原路径对空批次会崩，此处为安全增强）。
    返回: (src_app, dst_app)，均为 float32 ndarray，形状 (B, Ls, 2) / (B, Ld, 2)。
    """
    src = np.ascontiguousarray(src_padded_ids, dtype=np.int64)
    dst = np.ascontiguousarray(dst_padded_ids, dtype=np.int64)
    if src.ndim != 2 or dst.ndim != 2:
        raise ValueError(f"[accel] cn_counts_vec 要求 2D 输入，收到 {src.shape} / {dst.shape}")
    if src.shape[0] != dst.shape[0]:
        raise ValueError(f"[accel] cn_counts_vec 要求 src/dst 同 batch，收到 {src.shape} / {dst.shape}")
    B, Ls = src.shape
    Ld = dst.shape[1]
    stride = int(max(src.max(initial=0), dst.max(initial=0))) + 1
    rows_s = np.repeat(np.arange(B, dtype=np.int64), Ls)
    rows_d = np.repeat(np.arange(B, dtype=np.int64), Ld)
    ks = rows_s * stride + src.reshape(-1)
    kd = rows_d * stride + dst.reshape(-1)

    if ks.size:
        us, inv_s, cnt_s = np.unique(ks, return_inverse=True, return_counts=True)
        s_in_s = cnt_s[inv_s]
    else:
        us = np.empty(0, dtype=np.int64)
        cnt_s = np.empty(0, dtype=np.int64)
        s_in_s = np.empty(0, dtype=np.int64)
    if kd.size:
        ud, inv_d, cnt_d = np.unique(kd, return_inverse=True, return_counts=True)
        d_in_d = cnt_d[inv_d]
    else:
        ud = np.empty(0, dtype=np.int64)
        cnt_d = np.empty(0, dtype=np.int64)
        d_in_d = np.empty(0, dtype=np.int64)

    if ks.size and ud.size:
        pos = np.searchsorted(ud, ks)
        pos_c = np.minimum(pos, ud.size - 1)
        s_in_d = np.where(ud[pos_c] == ks, cnt_d[pos_c], 0)
    else:
        s_in_d = np.zeros(ks.size, dtype=np.int64)
    if kd.size and us.size:
        pos2 = np.searchsorted(us, kd)
        pos2_c = np.minimum(pos2, us.size - 1)
        d_in_s = np.where(us[pos2_c] == kd, cnt_s[pos2_c], 0)
    else:
        d_in_s = np.zeros(kd.size, dtype=np.int64)

    src_app = np.stack([s_in_s, s_in_d], axis=1).reshape(B, Ls, 2)
    dst_app = np.stack([d_in_s, d_in_d], axis=1).reshape(B, Ld, 2)
    src_app[src == 0] = 0
    dst_app[dst == 0] = 0
    return src_app.astype(np.float32), dst_app.astype(np.float32)
