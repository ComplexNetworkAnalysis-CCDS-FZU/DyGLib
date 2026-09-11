"""部署校验：确认 CN 伪交集修复已生效（真交集语义）。

背景：common_neighbor_location 原实现用 np.intersect1d(..., assume_unique=True)，
在历史含重复 id（重复边）时会退化为伪交集（单侧重复也计入），与论文 §3.2 的
C = N_u ∩ N_v 定义不符。2026-09-11 用户决策修复为真交集（去掉 assume_unique）。

用法（仓库根目录）:
    python tools/verify/check_cn_fix.py

期望：
    修复版输出 keys= [2]        → CN_TRUE_INTERSECTION（退出码 0）
    修复前输出 keys= [1, 2]     → CN_FIX_MISSING（退出码 1）

判定样例：u 的邻居 [1,1,2]（1 重复、2 出现一次），v 的邻居 [2]。
真交集 = {2}；伪交集（旧实现）= {1, 2}（1 因排序后相邻相等被误判）。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.direct_neighbor_sampler import common_neighbor_location  # noqa: E402


def main() -> int:
    cn = common_neighbor_location(np.array([1, 1, 2]), np.array([2]))
    keys = sorted(cn.keys())
    print("keys=", keys)
    ok = keys == [2]
    print("CN_TRUE_INTERSECTION" if ok else "CN_FIX_MISSING (still pseudo-intersection?)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
