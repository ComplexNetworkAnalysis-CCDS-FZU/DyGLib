# -*- coding: utf-8 -*-
"""对比两次训练 run 的结果 JSON（accel on/off 双短跑），验证"除计时/显存外逐位一致"。

用法（仓库根）：python tools/verify/compare_accel_runs.py <off.json> <on.json>
判据：剔除易变字段（single run time / training time / inference time / peak memory / accel）
      后逐键相等（指标以 4 位小数字符串记录，即"记录位"逐位一致）。
退出码：0=一致；1=不一致。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

VOLATILE = {
    "single run time (s)",
    "training time (s)",
    "inference time (s)",
    "peak memory (MB)",
    "accel",
}


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in VOLATILE}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    da = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    db = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    print(f"off accel = {da.get('accel')}")
    print(f"on  accel = {db.get('accel')}")
    sa, sb = strip(da), strip(db)
    if sa == sb:
        print("PASS：除计时/显存外逐位一致（on/off 指标完全相同）")
        return 0
    print("FAIL：存在差异")
    for k in sorted(set(sa) | set(sb)):
        if sa.get(k) != sb.get(k):
            print(f"  [{k}]\n    off = {sa.get(k)}\n    on  = {sb.get(k)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
