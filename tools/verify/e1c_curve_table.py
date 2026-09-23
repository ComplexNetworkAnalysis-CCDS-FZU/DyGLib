# -*- coding: utf-8 -*-
"""E1c 双块窗口（`.RK-{m}`）m-sweep 曲线表（linksign seed42；对照 m=0 = full）。

读 `results/e1c_recent_block/raw/{ds}/*.json`（RK-* 文件），对照
`results/e1a_tailfill/raw_base/linksign/{ds}/*seed42*.json`（m=0 基线），
输出每数据集按 m 升序的 auc / f1_wt / f1_mac / ap 及其 Δ（vs m=0）。

用法（仓库根）：python tools/verify/e1c_curve_table.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"]
METRICS = ["auc", "f1_wt", "f1_mac", "ap"]


def load_metrics(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["test metrics"]


def baseline(ds: str):
    hits = [
        p
        for p in glob.glob(str(ROOT / f"results/e1a_tailfill/raw_base/linksign/{ds}/*.json"))
        if "seed42" in Path(p).name and "-profiler" not in Path(p).name
    ]
    return load_metrics(hits[0]) if hits else None


def main() -> int:
    for ds in DATASETS:
        base = baseline(ds)
        rows = []
        for p in glob.glob(str(ROOT / f"results/e1c_recent_block/raw/{ds}/*.json")):
            name = Path(p).name
            if "-profiler" in name:
                continue
            m = re.search(r"RK-(\d+)\.json$", name)
            if not m:
                continue
            rows.append((int(m.group(1)), load_metrics(p)))
        rows.sort(key=lambda x: x[0])
        print(f"\n=== {ds}（linksign seed42；m=0 = full 基线）===")
        header = f"{'m':>5}" + "".join(f"{mt:>18}" for mt in METRICS)
        print(header)
        if base is None:
            print("（缺 m=0 基线文件）")
        else:
            line = f"{0:>5}" + "".join(f"{float(base[mt]):>18.4f}" for mt in METRICS)
            print(line)
        for m, met in rows:
            line = f"{m:>5}"
            for mt in METRICS:
                v = float(met[mt])
                if base is not None:
                    d = v - float(base[mt])
                    line += f"{v:>10.4f}({d:+.4f})"
                else:
                    line += f"{v:>18.4f}"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
