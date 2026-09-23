# -*- coding: utf-8 -*-
"""阈值漂移对照表（Paper 2026-09-23 `3ee0` §二①）：E1a（`.TF-E`）vs full 的 thr 分布。

linksign：thr_sign / thr_exist（逐数据集 5 种子 mean±pstd、Δ）；sign：thr（seed42 单点）。
用法（仓库根）：python tools/verify/thr_drift_table.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]


def load_side(sub: str, task: str, ds: str) -> dict:
    out = {}
    for p in glob.glob(str(ROOT / f"results/e1a_tailfill/{sub}/{task}/{ds}/*.json")):
        name = Path(p).name
        if "-profiler" in name:
            continue
        m = re.search(r"seed(\d+)", name)
        seed = int(m.group(1)) if m else -1
        with open(p, encoding="utf-8") as f:
            out[seed] = json.load(f)["test metrics"]
    return out


def col(rows: dict, key: str):
    vals = [float(v[key]) for v in rows.values() if key in v]
    return np.array(vals) if vals else None


def main() -> int:
    print("=" * 96)
    print("阈值漂移对照 —— linksign：thr_sign / thr_exist（E1a=.TF-E vs full；5 种子）")
    print("=" * 96)
    for key in ["thr_sign", "thr_exist"]:
        print(f"\n--- {key} ---")
        print(f"{'数据集':<22}{'full(mean±pstd)':<20}{'E1a(mean±pstd)':<20}{'Δmean':>9}")
        for ds in DATASETS:
            b = col(load_side("raw_base", "linksign", ds), key)
            e = col(load_side("raw", "linksign", ds), key)
            if b is None or e is None or len(b) == 0 or len(e) == 0:
                print(f"{ds:<22}{'（缺）':<20}")
                continue
            print(f"{ds:<22}{b.mean():.4f}±{b.std(ddof=0):.4f}      "
                  f"{e.mean():.4f}±{e.std(ddof=0):.4f}      {e.mean() - b.mean():>+9.4f}")

    print("\n" + "=" * 96)
    print("阈值漂移对照 —— sign：thr（E1a vs full；seed42 单点）")
    print("=" * 96)
    print(f"{'数据集':<22}{'full':>10}{'E1a':>10}{'Δ':>10}")
    for ds in DATASETS:
        b = load_side("raw_base", "sign", ds)
        e = load_side("raw", "sign", ds)
        if not b or not e:
            print(f"{ds:<22}{'（缺）':>10}")
            continue
        s = sorted(set(b) & set(e))[0]
        bt, et = float(b[s].get("thr", float("nan"))), float(e[s].get("thr", float("nan")))
        print(f"{ds:<22}{bt:>10.4f}{et:>10.4f}{et - bt:>+10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
