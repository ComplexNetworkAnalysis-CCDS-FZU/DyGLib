"""tools/verify/snapshot_results.py — 结果快照：从本地已同步归档快速汇总关键指标。

用法（仓库根目录）:
    python tools/verify/snapshot_results.py

仅读取本地 results/**/raw 下的 JSON（不访问服务器），输出各组的
auc / ap / sign_f1 明细与 5 种子均值±pstd，供 PROGRESS/汇报引用。
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COLS = ("auc", "ap", "sign_f1", "f1_wt", "f1_mac", "f1_mic")


def fnum(m, k):
    try:
        return float(m.get(k))
    except (TypeError, ValueError):
        return float("nan")


def show(title, pattern):
    paths = sorted(glob.glob(str(ROOT / pattern)))
    print(f"== {title} ==")
    if not paths:
        print("  (无文件)")
        return []
    got = []
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        m = d.get("test metrics", {})
        got.append((Path(p).name, m))
        vals = "  ".join(f"{c}={fnum(m, c):.4f}" for c in COLS)
        print(f"  {Path(p).name}")
        print(f"    {vals}")
    return got


def mean_line(tag, got, keys=("auc", "ap", "sign_f1")):
    if not got:
        return
    for k in keys:
        vs = [fnum(m, k) for _, m in got]
        vs = [v for v in vs if v == v]
        if vs:
            print(f"  > {tag} {k}: mean={st.mean(vs):.4f} pstd={st.pstdev(vs):.4f} (n={len(vs)})")


def main() -> int:
    print("# 结果快照（本地归档，CN 真交集修复版）\n")

    show("E-3 Patch RB（NN-80 LF-3, seed42）", "results/E-3_patch/raw/*NN-80.LF-3.RAS-E.RASE-E.BTE-E.CNAS-E.P[1357].TE.json")
    print()
    show("E-3 Patch WV（NN-15 LF-10, seed42）", "results/E-3_patch/raw/*NN-15.LF-10.RAS-E.RASE-E.BTE-E.CNAS-E.P[1357].TE.json")
    print()
    print("== E-4 时序：TD(λ=1.0) vs TE（WV NN-15 LF-10, seed42，同配置）==")
    td = show("  TD", "results/E-4_time_decay/raw/*NN-15.LF-10.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TD.json")
    te = show("  TE（E-3 P1 对照）", "results/E-3_patch/raw/*NN-15.LF-10.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json")
    if td and te:
        for k in ("auc", "ap", "sign_f1"):
            a, b = fnum(td[0][1], k), fnum(te[0][1], k)
            if a == a and b == b:
                print(f"  > Δ{k} (TD-TE) = {a - b:+.4f}")
    print()
    g = show("主表 sign RT（NN-100 LF-1, 5 seeds）", "results/main_tables/raw/sign/RedditHyperlinkTitle/*.json")
    mean_line("sign RT", g)
    print()
    g = show("主表 sign RB（NN-60 LF-1, 5 seeds）", "results/main_tables/raw/sign/RedditHyperlinkBody/*.json")
    mean_line("sign RB", g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
