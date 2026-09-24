# -*- coding: utf-8 -*-
"""G1-EVT 修正批判定（去 TF；2026-09-24 深夜）。

对比：test = G1-EVT 修正批（固定阈值） vs base = full（raw_base / main_tables 同代）.
每 ds：Δ 均值(‰)、同向计数(正/5)、配对 p；含 sign 侧单点对照。
输出：results/g1_fixevt_verdict_20260924.txt
用法：python tools/verify/g1_fix_verdict.py（需先 fetch --set g1fix）
"""
from __future__ import annotations

import glob
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]
LINK = ["f1_wt", "f1_mac", "sign_f1", "ap", "auc"]
SIGN = ["auc", "f1_macro", "f1_binary"]


def _seed(p):
    return int(re.search(r"seed(\d+)", Path(p).name).group(1))


def load(pattern):
    out = {}
    for p in glob.glob(str(ROOT / pattern)):
        if "-profiler" in p:
            continue
        tm = json.load(open(p, encoding="utf-8"))["test metrics"]
        out[_seed(p)] = {k: float(v) for k, v in tm.items() if v is not None}
    return out


def stat(a, b, k):
    pairs = [(a[s][k], b[s][k]) for s in a if s in b and k in a[s] and k in b[s]]
    if not pairs:
        return None
    d = [x - y for x, y in pairs]
    m = statistics.mean(d) * 1000
    npos = sum(1 for x in d if x > 0)
    p = None
    try:
        from scipy import stats

        p = stats.ttest_rel([x for x, _ in pairs], [y for _, y in pairs]).pvalue
    except Exception:
        pass
    return m, npos, len(d), p


L = []
ap_ = L.append
ap_("G1-EVT 修正批判定（去 TF；test=G1-EVT 固定阈值 vs base=full 自适应）")
ap_("Δ 单位 ‰；正 = G1 更高；(正/5)；p = 配对 t")
ap_("")

ap_("== linksign（5×5） ==")
ap_(f"{'ds':<22}" + "".join(f"{k:>20}" for k in LINK))
for ds in DS:
    base = load(f"results/e1a_tailfill/raw_base/linksign/{ds}/*.json")
    test = load(f"results/g1_fixevt/raw/linksign/{ds}/*.json")
    cells = []
    for k in LINK:
        r = stat(test, base, k)
        cells.append("—" if r is None else f"{r[0]:+.1f} ({r[1]}/{r[2]})" + (f" p={r[3]:.3f}" if r[3] is not None else ""))
    ap_(f"{ds:<22}" + "".join(f"{c:>20}" for c in cells))
ap_("")

ap_("== sign（seed42 单点对照；test=G1-EVT vs base=full） ==")
ap_(f"{'ds':<22}" + "".join(f"{k:>14}" for k in SIGN))
for ds in DS:
    base = load(f"results/e1a_tailfill/raw_base/sign/{ds}/*.json")
    test = load(f"results/g1_fixevt/raw/sign/{ds}/*.json")
    cells = []
    for k in SIGN:
        r = stat(test, base, k)
        cells.append("—" if r is None else f"{r[0]:+.1f}")
    ap_(f"{ds:<22}" + "".join(f"{c:>14}" for c in cells))

out = ROOT / "results/g1_fixevt_verdict_20260924.txt"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"写出 {out}")
print("\n".join(L))
