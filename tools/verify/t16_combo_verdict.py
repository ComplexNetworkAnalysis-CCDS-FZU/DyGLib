# -*- coding: utf-8 -*-
"""T16 组合判定（RB：E1c m=80 × G1；训练 5 种子 + 固定阈值伴行 seed42）。

test = results/t16_combo/raw（RK-80.G1）vs base = results/e1a_tailfill/raw_base/linksign/RedditHyperlinkBody
输出：results/t16_combo_verdict_20260925.txt
"""
from __future__ import annotations

import glob
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINK = ["f1_wt", "f1_mac", "sign_f1", "ap", "auc"]


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
ap_("T16 组合判定（RB：E1c m=80 × G1；2026-09-25）")
ap_("Δ 单位 ‰；正 = 组合更高；(正/5)；p = 配对 t")
combo = load("results/t16_combo/raw/*.json")
base = load("results/e1a_tailfill/raw_base/linksign/RedditHyperlinkBody/*.json")
ap_(f"{'metric':<10}{'combo':>10}{'full':>10}{'Δ‰':>10}{'(正/5)':>9}{'p':>10}")
for k in LINK:
    r = stat(combo, base, k)
    mc = statistics.mean([combo[s][k] for s in combo if k in combo[s]])
    mb = statistics.mean([base[s][k] for s in base if k in base[s]])
    ap_(f"{k:<10}{mc:>10.4f}{mb:>10.4f}{r[0]:>10.1f}{f'{r[1]}/{r[2]}':>9}" + (f"{r[3]:>10.4f}" if r[3] is not None else f"{'—':>10}"))
evt = load("results/t16_combo/evt/*.json")
ap_("")
ap_("固定阈值伴行（seed42 单点；combo-EVT vs full seed42）：")
for k in LINK:
    if 42 in evt and 42 in base:
        ap_(f"  {k:<10}{(evt[42][k]-base[42][k])*1000:>+8.1f}‰   (evt={evt[42][k]:.4f}, full={base[42][k]:.4f})")

out = ROOT / "results/t16_combo_verdict_20260925.txt"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"写出 {out}")
print("\n".join(L))
