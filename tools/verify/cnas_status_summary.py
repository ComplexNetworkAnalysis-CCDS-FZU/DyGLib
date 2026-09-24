# -*- coding: utf-8 -*-
"""CNAS 现状小结（Paper d350 §二）：一句话结论 + 三组数。

(i) 未改进 CNAS 净效应：
    linksign 2×2：CNAS-only([F,F,F,T]) − 骨干([F,F,F,F])；LOO：full − w/o CNAS([T,T,T,F])
    sign：w/o CNAS（[T,T,T,F]）对 full
(ii) 改进版净效应：E1a(.TF-E) vs full；E1c m=80 vs full（RB）
(iii) 归因：E1a 修复 last-CN 截断后各面符号
输出：results/cnas_status_20260924.txt
用法：python tools/verify/cnas_status_summary.py
"""
from __future__ import annotations

import glob
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]


def _seed(p):
    return int(re.search(r"seed(\d+)", Path(p).name).group(1))


def load(pattern):
    out = {}
    for p in glob.glob(str(ROOT / pattern)):
        if "-profiler" in p:
            continue
        j = json.load(open(p, encoding="utf-8"))
        tm = j.get("test metrics", j.get("metrics", {}))
        out[_seed(p)] = {k: float(v) for k, v in tm.items() if v is not None}
    return out


def paired(a, b, key):
    d = [a[s][key] - b[s][key] for s in a if s in b and key in a[s] and key in b[s]]
    if not d:
        return None
    m = statistics.mean(d) * 1000
    n_plus = sum(1 for x in d if x > 0)
    t = None
    try:
        from scipy import stats

        t = stats.ttest_rel([a[s][key] for s in a if s in b and key in a[s] and key in b[s]],
                            [b[s][key] for s in a if s in b and key in a[s] and key in b[s]])
    except Exception:
        pass
    return m, n_plus, len(d), (t.pvalue if t is not None else None)


L = []
ap_ = L.append
ap_("CNAS 现状小结（2026-09-24；Δ 单位 ‰；正 = 前者更大）")
ap_("")

ap_("## (i) 未改进 CNAS 净效应（linksign；5 种子配对）")
ap_("-- linksign：CNAS-only([F,F,F,T]) − 骨干([F,F,F,F])")
hdr = f"{'ds':<22}" + "".join(f"{k:>18}" for k in ("f1_wt", "f1_mac", "auc"))
ap_(hdr)
for ds in DS:
    base = load(f"results/E-2_ablation/raw_seeds/{ds}/*RAS-D.RASE-D.BTE-D.CNAS-D.P1.TE.json")
    cnas = load(f"results/E-2_ablation/raw_seeds/{ds}/*RAS-D.RASE-D.BTE-D.CNAS-E.P1.TE.json")
    cells = []
    for k in ("f1_wt", "f1_mac", "auc"):
        r = paired(cnas, base, k)
        cells.append("—" if r is None else f"{r[0]:+.1f} ({r[1]}/{r[2]})")
    ap_(f"{ds:<22}" + "".join(f"{c:>18}" for c in cells))
ap_("")
ap_("-- linksign：full − w/o CNAS([T,T,T,F])")
ap_(hdr)
for ds in DS:
    full = load(f"results/E-2_ablation/raw_seeds/{ds}/*RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json")
    woc = load(f"results/E-2_ablation/raw_seeds/{ds}/*RAS-E.RASE-E.BTE-E.CNAS-D.P1.TE.json")
    cells = []
    for k in ("f1_wt", "f1_mac", "auc"):
        r = paired(full, woc, k)
        cells.append("—" if r is None else f"{r[0]:+.1f} ({r[1]}/{r[2]})")
    ap_(f"{ds:<22}" + "".join(f"{c:>18}" for c in cells))
ap_("")
ap_("-- sign：w/o CNAS([T,T,T,F]) − full（5 种子）")
ap_(f"{'ds':<22}" + "".join(f"{k:>18}" for k in ("auc", "f1_macro", "f1_binary")))
for ds in DS:
    full = load(f"results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.json")
    woc = load(f"results/sign_wocnas/raw/{ds}/*CNAS-D.P1.TE.json")
    cells = []
    for k in ("auc", "f1_macro", "f1_binary"):
        r = paired(woc, full, k)
        cells.append("—" if r is None else f"{r[0]:+.1f} ({r[1]}/{r[2]})")
    ap_(f"{ds:<22}" + "".join(f"{c:>18}" for c in cells))
ap_("")

ap_("## (ii) 改进版净效应（vs full）")
ap_("-- E1a(.TF-E) − full（linksign 5×5）")
ap_(f"{'ds':<22}" + "".join(f"{k:>18}" for k in ("f1_wt", "f1_mac", "auc")))
for ds in DS:
    base = load(f"results/e1a_tailfill/raw_base/linksign/{ds}/*.json")
    tf = load(f"results/e1a_tailfill/raw/linksign/{ds}/*.json")
    cells = []
    for k in ("f1_wt", "f1_mac", "auc"):
        r = paired(tf, base, k)
        cells.append("—" if r is None else f"{r[0]:+.1f} ({r[1]}/{r[2]})")
    ap_(f"{ds:<22}" + "".join(f"{c:>18}" for c in cells))
ap_("-- E1c m=80(.RK-80) − full（RB；5 种子）")
row = []
base = load("results/e1a_tailfill/raw_base/linksign/RedditHyperlinkBody/*.json")
rb80 = load("results/e1c_rb80/raw/*.json")
for k in ("f1_wt", "f1_mac", "auc"):
    r = paired(rb80, base, k)
    row.append("—" if r is None else f"{k}: {r[0]:+.1f} ({r[1]}/{r[2]})")
ap_("   " + " | ".join(row))
ap_("")

ap_("## (iii) 归因一句话素材（E1a≈取消 last-CN 截断）")
ap_("- E1a 修好后：auc 面 —— 逐 ds 符号见 (ii)（均值 & 5/5 计数）；f1_wt 面见 (ii)。")
ap_("- 未改进 CNAS vs 骨干/LOO 的负收益格：见 (i)。")

out = ROOT / "results/cnas_status_20260924.txt"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"写出 {out}")
print("\n".join(L))
