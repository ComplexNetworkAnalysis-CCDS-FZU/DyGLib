# -*- coding: utf-8 -*-
"""新网格（方案 B）审计：当前点 vs 最优点（每图 f1 主指标 + auc 两列）。

数据：results/grid_new/raw/{task}/{ds}/（seed42；25 格轴 NN{15,40,60,80,100}×LF{1,3,5,10,15}；
      含旧轴额外点则自动忽略）。单种子 ⇒ 差值仅定位、无显著性。
输出：results/grid_new_audit_20260925.txt
用法：python tools/verify/grid_new_audit.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ["RedditHyperlinkTitle", "RedditHyperlinkBody", "WikiVote"]
AXIS_NN = [15, 40, 60, 80, 100]
AXIS_LF = [1, 3, 5, 10, 15]
CUR = {
    "linksign": {"RedditHyperlinkTitle": (60, 1), "RedditHyperlinkBody": (80, 3), "WikiVote": (15, 10)},
    "sign": {"RedditHyperlinkTitle": (100, 1), "RedditHyperlinkBody": (60, 1), "WikiVote": (40, 15)},
}
MAIN = {"linksign": ["f1_wt", "f1_mac", "sign_f1", "ap", "auc"],
        "sign": ["f1_macro", "f1_binary", "ap", "auc"]}


def load_sigma(task, ds):
    """单点噪声 = 现有 5 种子 std（linksign: raw_base；sign: sign_valthr）。"""
    import statistics

    if task == "linksign":
        pat = f"results/e1a_tailfill/raw_base/linksign/{ds}/*.json"
    else:
        pat = f"results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.json"
    vals = {"f1_wt": [], "f1_macro": [], "auc": []}
    for p in glob.glob(str(ROOT / pat)):
        if "-profiler" in p:
            continue
        tm = json.load(open(p, encoding="utf-8"))["test metrics"]
        for k in vals:
            if tm.get(k) is not None:
                vals[k].append(float(tm[k]))
    return {k: (statistics.stdev(v) if len(v) > 1 else float("nan")) for k, v in vals.items()}

L = []
ap_ = L.append
ap_("新网格（方案 B·新代际）审计：当前点 vs 最优点（单种子 seed42；差值为定位性、无显著性）")
ap_("")
for task in ("linksign", "sign"):
    for ds in DS:
        pts = {}
        for p in glob.glob(str(ROOT / f"results/grid_new/raw/{task}/{ds}/SignDyGFormer_seed42.NN-*.LF-*.json")):
            name = Path(p).name
            if not name.endswith(".RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"):
                continue
            m = re.search(r"NN-(\d+)\.LF-(\d+)", name)
            if not m:
                continue
            nn, lf = int(m.group(1)), int(m.group(2))
            if nn not in AXIS_NN or lf not in AXIS_LF:
                continue
            tm = json.load(open(p, encoding="utf-8"))["test metrics"]
            pts[(nn, lf)] = tm
        if not pts:
            ap_(f"===== {task}/{ds}: （暂无数据）")
            continue
        ap_(f"===== {task}/{ds}（格数 {len(pts)}/25）")
        main_k = "f1_wt" if task == "linksign" else "f1_macro"
        sigma = load_sigma(task, ds)
        for k in MAIN[task]:
            best = max(pts.items(), key=lambda kv: float(kv[1][k]))
            cur = CUR[task][ds]
            sig = sigma.get(k, float("nan"))
            if cur in pts:
                cv = float(pts[cur][k])
                rank = sorted(pts.items(), key=lambda kv: -float(kv[1][k])).index((cur, pts[cur])) + 1
                dd = float(best[1][k]) - cv
                ratio = dd / sig if isinstance(sig, float) and sig == sig and sig > 0 else float("nan")
                flag = " ★候选(>2σ)" if ratio == ratio and ratio > 2 else ""
                ap_(f"  [{k}] 当前 {cur[0]}/{cur[1]} = {cv:.4f}（rank {rank}/{len(pts)}）｜"
                    f"最优 {best[0][0]}/{best[0][1]} = {float(best[1][k]):.4f}"
                    f"（Δ={dd*1000:+.1f}‰；Δ/σ={ratio:.2f}）{flag}")
            else:
                ap_(f"  [{k}] 当前 {cur[0]}/{cur[1]} 缺格｜最优 {best[0][0]}/{best[0][1]} = {float(best[1][k]):.4f}")
        ap_("")
out = ROOT / "results/grid_new_audit_20260925.txt"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"写出 {out}")
print("\n".join(L))
