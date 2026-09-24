# -*- coding: utf-8 -*-
"""k×N 网格点审计（4ded §二）：每数据集当前点 vs 图最优点（主指标=f1_macro、auc）。

口径：sign 网格（results/sign_param/raw）为**单种子 seed42**；差值只作定位、无显著性。
输出：results/grid_audit_sign_20260924.txt
用法：python tools/verify/grid_point_audit.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 当前主表 sign 配置（run_experiments TASK_DATASET_BEST_PARAMS）
CURRENT = {"WikiVote": (40, 15), "RedditHyperlinkTitle": (100, 1), "RedditHyperlinkBody": (60, 1),
           "BitcoinAlpha": (40, 15), "BitcoinOTC": (40, 15)}
DS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]

L = []
ap_ = L.append
ap_("k×N 网格点审计（sign；单种子 seed42；只作定位，差值无显著性）")
ap_("当前点 = 主表配置；最优点 = 图中 f1_macro / auc 各自最大点")
ap_("")

for ds in DS:
    pts = {}
    for p in glob.glob(str(ROOT / f"results/sign_param/raw/{ds}/*.json")):
        m = re.search(r"NN-(\d+)\.LF-(\d+)", Path(p).name)
        if not m:
            continue
        tm = json.load(open(p, encoding="utf-8"))["test metrics"]
        pts[(int(m.group(1)), int(m.group(2)))] = {
            "f1_macro": float(tm["f1_macro"]), "auc": float(tm["auc"]),
        }
    if not pts:
        ap_(f"===== {ds}: （无数据）")
        continue
    cur = CURRENT[ds]
    cur_m = pts.get(cur)
    best_f1 = max(pts.items(), key=lambda kv: kv[1]["f1_macro"])
    best_auc = max(pts.items(), key=lambda kv: kv[1]["auc"])
    rank_f1 = sorted(pts.items(), key=lambda kv: -kv[1]["f1_macro"]).index((cur, pts[cur])) + 1
    rank_auc = sorted(pts.items(), key=lambda kv: -kv[1]["auc"]).index((cur, pts[cur])) + 1
    ap_(f"===== {ds}（点数 {len(pts)}）")
    ap_(f"  当前点 NN-{cur[0]}.LF-{cur[1]}: f1_macro {cur_m['f1_macro']:.4f}（rank {rank_f1}/{len(pts)}）"
        f"  auc {cur_m['auc']:.4f}（rank {rank_auc}/{len(pts)}）")
    ap_(f"  最优 f1_macro: NN-{best_f1[0][0]}.LF-{best_f1[0][1]} {best_f1[1]['f1_macro']:.4f}"
        f"  Δ=+{best_f1[1]['f1_macro']-cur_m['f1_macro']:.4f}")
    ap_(f"  最优 auc    : NN-{best_auc[0][0]}.LF-{best_auc[0][1]} {best_auc[1]['auc']:.4f}"
        f"  Δ=+{best_auc[1]['auc']-cur_m['auc']:.4f}")
    ap_("")

out = ROOT / "results/grid_audit_sign_20260924.txt"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"写出 {out}")
print("\n".join(L))
