# -*- coding: utf-8 -*-
"""新旧网格差异（符号与量级）：新代际（grid_new） vs 旧档（sign=sign_param 完整；linksign=gridbak 混合代）。

对每 ds×task×指标（主指标 + auc）：取两代共同存在的 25 格，输出
  - 每格 Δ(新−旧) 的符号分布（正/负/零）
  - |Δ| 的分位与均值（‰）
  - 最优点是否易位
输出：results/grid_diff_20260925.txt
用法：python tools/verify/grid_diff_table.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ["RedditHyperlinkTitle", "RedditHyperlinkBody", "WikiVote"]
AXIS = [(nn, lf) for nn in (15, 40, 60, 80, 100) for lf in (1, 3, 5, 10, 15)]


SUFFIX = ".RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"


def load(dirs, task):
    out = {}
    for d in dirs:
        for p in glob.glob(str(ROOT / d)):
            name = Path(p).name
            if not name.endswith(SUFFIX):
                continue
            m = re.search(r"NN-(\d+)\.LF-(\d+)", name)
            if not m:
                continue
            nn, lf = int(m.group(1)), int(m.group(2))
            if (nn, lf) not in AXIS:
                continue
            try:
                tm = json.load(open(p, encoding="utf-8"))["test metrics"]
            except Exception:
                continue
            out[(nn, lf)] = tm
    return out


L = []
ap_ = L.append
ap_("新旧网格差异（新=09-25 新代际；旧=sign_param 完整 / linksign=gridbak 混合代；共同 25 格）")
ap_("")
for task, main_k in (("linksign", "f1_wt"), ("sign", "f1_macro")):
    for ds in DS:
        new = load([f"results/grid_new/raw/{task}/{ds}/SignDyGFormer_seed42.NN-[0-9]*.LF-[0-9]*.json"], task)
        if task == "sign":
            old = load([f"results/sign_param/raw/{ds}/SignDyGFormer_seed42.NN-[0-9]*.LF-[0-9]*.json"], task)
        else:
            old = load([f"results/grid_oldgen_backup/linksign/{ds}/SignDyGFormer_seed42.NN-[0-9]*.LF-[0-9]*.json"], task)
        common = sorted(set(new) & set(old))
        if not common:
            ap_(f"===== {task}/{ds}：无共同格（new {len(new)} / old {len(old)}）")
            continue
        for k in (main_k, "auc"):
            try:
                d = np.array([(float(new[c][k]) - float(old[c][k])) * 1000 for c in common])
            except KeyError:
                ap_(f"===== {task}/{ds} [{k}]：键缺失（跳过）")
                continue
            npos, nneg, nz = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
            q = np.percentile(np.abs(d), [50, 90])
            bn, bo = max(common, key=lambda c: float(new[c][k])), max(common, key=lambda c: float(old[c][k]))
            ap_(f"===== {task}/{ds} [{k}] 共同格 {len(common)}（new {len(new)}/old {len(old)}）")
            ap_(f"  Δ(新−旧)：正 {npos} / 负 {nneg} / 零 {nz}；|Δ| p50={q[0]:.1f}‰ p90={q[1]:.1f}‰；均值 {d.mean():+.1f}‰")
            ap_(f"  最优点：新 {bn[0]}/{bn[1]}（{float(new[bn][k]):.4f}）｜旧 {bo[0]}/{bo[1]}（{float(old[bo][k]):.4f}）"
                + ("（易位）" if bn != bo else "（同点）"))
        ap_("")

out = ROOT / "results/grid_diff_20260925.txt"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"写出 {out}")
print("\n".join(L))
