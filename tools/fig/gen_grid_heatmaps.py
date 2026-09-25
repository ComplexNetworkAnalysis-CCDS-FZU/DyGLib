#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""网格方案 B 热力图（new generation, 2026-09-25）：RT/RB/WV × {sign, link&sign}。

Reads:  results/grid_new/raw/{task}/{ds}/SignDyGFormer_seed42.NN-*.LF-*.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json
Writes: figures/fig_grid_{task}_{ds}.png/.pdf （双面板：主指标 | AUC；6 图）
        figures/fig_grid_heatmap.csv （长表：task,dataset,NN,LF,{main},auc）

Style mirrors analysis/param-graph.py（sns.heatmap, cmap="crest", annot .4f, fontsize 14）.
Run from repo root:  python tools/fig/gen_grid_heatmaps.py
"""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DS_LIST = ["RedditHyperlinkTitle", "RedditHyperlinkBody", "WikiVote"]
TASKS = {"linksign": "f1_wt", "sign": "f1_macro"}
AXIS_NN = [15, 40, 60, 80, 100]
AXIS_LF = [1, 3, 5, 10, 15]
DISP = {"RedditHyperlinkTitle": "RedditTitle", "RedditHyperlinkBody": "RedditBody", "WikiVote": "WikiRfA"}
CUR = {"linksign": {"RedditHyperlinkTitle": (60, 1), "RedditHyperlinkBody": (80, 3), "WikiVote": (15, 10)},
       "sign": {"RedditHyperlinkTitle": (100, 1), "RedditHyperlinkBody": (60, 1), "WikiVote": (40, 15)}}


def load_matrix(task, ds, metric):
    m = np.full((len(AXIS_NN), len(AXIS_LF)), np.nan)
    for p in glob.glob(str(ROOT / f"results/grid_new/raw/{task}/{ds}/SignDyGFormer_seed42.NN-*.LF-*.json")):
        name = Path(p).name
        if not name.endswith(".RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"):
            continue
        mm = re.search(r"NN-(\d+)\.LF-(\d+)", name)
        if not mm:
            continue
        nn, lf = int(mm.group(1)), int(mm.group(2))
        if nn not in AXIS_NN or lf not in AXIS_LF:
            continue
        tm = json.load(open(p, encoding="utf-8"))["test metrics"]
        m[AXIS_NN.index(nn), AXIS_LF.index(lf)] = float(tm[metric])
    return m


rows = []
for task, main_k in TASKS.items():
    for ds in DS_LIST:
        main_m = load_matrix(task, ds, main_k)
        auc_m = load_matrix(task, ds, "auc")
        if np.isnan(main_m).all():
            print(f"[skip] {task}/{ds}: 无数据")
            continue
        for i, nn in enumerate(AXIS_NN):
            for j, lf in enumerate(AXIS_LF):
                if not np.isnan(main_m[i, j]):
                    rows.append((task, ds, nn, lf, main_m[i, j], auc_m[i, j]))
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
        cur = CUR[task][ds]
        for ax, mat, title in ((axes[0], main_m, main_k), (axes[1], auc_m, "auc")):
            sns.heatmap(
                mat, ax=ax, cmap="crest", annot=True, fmt=".4f", cbar=True,
                xticklabels=[str(v) for v in AXIS_LF], yticklabels=[str(v) for v in AXIS_NN],
                vmin=np.nanmin(mat),
            )
            ax.set_title(f"{DISP[ds]} · {task} · {title}", fontsize=14)
            ax.set_xlabel("LF (common-neighbors-look-forward)", fontsize=14)
            ax.set_ylabel("NN (num-neighbors)", fontsize=14)
            # 标注当前配置点
            if cur[0] in AXIS_NN and cur[1] in AXIS_LF:
                ax.add_patch(plt.Rectangle((AXIS_LF.index(cur[1]), AXIS_NN.index(cur[0])), 1, 1,
                                           fill=False, edgecolor="red", lw=2.5))
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"fig_grid_{task}_{ds}.{ext}", dpi=300)
        plt.close(fig)
        print(f"[fig] fig_grid_{task}_{ds}.png/pdf")

with open(FIG_DIR / "fig_grid_heatmap.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["task", "dataset", "NN", "LF", "main", "auc"])
    for r in rows:
        w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.6f}", "" if np.isnan(r[5]) else f"{r[5]:.6f}"])
print(f"[csv] {FIG_DIR / 'fig_grid_heatmap.csv'}（{len(rows)} 格）")
