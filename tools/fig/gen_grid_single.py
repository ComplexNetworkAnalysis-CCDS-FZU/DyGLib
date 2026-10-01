"""gen_grid_single.py — 网格热力图「单面板紧凑件」重渲染（Paper 8b2f 行动项 · 2026-10-02）。

要求（Paper 2026-10-01 `8b2f`，用户指示）：
  1. **去掉红框**（当前点标记）与相关图例/说明；
  2. **单面板独立文件**：每个 (task, ds, metric) 一个文件，命名 `fig_grid_{task}_{ds}_{metric}.{pdf,png}`；
  3. **按目标尺寸设计字号**：目标最终宽 ≈0.32\\textwidth（≈154 pt）⇒ 单面板页面宽 ≈380–420 pt
     （本脚本 figsize 宽 = 400/72 in ≈ 5.56 in），单元格字号 ≈7.5 pt ⇒ 缩放后仍 ≥6.5 pt；
  4. **去掉面板内重复标题/图例**（保留 NN/LF 轴名；标题只写简短 metric 名，dataset/task 由 LaTeX subcaption 承担）。

Reads : results/grid_new/raw/{task}/{ds}/SignDyGFormer_seed42.NN-*.LF-*.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json
Writes: figures/single/fig_grid_{task}_{ds}_{metric}.{png,pdf}
        results/grid_figures_single_20261002/  （交付副本）
Run from repo root: python tools/fig/gen_grid_single.py
"""
from __future__ import annotations

import glob
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures" / "single"
DELIV = ROOT / "results" / "grid_figures_single_20261002"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DELIV.mkdir(parents=True, exist_ok=True)

DS_LIST = ["RedditHyperlinkTitle", "RedditHyperlinkBody", "WikiVote"]
TASKS = {"linksign": "f1_wt", "sign": "f1_macro"}
AXIS_NN = [15, 40, 60, 80, 100]
AXIS_LF = [1, 3, 5, 10, 15]
TAG = {("linksign", "f1_wt"): "f1wt", ("sign", "f1_macro"): "f1mac", ("linksign", "auc"): "auc",
       ("sign", "auc"): "auc"}
TITLE = {"f1_wt": "f1$_{wt}$", "f1_macro": "f1$_{macro}$", "auc": "AUC"}

# 两档尺寸（Paper 8b2f §一.3 的目标是"最终落在页面里的字号 ≥6–7 pt"，其给出的
# "页面宽 380–420 pt + 单元格字号 7–8 pt" 与"缩到 0.32\textwidth(≈154 pt)"在算术上互斥
# —— 若按 400 pt 设计再缩到 154 pt，7.5 pt 会变成 ~2.9 pt。故同时交付两档：
#   * 默认件（无后缀）：**按最终尺寸设计** —— 面板宽 ≈165 pt、字号 7 pt ⇒ 1:1 置入 0.32\textwidth 即可读；
#   * `_natural` 件：面板宽 400 pt、字号 7.5 pt ⇒ 供"按自然尺寸(≈0.8\textwidth)置入"时使用。
SIZES = [
    ("", 168.0, 152.0, {"annot": 6.8, "tick": 5.5, "label": 6.0, "title": 6.8, "cbar": 5.5, "bar": False}),
    ("_natural", 400.0, 330.0, {"annot": 7.5, "tick": 7.0, "label": 7.5, "title": 8.0, "cbar": 6.5, "bar": True}),
]


def load_matrix(task: str, ds: str, metric: str) -> np.ndarray:
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
        tm = json.loads(Path(p).read_text(encoding="utf-8"))["test metrics"]
        m[AXIS_NN.index(nn), AXIS_LF.index(lf)] = float(tm[metric])
    return m


def render(task: str, ds: str, metric: str, mat: np.ndarray) -> list[Path]:
    outs: list[Path] = []
    for suffix, w_pt, h_pt, fs in SIZES:
        plt.rcParams.update({
            "font.size": fs["annot"], "axes.labelsize": fs["label"], "axes.titlesize": fs["title"],
            "xtick.labelsize": fs["tick"], "ytick.labelsize": fs["tick"],
            "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        })
        fig, ax = plt.subplots(figsize=(w_pt / 72.0, h_pt / 72.0))
        sns.heatmap(
            mat, ax=ax, cmap="crest", annot=True, fmt=".4f",
            annot_kws={"fontsize": fs["annot"], "ha": "center", "va": "center"},
            xticklabels=[str(v) for v in AXIS_LF], yticklabels=[str(v) for v in AXIS_NN],
            vmin=float(np.nanmin(mat)), vmax=float(np.nanmax(mat)),
            cbar=bool(fs["bar"]),
            cbar_kws={"shrink": 0.85, "pad": 0.02, "aspect": 22},
            linewidths=0.3, linecolor="white",
        )
        if fs["bar"]:
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=fs["cbar"], length=1.5, pad=1.0)
        ax.set_title(TITLE[metric], fontsize=fs["title"], pad=3)
        ax.set_xlabel("LF", fontsize=fs["label"], labelpad=2)
        ax.set_ylabel("NN", fontsize=fs["label"], labelpad=2)
        ax.tick_params(length=1.5, pad=1.0)
        for ext in ("png", "pdf"):
            f = FIG_DIR / f"fig_grid_{task}_{ds}_{TAG[(task, metric)]}{suffix}.{ext}"
            fig.savefig(f)
            outs.append(f)
        plt.close(fig)
    return outs


n = 0
for task, main_k in TASKS.items():
    for ds in DS_LIST:
        for metric in (main_k, "auc"):
            mat = load_matrix(task, ds, metric)
            if np.isnan(mat).all():
                print(f"[skip] {task}/{ds}/{metric}: 无数据")
                continue
            files = render(task, ds, metric, mat)
            for f in files:
                shutil.copy2(f, DELIV / f.name)
            print(f"[fig] {TAG[(task, metric)]:<6} {ds:<22} 格数={int((~np.isnan(mat)).sum())}/25  "
                  f"-> {len(files)} 文件（含 _natural）")
            n += 1
print(f"[done] 单面板 {n} 组 × 2 尺寸 = {n * 2} 图 × 2 格式；目标 {len(DS_LIST) * 2 * 2} 组 = 3 ds × 2 task × 2 metric")
