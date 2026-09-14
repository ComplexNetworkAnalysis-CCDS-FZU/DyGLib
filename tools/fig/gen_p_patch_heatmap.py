#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R2-11: patch-size (P) sensitivity heatmap for the SignDyG revision (NEUCOM-D-26-13975).

Reads (read-only) the E-3 v3 raw JSONs -- "CN true-intersection" fix, link & sign task,
SignDyGFormer, seed 42, TE, best NN/LF per dataset:

    results/E-3_patch/raw/

and writes:

    figures/fig_p_patch_heatmap.png   -> 2-panel heatmap (AUC + weighted F1), 300 dpi
    figures/fig_p_patch_heatmap.pdf   -> same figure (vector PDF)
    figures/fig_p_patch_heatmap.csv   -> long-format matrix: dataset,P,auc,f1_weighted

Run from the repository root:

    python tools/fig/gen_p_patch_heatmap.py

Notes
-----
* Metric values inside the raw JSONs are strings (e.g. "0.9149") and are cast with float().
* The JSON files carry no dataset field, so each dataset is mapped to the NN/LF tuple that
  results/E-3_patch/E3_patch_summary.md lists as that dataset's configuration; the mapping
  is verified at runtime against the summary tables (cross-check block printed to stdout).
  If anything disagrees, the raw JSON is the source of truth.
* Weighted F1 is stored under the key "f1_wt" in the raw JSONs (also accepting "f1_weighted").
* Visual style deliberately mirrors the repository's existing hyper-parameter heatmaps
  (analysis/param-graph.py): seaborn ``sns.heatmap`` with ``cmap="crest"``, ``annot`` in
  ``.4f`` format, the same dataset display names (WikiRfA / RedditTitle / RedditBody),
  ``fontsize=14`` axis labels, colour scale anchored at the matrix minimum.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]  # tools/fig/gen_p_patch_heatmap.py -> repo root
RAW_DIR = ROOT / "results" / "E-3_patch" / "raw"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "tools" / "fig").mkdir(parents=True, exist_ok=True)

CSV_OUT = FIG_DIR / "fig_p_patch_heatmap.csv"
PNG_OUT = FIG_DIR / "fig_p_patch_heatmap.png"
PDF_OUT = FIG_DIR / "fig_p_patch_heatmap.pdf"

P_LIST = [1, 3, 5, 7]

# Row order fixed as requested: BA, OTC, RT, RB, WV.
# (display name, NN/LF config of that dataset's best setup in E-3)
DATASETS = [
    ("BitcoinAlpha", "NN-40.LF-15"),
    ("BitcoinOTC", "NN-80.LF-5"),
    ("RedditHyperlinkTitle", "NN-60.LF-1"),
    ("RedditHyperlinkBody", "NN-80.LF-3"),
    ("WikiVote", "NN-15.LF-10"),
]

# Display names used by the paper figures (same map as analysis/param-graph.py).
DATASET_DISPLAY = {
    "BitcoinAlpha": "BitcoinAlpha",
    "BitcoinOTC": "BitcoinOTC",
    "WikiVote": "WikiRfA",
    "RedditHyperlinkTitle": "RedditTitle",
    "RedditHyperlinkBody": "RedditBody",
}

FNAME_TMPL = "SignDyGFormer_seed42.{cfg}.RAS-E.RASE-E.BTE-E.CNAS-E.P{p}.TE.json"

# Values transcribed from results/E-3_patch/E3_patch_summary.md (reading order: the
# full-dataset AUC table + the per-dataset detail tables). Used only for the runtime
# cross-check; raw JSON remains authoritative. P columns are [1, 3, 5, 7].
MD_REFERENCE = {
    "BitcoinAlpha": {
        "auc": [0.9586, 0.9549, 0.9564, 0.9526],
        "ap": [0.8195, 0.8071, 0.8136, 0.8061],
        "sign_f1": [0.9265, 0.9138, 0.9205, 0.9176],
    },
    "BitcoinOTC": {
        "auc": [0.9668, 0.9728, 0.9728, 0.9704],
        "ap": [0.8460, 0.8660, 0.8619, 0.8591],
        "sign_f1": [0.9376, 0.9453, 0.9467, 0.9465],
    },
    "RedditHyperlinkTitle": {
        "auc": [0.9386, 0.9355, 0.9347, 0.9350],
        "ap": [0.7185, 0.7133, 0.7125, 0.7119],
        "sign_f1": [0.8804, 0.8757, 0.8752, 0.8550],
    },
    "RedditHyperlinkBody": {
        "auc": [0.9239, 0.9149, 0.9222, 0.9284],
        "ap": [0.6899, 0.6802, 0.6861, 0.6947],
        "f1_wt": [0.9378, 0.9300, 0.9222, 0.9337],
        "f1_mac": [0.6753, 0.6636, 0.6786, 0.6828],
        "f1_mic": [0.9375, 0.9245, 0.9072, 0.9283],
        "sign_f1": [0.9328, 0.9178, 0.8957, 0.9219],
    },
    "WikiVote": {
        "auc": [0.9607, 0.9611, 0.9619, 0.9594],
        "ap": [0.8150, 0.8150, 0.8183, 0.8096],
        "f1_wt": [0.9032, 0.9058, 0.9058, 0.9049],
        "f1_mac": [0.7784, 0.7955, 0.7915, 0.7906],
        "f1_mic": [0.9056, 0.9046, 0.9063, 0.9033],
        "sign_f1": [0.8732, 0.8682, 0.8732, 0.8670],
    },
}

TOL = 5e-5  # half of the last reported decimal (4 decimals) -> anything beyond = mismatch


# ----------------------------------------------------------------------------- data
def load_all() -> tuple[dict[str, dict[int, dict[str, float]]], list[str]]:
    """Load all test metrics; returns ({dataset: {P: {metric: float}}}, missing_files)."""
    data: dict[str, dict[int, dict[str, float]]] = {}
    missing: list[str] = []
    for ds, cfg in DATASETS:
        data[ds] = {}
        for p in P_LIST:
            path = RAW_DIR / FNAME_TMPL.format(cfg=cfg, p=p)
            if not path.is_file():
                missing.append(str(path))
                continue
            obj = json.loads(path.read_text(encoding="utf-8"))
            tm = obj["test metrics"]
            data[ds][p] = {k: float(v) for k, v in tm.items()}
    return data, missing


def weighted_f1(rec: dict[str, float]) -> float:
    return rec.get("f1_wt", rec.get("f1_weighted", float("nan")))


def build_matrix(data, metric: str, key_alt: str | None = None) -> np.ndarray:
    m = np.full((len(DATASETS), len(P_LIST)), np.nan)
    for i, (ds, _) in enumerate(DATASETS):
        for j, p in enumerate(P_LIST):
            rec = data[ds].get(p)
            if not rec:
                continue
            val = rec.get(metric)
            if val is None and key_alt is not None:
                val = rec.get(key_alt)
            if val is not None:
                m[i, j] = val
    return m


# ----------------------------------------------------------------------------- checks
def print_matrix(title: str, m: np.ndarray) -> None:
    print(title)
    print(" " * 24 + "".join(f"{('P' + str(p)):>10}" for p in P_LIST))
    for i, (ds, _) in enumerate(DATASETS):
        cells = "".join(("       n/a" if np.isnan(v) else f"{v:>10.4f}") for v in m[i])
        print(f"{ds:<24}{cells}")
    print()


def cross_check(data) -> int:
    """Compare raw JSON values with the transcribed summary table; raw wins on conflict."""
    print("=== Cross-check vs results/E-3_patch/E3_patch_summary.md (raw JSON = source of truth) ===")
    mismatches = 0
    for ds, _ in DATASETS:
        for metric, md_vals in MD_REFERENCE[ds].items():
            raw_vals = []
            for p in P_LIST:
                rec = data[ds].get(p)
                if metric == "f1_wt":
                    raw_vals.append(weighted_f1(rec) if rec else float("nan"))
                else:
                    raw_vals.append(rec.get(metric, float("nan")) if rec else float("nan"))
            raw_s = [v for v in raw_vals if not np.isnan(v)]
            deltas = [abs(a - b) for a, b in zip(raw_vals, md_vals) if not np.isnan(a)]
            ok = len(raw_s) == len(P_LIST) and all(d <= TOL for d in deltas)
            if not ok:
                mismatches += 1
            raw_txt = " ".join(f"{v:.4f}" for v in raw_vals)
            md_txt = " ".join(f"{v:.4f}" for v in md_vals)
            print(f"  {ds:<21} {metric:<8} raw [{raw_txt}] | md [{md_txt}] -> {'OK' if ok else 'MISMATCH'}")
    print(f"  -> {mismatches} mismatch(es)\n")
    return mismatches


# ----------------------------------------------------------------------------- figure
def draw_panel(ax, m: np.ndarray, title: str, has_ylabels: bool) -> None:
    """One seaborn heatmap, styled after the repo's existing heatmaps
    (analysis/param-graph.py): cmap="crest", annot with .4f, axis labels 14 pt,
    colour scale anchored at the smallest value in the matrix."""
    sns.heatmap(
        m,
        ax=ax,
        annot=True,
        cmap="crest",
        fmt=".4f",
        vmin=float(np.nanmin(m)),
        mask=np.isnan(m),
        xticklabels=[str(p) for p in P_LIST],
        yticklabels=[DATASET_DISPLAY[ds] for ds, _ in DATASETS] if has_ylabels else False,
    )
    ax.set_xlabel("Patch Size", fontsize=14)
    if has_ylabels:
        ax.set_ylabel("Dataset", fontsize=14)
    ax.set_title(title)


def make_figure(m_auc: np.ndarray, m_f1: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)
    draw_panel(axes[0], m_auc, "AUC", has_ylabels=True)
    draw_panel(axes[1], m_f1, r"$F1_{wt}$", has_ylabels=False)
    fig.savefig(PNG_OUT, dpi=300)
    fig.savefig(PDF_OUT)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    print(f"raw dir : {RAW_DIR}")
    data, missing = load_all()

    m_auc = build_matrix(data, "auc")
    m_f1 = build_matrix(data, "f1_wt", key_alt="f1_weighted")

    n_loaded = sum(len(v) for v in data.values())
    print(f"loaded  : {n_loaded}/{len(DATASETS) * len(P_LIST)} JSON files")
    if missing:
        print("missing :")
        for path in missing:
            print(f"  - {path}")
    print()

    print("=== P-sensitivity matrix (link & sign, SignDyGFormer, seed 42, E-3 v3 CN-fix) ===")
    print()
    print_matrix("AUC", m_auc)
    print_matrix("F1_weighted", m_f1)

    cross_check(data)

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "P", "auc", "f1_weighted"])
        for i, (ds, _) in enumerate(DATASETS):
            for j, p in enumerate(P_LIST):
                if not np.isnan(m_auc[i, j]):
                    w.writerow([ds, p, f"{m_auc[i, j]:.4f}", f"{m_f1[i, j]:.4f}"])
    print(f"csv     : {CSV_OUT}")

    make_figure(m_auc, m_f1)
    for out in (PNG_OUT, PDF_OUT):
        print(f"figure  : {out}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
