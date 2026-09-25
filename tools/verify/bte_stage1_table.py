# -*- coding: utf-8 -*-
"""BTE 四变体第一段总表（B2/B3/B4/B5 × 5 ds × 2 任务；seed42 屏幕）。

对比：linksign → results/bte_stage1/raw/linksign vs results/e1a_tailfill/raw_base/linksign
      sign     → results/bte_stage1/raw/sign     vs results/sign_valthr/raw
输出：results/bte_stage1_table_20260925.{txt,csv}
用法：python tools/verify/bte_stage1_table.py
"""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]
VARIANTS = ["B2", "B3", "B4", "B5"]
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


L = []
ap_ = L.append
csv_rows = []
ap_("BTE 四变体第一段总表（seed42 屏幕；Δ‰ 正 = 变体更高；2026-09-25）")
ap_("")

for task, d, metrics in (("linksign", "linksign", LINK), ("sign", "sign", SIGN)):
    ap_(f"===== {task} =====")
    for tag in VARIANTS:
        ap_(f"-- {tag}")
        ap_(f"{'ds':<22}" + "".join(f"{k:>12}" for k in metrics))
        for ds in DS:
            test = load(f"results/bte_stage1/raw/{d}/{ds}/*.json") if False else load(
                f"results/bte_stage1/raw/{d}/{ds}/*.{tag}.json")
            if task == "linksign":
                base = load(f"results/e1a_tailfill/raw_base/linksign/{ds}/*.json")
            else:
                base = load(f"results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.json")
            cells = []
            for k in metrics:
                if 42 in test and 42 in base:
                    dv = (test[42][k] - base[42][k]) * 1000
                    cells.append(f"{dv:+.1f}")
                    csv_rows.append((task, tag, ds, k, dv))
                else:
                    cells.append("—")
            ap_(f"{ds:<22}" + "".join(f"{c:>12}" for c in cells))
        ap_("")

out_txt = ROOT / "results/bte_stage1_table_20260925.txt"
out_txt.write_text("\n".join(L) + "\n", encoding="utf-8")
out_csv = ROOT / "results/bte_stage1_table_20260925.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["task", "variant", "dataset", "metric", "delta_permille"])
    for r in csv_rows:
        w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.2f}"])
print(f"写出 {out_txt}")
print(f"写出 {out_csv}")
print("\n".join(L))
