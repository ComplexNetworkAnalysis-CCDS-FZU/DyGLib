# -*- coding: utf-8 -*-
"""T14b 紧凑切片（Paper d1c9 §三.1）：
  linksign: sign_f1, f1_wt, f1_mac —— 7 方法 × 5 数据集
  sign:     f1_mac, auc, f1_binary —— 7 方法 × 5 数据集
输出：results/t14b_compact_20260925.csv / .txt
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "results/t14b_full_keys_20260925.csv")

DS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]
METHODS = ["gcn", "sgcn", "sigat", "tgn", "semba", "DyGFormer", "ours"]

specs = [
    ("linksign", ["sign_f1", "f1_wt", "f1_mac"], "sign_f1|f1_wt|f1_mac"),
    ("sign", ["f1_macro", "auc", "f1_binary"], "f1_macro|auc|f1_binary"),
]

lines = []
for task, mets, label in specs:
    sub = df[(df["task"] == task) & (df["metric"].isin(mets)) & (df["dataset"].isin(DS))]
    print(f"== {task}: metrics present = {sorted(sub['metric'].unique())}")
    piv = sub.pivot_table(index=["method", "dataset"], columns="metric", values="mean", aggfunc="first")
    piv = piv.reindex(pd.MultiIndex.from_product([METHODS, DS], names=["method", "dataset"]))
    # 紧凑文本：一行一个 方法×数据集
    lines.append(f"===== {task}（mean, 5 种子）=====")
    hdr = "method," + ",".join(f"{m}" for m in mets)
    lines.append("dataset | " + hdr)
    for (method, ds) in piv.index:
        row = piv.loc[(method, ds)]
        cells = []
        for m in mets:
            v = row.get(m)
            cells.append("—" if pd.isna(v) else f"{v:.4f}")
        lines.append(f"{ds:>22} | {method:<9} | " + " | ".join(cells))
    lines.append("")
    # 同时写 CSV（紧凑版）
    out = piv.reset_index()
    out.to_csv(ROOT / f"results/t14b_compact_{task}_20260925.csv", index=False)

txt = "\n".join(lines)
(ROOT / "results/t14b_compact_20260925.txt").write_text(txt + "\n", encoding="utf-8")
print(txt)
