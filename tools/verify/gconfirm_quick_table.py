# -*- coding: utf-8 -*-
"""采纳 3 点全指标快表（f8c2 §二 priority）：linksign RT 15/3、linksign RB 60/1、sign RT 60/3。

来源：results/grid_confirm/raw/{linksign|sign}/{ds}/*P1.TE.G2.json（5 种子）
输出：results/gconfirm_quick_table_20260929.txt（+ .csv）
"""
import csv
import glob
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POINTS = [
    ("linksign", "RedditHyperlinkTitle", 15, 3, "linksign RT 15/3（采纳）"),
    ("linksign", "RedditHyperlinkBody", 60, 1, "linksign RB 60/1（采纳·边缘）"),
    ("sign", "RedditHyperlinkTitle", 60, 3, "sign RT 60/3（采纳）"),
]
SEEDS = [42, 123, 456, 789, 1024]


def load_point(task, ds, nn, lf):
    out = {}
    for p in glob.glob(str(ROOT / f"results/grid_confirm/raw/{task}/{ds}/SignDyGFormer_seed*.NN-{nn}.LF-{lf}*.G2.json")):
        s = int(re.search(r"seed(\d+)", Path(p).name).group(1))
        out[s] = json.load(open(p, encoding="utf-8"))["test metrics"]
    return out


lines, csv_rows = [], []
lines.append("采纳 3 点全指标快表（.G2 五种子；test metrics 全键；mean±std）")
lines.append("")
for (task, ds, nn, lf, label) in POINTS:
    d = load_point(task, ds, nn, lf)
    keys = sorted(d[SEEDS[0]].keys())
    lines.append(f"===== {label}（n={len(d)}）=====")
    hdr = f"{'metric':<16s}" + "".join(f"{('seed'+str(s)):>10s}" for s in SEEDS) + f"{'mean':>10s}{'std':>9s}"
    lines.append(hdr)
    for k in keys:
        vals = [float(d[s][k]) for s in SEEDS if k in d[s]]
        if not vals:
            continue
        row = f"{k:<16s}" + "".join(f"{float(d[s].get(k, 'nan')):>10.4f}" for s in SEEDS)
        row += f"{st.mean(vals):>10.4f}{st.stdev(vals):>9.4f}"
        lines.append(row)
        csv_rows.append([task, ds, nn, lf, k] + [f"{float(d[s].get(k, 'nan')):.4f}" for s in SEEDS]
                        + [f"{st.mean(vals):.4f}", f"{st.stdev(vals):.4f}"])
    lines.append("")

out = ROOT / "results/gconfirm_quick_table_20260929.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
with open(ROOT / "results/gconfirm_quick_table_20260929.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["task", "ds", "nn", "lf", "metric"] + [f"seed{s}" for s in SEEDS] + ["mean", "std"])
    w.writerows(csv_rows)
print("\n".join(lines))
print(f"[ok] {out.relative_to(ROOT)} + csv")
