# -*- coding: utf-8 -*-
"""CNS 行（采纳 3 组合 + 采样器替换）行值表：CNS-D vs 同代 G2 全口径（5 种子）。

来源：
  CNS：results/cns_g2/raw/{linksign|sign}/{ds}/*CNAS-D*P1.TE.G2.json
  全口径：results/grid_confirm/raw/{linksign|sign}/{ds}/*CNAS-E*P1.TE.G2.json（.G2 采纳件）
输出：results/cns_g2_table_20260930.txt
"""
import glob
import io
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
POINTS = [
    ("linksign", "RedditHyperlinkTitle", 15, 3, "linksign RT 15/3", ["f1_wt", "f1_mac", "sign_f1", "ap", "auc"]),
    ("linksign", "RedditHyperlinkBody", 60, 1, "linksign RB 60/1", ["f1_wt", "f1_mac", "sign_f1", "ap", "auc"]),
    ("sign", "RedditHyperlinkTitle", 60, 3, "sign RT 60/3", ["f1_macro", "f1_binary", "auc", "ap", "f1_weighted"]),
]
SEEDS = [42, 123, 456, 789, 1024]


def load(pat):
    out = {}
    for p in glob.glob(str(ROOT / pat)):
        if "-profiler" in p:
            continue
        s = int(re.search(r"seed(\d+)", Path(p).name).group(1))
        out[s] = json.load(open(p, encoding="utf-8"))["test metrics"]
    return out


lines = ["CNS 行值表（采样器替换 CNAS-D vs 全口径 CNAS-E（.G2 采纳件）；5 种子；test metrics）", ""]
for (task, ds, nn, lf, label, keys) in POINTS:
    cns = load(f"results/cns_g2/raw/{task}/{ds}/SignDyGFormer_seed*.NN-{nn}.LF-{lf}*.CNAS-D*.G2.json")
    full = load(f"results/grid_confirm/raw/{task}/{ds}/SignDyGFormer_seed*.NN-{nn}.LF-{lf}*.G2.json")
    seeds = sorted(set(cns) & set(full))
    lines.append(f"===== {label}（CNS {len(cns)}/5；对照 {len(full)}/5；配对 n={len(seeds)}）=====")
    if not seeds:
        lines.append("  数据缺失（待跑/待取）")
        lines.append("")
        continue
    for k in keys:
        if k not in cns[seeds[0]] or k not in full[seeds[0]]:
            continue
        cv = [float(cns[s][k]) for s in seeds]
        fv = [float(full[s][k]) for s in seeds]
        d = [(a - b) * 1000 for a, b in zip(cv, fv)]
        lines.append(f"  [{k:>12s}] CNS-D {st.mean(cv):.4f}±{st.stdev(cv):.4f} | "
                     f"CNAS-E {st.mean(fv):.4f}±{st.stdev(fv):.4f} | Δ {st.mean(d):+.1f}‰ "
                     f"（{sum(1 for x in d if x > 0)}/{len(d)} 正）")
        lines.append(f"      CNS-D 逐种子: " + " ".join(f"{v:.4f}" for v in cv))
    lines.append("")
lines.append("注：CNS 行 = 与采纳组合同配置、仅采样器替换（--no-module-common-neighbor-aware-sampler）；")
lines.append("按 f8c2 §三-1 定义，与主表 CNS 行同款协议。")

out = ROOT / "results/cns_g2_table_20260930.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"[ok] {out.relative_to(ROOT)}")
