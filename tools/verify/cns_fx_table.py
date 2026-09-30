# -*- coding: utf-8 -*-
"""CNS 行 FX 固定阈值伴行 vs 主口径对照（linksign RT/RB 的 CNS 行；5 种子）。

主口径：results/cns_g2/raw/linksign/{ds}/*CNAS-D*P1.TE.G2.json
FX    ：results/cns_g2/fx/{ds}/*CNAS-D*P1.TE.G2.EVT.FX.json
输出：results/cns_fx_table_20260930.txt
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
    ("RedditHyperlinkTitle", 15, 3, "linksign RT 15/3（CNS 行）"),
    ("RedditHyperlinkBody", 60, 1, "linksign RB 60/1（CNS 行）"),
]
M = ["f1_wt", "f1_mac", "auc", "ap"]


def load(pat):
    out = {}
    for p in glob.glob(str(ROOT / pat)):
        if "-profiler" in p:
            continue
        s = int(re.search(r"seed(\d+)", Path(p).name).group(1))
        out[s] = json.load(open(p, encoding="utf-8"))["test metrics"]
    return out


lines = ["CNS 行 FX 固定阈值伴行 vs 主口径（5 种子）", ""]
for (ds, nn, lf, label) in POINTS:
    main = load(f"results/cns_g2/raw/linksign/{ds}/SignDyGFormer_seed*.NN-{nn}.LF-{lf}*.CNAS-D*.G2.json")
    fx = load(f"results/cns_g2/fx/{ds}/SignDyGFormer_seed*.NN-{nn}.LF-{lf}*.CNAS-D*.G2.EVT.FX.json")
    seeds = sorted(set(main) & set(fx))
    lines.append(f"===== {label}（主 {len(main)} 件 / FX {len(fx)} 件；配对 n={len(seeds)}）=====")
    if not seeds:
        lines.append("  无配对数据")
        lines.append("")
        continue
    for m in M:
        d = [(float(fx[s][m]) - float(main[s][m])) * 1000 for s in seeds]
        lines.append(f"  [{m:>7s}] 主={st.mean([float(main[s][m]) for s in seeds]):.4f} "
                     f"FX={st.mean([float(fx[s][m]) for s in seeds]):.4f} Δ={st.mean(d):+.1f}‰ "
                     f"（{sum(1 for x in d if abs(x) < 0.05)}/{len(d)} 逐位一致）")
    lines.append("")
lines.append("注：FX = eval-only 装载训练 ckpt、以各 seed param.json best_thr 固定阈值重评（.EVT.FX）；")
lines.append("正确构造时应与主口径逐位一致（Δ=0.0‰）。")

out = ROOT / "results/cns_fx_table_20260930.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"[ok] {out.relative_to(ROOT)}")
