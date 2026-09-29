# -*- coding: utf-8 -*-
"""G2 确认批 FX 固定阈值伴行 vs 主口径对照（linksign RT 15/3、RB 60/1；5 种子）。

主口径：results/grid_confirm/raw/linksign/{ds}/*P1.TE.G2.json
FX    ：results/grid_confirm/fx/{ds}/*P1.TE.G2.EVT.FX.json
输出：results/gconfirm_fx_table_20260929.txt
"""
import glob
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POINTS = [
    ("RedditHyperlinkTitle", 15, 3, "linksign RT 15/3"),
    ("RedditHyperlinkBody", 60, 1, "linksign RB 60/1"),
]
M = ["f1_wt", "f1_mac", "auc", "ap"]


def load(pat):
    out = {}
    for p in glob.glob(str(ROOT / pat)):
        s = int(re.search(r"seed(\d+)", Path(p).name).group(1))
        out[s] = json.load(open(p, encoding="utf-8"))["test metrics"]
    return out


lines = ["G2 确认批 FX 固定阈值伴行 vs 主口径（5 种子；linksign 两点）", ""]
for (ds, nn, lf, label) in POINTS:
    main = load(f"results/grid_confirm/raw/linksign/{ds}/SignDyGFormer_seed*.NN-{nn}.LF-{lf}*.G2.json")
    fx = load(f"results/grid_confirm/fx/{ds}/SignDyGFormer_seed*.NN-{nn}.LF-{lf}*.G2.EVT.FX.json")
    lines.append(f"===== {label}（主 {len(main)} 件 / FX {len(fx)} 件）=====")
    seeds = sorted(set(main) & set(fx))
    for m in M:
        d = [(float(fx[s][m]) - float(main[s][m])) * 1000 for s in seeds]
        lines.append(f"  [{m:>7s}] 主={st.mean([float(main[s][m]) for s in seeds]):.4f} "
                     f"FX={st.mean([float(fx[s][m]) for s in seeds]):.4f} "
                     f"Δ={st.mean(d):+.1f}‰  逐种子: " + " ".join(f"{x:+.1f}" for x in d))
    lines.append("")
lines.append("注：同组合批经验——正确构造的 FX 与主口径应逐位一致（主口径本就 val-thr 选择）。")
out = ROOT / "results/gconfirm_fx_table_20260929.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"[ok] {out.relative_to(ROOT)}")
