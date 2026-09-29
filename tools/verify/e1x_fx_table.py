# -*- coding: utf-8 -*-
"""组合批（RB）FX 固定阈值伴行 vs 主口径对照表（5 种子）。

主口径：results/e1x/raw/RedditHyperlinkBody/*.TF-E.RK-80.json（自适应/val 阈值）
FX    ：results/e1x/fx/raw/*.TF-E.RK-80.EVT.FX.json（固定阈值 = 各 seed param.json best_thr）
输出：results/e1x_fx_table_20260929.txt
"""
import glob
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M = ["f1_wt", "f1_mac", "auc", "ap"]


def load(pat):
    out = {}
    for p in glob.glob(str(ROOT / pat)):
        s = int(re.search(r"seed(\d+)", Path(p).name).group(1))
        out[s] = json.load(open(p, encoding="utf-8"))["test metrics"]
    return out


main = load("results/e1x/raw/RedditHyperlinkBody/*.TF-E.RK-80.json")
fx = load("results/e1x/fx/raw/*.TF-E.RK-80.EVT.FX.json")

lines = []
lines.append("组合批（RB, NN-80.LF-3.TF-E.RK-80）FX 固定阈值伴行 vs 主口径（5 种子）")
lines.append("")
seeds = sorted(set(main) & set(fx))
for m in M:
    lines.append(f"===== {m} =====")
    d = []
    for s in seeds:
        a, b = float(main[s][m]), float(fx[s][m])
        d.append((b - a) * 1000)
        lines.append(f"  seed{s:<5d} 主={a:.4f}  FX={b:.4f}  Δ={d[-1]:+.1f}‰")
    mu = (st.mean([float(fx[s][m]) for s in seeds]) - st.mean([float(main[s][m]) for s in seeds])) * 1000
    pos = sum(1 for x in d if x >= -0.05)
    lines.append(f"  → 均值 主={st.mean([float(main[s][m]) for s in seeds]):.4f} "
                 f"FX={st.mean([float(fx[s][m]) for s in seeds]):.4f} Δ={mu:+.1f}‰（{pos}/{len(d)} 不低于）")
    lines.append("")
lines.append("注：FX 行 = eval-only 装载训练 ckpt、以各 seed param.json 的 best_sign_thr/best_exist_thr")
lines.append("固定阈值重评（结果名 .EVT.FX）；阈值控制下与主口径方向一致即说明结论不由阈值选择漂移驱动。")

out = ROOT / "results/e1x_fx_table_20260929.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"[ok] 写出 {out.relative_to(ROOT)}")
