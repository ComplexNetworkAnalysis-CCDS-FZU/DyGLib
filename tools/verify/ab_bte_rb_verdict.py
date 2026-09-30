# -*- coding: utf-8 -*-
"""BTE×RB 结构交叉复核 A/B 判定（90d1；预登记）。

A = 组合窗（TF-E.RK-80，80/3）+ 删 BTE；对照 = 既有组合件（results/e1x/raw/RedditHyperlinkBody）
B = 采纳配置（60/1）+ 删 BTE；   对照 = 既有 .G2 件（results/grid_confirm/raw/linksign/RedditHyperlinkBody）
Δ = f1_wt（full − NoBTE），同种子配对；通过 = Δ≥+5‰ 且 p<.05，或 ≥4/5 同向且点估计为正。
另报 auc / f1_mac / ap（同口径）。
输出：results/ab_bte_rb_verdict_20260930.txt
"""
import glob
import io
import json
import re
import statistics as st
import sys
from pathlib import Path

from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
M = [("f1_wt", "f1_wt‰"), ("f1_mac", "f1_mac‰"), ("auc", "auc‰"), ("ap", "ap‰")]

CASES = [
    ("A", "组合窗（TF-E.RK-80）删 BTE",
     "results/ab_bte_rb/raw/A/*.TF-E.RK-80.json",
     "results/e1x/raw/RedditHyperlinkBody/*.TF-E.RK-80.json"),
    ("B", "采纳配置（60/1）删 BTE",
     "results/ab_bte_rb/raw/B/*.G2.json",
     "results/grid_confirm/raw/linksign/RedditHyperlinkBody/*.NN-60.LF-1*.G2.json"),
]


def load(pat):
    out = {}
    for p in sorted(glob.glob(str(ROOT / pat))):
        if "-profiler" in p:
            continue
        seed = int(re.search(r"seed(\d+)", Path(p).name).group(1))
        out[seed] = json.load(open(p, encoding="utf-8"))["test metrics"]
    return out


lines = []


def say(s=""):
    lines.append(s)
    print(s)


say("BTE×RB 结构交叉复核 A/B 判定（预登记 90d1；Δ = f1_wt(full - NoBTE)，同种子配对）")
say("通过 = Δ≥+5‰ 且 p<.05，或 ≥4/5 同向且点估计为正")
say()
results = {}
for (tag, label, nobte_pat, full_pat) in CASES:
    nobte = load(nobte_pat)
    full = load(full_pat)
    seeds = sorted(set(nobte) & set(full))
    say(f"===== {tag}：{label}（n={len(seeds)} 配对种子：{seeds}）=====")
    if not seeds:
        say("  数据缺失（待跑/待取）")
        say()
        continue
    for key, lbl in M:
        if key not in nobte[seeds[0]] or key not in full[seeds[0]]:
            continue
        d = [(float(full[s][key]) - float(nobte[s][key])) * 1000 for s in seeds]
        mu = st.mean(d)
        try:
            t, p = stats.ttest_rel([float(full[s][key]) for s in seeds],
                                   [float(nobte[s][key]) for s in seeds])
        except Exception:
            t, p = float("nan"), float("nan")
        pos = sum(1 for x in d if x > 0)
        detail = " ".join(f"{x:+.1f}" for x in d)
        star = " ★" if key == "f1_wt" else ""
        say(f"  [{lbl:>9s}]{star} 均值 {mu:+6.1f}‰（{pos}/{len(d)} 正） p={p:.4f}  逐种子: {detail}")
        if key == "f1_wt":
            gate = (mu >= 5 and p < 0.05) or (pos >= len(d) - 1 and mu > 0)
            results[tag] = (mu, pos, len(d), p, gate)
    say()

say("===== 预登记结论（f1_wt）=====")
for tag, label in (("A", "采样窗口修复后，BTE 在 RB 转正"), ("B", "采纳配置下 BTE 在 RB 转正")):
    if tag in results:
        mu, pos, n, p, gate = results[tag]
        verdict = ("通过 ⇒ 结论文本：" + label) if gate else "未过 ⇒ 如实并入『适用边界』结论"
        say(f"  {tag}: Δ={mu:+.1f}‰（{pos}/{n}） p={p:.4f} → {verdict}")
    else:
        say(f"  {tag}: 数据缺失")
say()
say("注：A 对照 = 既有组合件（.BTE-E）；B 对照 = 既有 .G2 件。用于 §4.3 消融叙述/附录 B，不替换默认、不动主表。")

out = ROOT / "results/ab_bte_rb_verdict_20260930.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\n[ok] {out.relative_to(ROOT)}")
