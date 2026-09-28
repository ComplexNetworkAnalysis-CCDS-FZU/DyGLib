# -*- coding: utf-8 -*-
"""E1a×E1c 组合批判定（预登记，Paper 54b4 §三；linksign 5 ds × 5 seeds vs 同代际 Full）。

靶心（A 档）：RB f1_wt 转正（Δ≥+5‰ 且配对显著，或 ≥4/5 同向且点估计为正），且其余 ds Δf1_wt ≥ −5‰ 或不显著负。
B 档：RB 不再显著负 + AUC/f1_mac 保持 E1a 的正向（"伤害抹平 + 排序面保持"）。
C 档：仍无正收益 → 采样线正式收口。
输入：results/e1x/raw/{ds}/*.TF-E.RK-*.json ；对照 results/e1a_tailfill/raw_base/linksign/{ds}/*.json
输出：results/e1x_combo_verdict_20260928.txt （幂等覆盖）
"""
import glob
import json
import re
import statistics as st
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DS = ["RedditHyperlinkBody", "RedditHyperlinkTitle", "WikiVote", "BitcoinAlpha", "BitcoinOTC"]
METRICS = [("f1_wt", "f1_wt‰"), ("f1_mac", "f1_mac‰"), ("auc", "auc‰"), ("ap", "ap‰")]

lines = []


def say(s=""):
    lines.append(s)
    print(s)


def load_map(pattern):
    out = {}
    for p in sorted(glob.glob(str(ROOT / pattern))):
        if "-profiler" in p:
            continue
        seed = int(re.search(r"seed(\d+)", Path(p).name).group(1))
        out[seed] = json.load(open(p, encoding="utf-8"))["test metrics"]
    return out


say("E1a×E1c 组合批判定（linksign；5 种子配对；Δ‰ = E1X − Full；预登记见 54b4 §三）")
say()
summary = {}
for ds in DS:
    e1x = load_map(f"results/e1x/raw/{ds}/*.TF-E.RK-*.json")
    full = load_map(f"results/e1a_tailfill/raw_base/linksign/{ds}/*.json")
    seeds = sorted(set(e1x) & set(full))
    if not seeds:
        say(f"===== {ds}: 无数据（待重跑）=====")
        continue
    mtag = re.search(r"RK-(\d+)", list(glob.glob(str(ROOT / f'results/e1x/raw/{ds}/*.json')))[0])
    say(f"===== {ds}（n={len(seeds)}；RK-{mtag.group(1) if mtag else '?'}）=====")
    for key, lbl in METRICS:
        if key not in e1x[seeds[0]]:
            continue
        deltas = [(float(e1x[s][key]) - float(full[s][key])) * 1000 for s in seeds]
        mu = st.mean(deltas)
        if len(deltas) > 1 and st.stdev(deltas) > 0:
            t, p = stats.ttest_rel([float(e1x[s][key]) for s in seeds], [float(full[s][key]) for s in seeds])
        else:
            t, p = float("nan"), float("nan")
        pos = sum(1 for d in deltas if d > 0)
        detail = " ".join(f"{d:+.1f}" for d in deltas)
        say(f"  [{lbl:>9s}] 均值 {mu:+6.1f}‰  ({pos}/{len(deltas)} 正)  p={p:.4f}  逐种子: {detail}")
        summary.setdefault(ds, {})[key] = (mu, pos, len(deltas), p)
    say()

say("===== 档位判定（机械汇总，人工复核）=====")
rb = summary.get("RedditHyperlinkBody")
if rb and "f1_wt" in rb:
    mu, pos, n, p = rb["f1_wt"]
    cond_a_rb = (mu >= 5 and p < 0.05) or (pos >= n - 1 and mu > 0)
    others = [(ds, summary[ds]["f1_wt"][0]) for ds in DS if ds != "RedditHyperlinkBody" and ds in summary]
    cond_a_others = all((v >= -5) or True for _, v in others)  # 简化：打印供人工判断
    say(f"  靶心 RB f1_wt: Δ={mu:+.1f}‰ ({pos}/{n}) p={p:.4f} → A-RB 条件: {'满足' if cond_a_rb else '不满足'}")
    say(f"  其余 ds Δf1_wt: " + " ".join(f"{ds[:4]}={v:+.1f}" for ds, v in others))
    b_rb = not (mu < 0 and p < 0.05)  # RB 不显著负
    auc_pos = summary["RedditHyperlinkBody"].get("auc", (0,))[0]
    f1mac_pos = summary["RedditHyperlinkBody"].get("f1_mac", (0,))[0]
    say(f"  B 档条件: RB 不显著负={b_rb}（−5.0 判据见人工）；RB auc={auc_pos:+.1f}‰ f1_mac={f1mac_pos:+.1f}‰")
    say("  注：FX 双口径待伴行（训练出参后组行）。")
else:
    say("  RB 数据缺失（重跑中）——暂不判定。")

OUT = ROOT / "results/e1x_combo_verdict_20260928.txt"
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\n[ok] 写出 {OUT.relative_to(ROOT)}")
