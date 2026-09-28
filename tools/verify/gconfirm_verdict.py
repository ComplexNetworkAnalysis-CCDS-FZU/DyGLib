# -*- coding: utf-8 -*-
"""网格候选点 5 种子确认批判定（预登记：Paper a43d，用户已批"批"）。

5 点 × 5 种子 = 25 runs；对照 = 同代际 Full（同种子配对）。
配对来源：
  linksign → results/e1a_tailfill/raw_base/linksign/{ds}/…NN-{ctrl_nn}.LF-{ctrl_lf}….json
  sign     → results/sign_valthr/raw/{ds}/…NN-{ctrl_nn}.LF-{ctrl_lf}….json
候选：results/grid_confirm/raw/{task}/{ds}/…P1.TE.G2.json（fetch --set gconfirm）
主指标：linksign = f1_wt；sign = f1_macro（Paper 2877；落盘键名 f1_macro）。
Gate 2（预登记）：主指标 Δ ≥ +5‰ 且配对 p<.05，或 ≥4/5 同向且点估计为正。
Gate 1（val 侧）另行交付（extract_val_from_logs）；采纳与否 = 9-30 检查点。
输出：results/gconfirm_verdict_20260929.txt（幂等覆盖）
"""
import glob
import json
import re
import statistics as st
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]

# (task, ds, cand_nn, cand_lf, ctrl_nn, ctrl_lf, label)
POINTS = [
    ("linksign", "RedditHyperlinkTitle", 15, 3, 60, 1, "linksign RT 15/3（当前 60/1）"),
    ("linksign", "RedditHyperlinkBody", 60, 1, 80, 3, "linksign RB 60/1（当前 80/3）"),
    ("sign", "RedditHyperlinkTitle", 60, 3, 100, 1, "sign RT 60/3（当前 100/1）"),
    ("sign", "RedditHyperlinkBody", 40, 1, 60, 1, "sign RB 40/1（当前 60/1）"),
    ("sign", "WikiVote", 15, 10, 40, 15, "sign WV 15/10（当前 40/15）"),
]

METRICS = {
    "linksign": [("f1_wt", "f1_wt‰"), ("f1_mac", "f1_mac‰"), ("auc", "auc‰"), ("ap", "ap‰")],
    "sign": [("f1_macro", "f1_macro‰"), ("f1_binary", "f1_bin‰"), ("auc", "auc‰"), ("ap", "ap‰")],
}
PRIMARY = {"linksign": "f1_wt", "sign": "f1_macro"}

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


def fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else str(x)


say("网格候选点 5 种子确认批判定（对比 = 同代际 Full 同种子配对；Δ‰ = 候选 − 当前）")
say("预登记：Gate2 = 主指标 Δ≥+5‰ 且 p<.05，或 ≥4/5 同向且点估计为正；采纳与否 = 9-30 检查点")
say()
gate_rows = []
for (task, ds, nn, lf, cnn, clf, label) in POINTS:
    cand = load_map(f"results/grid_confirm/raw/{task}/{ds}/SignDyGFormer_seed*.NN-{nn}.LF-{lf}*.G2.json")
    if task == "linksign":
        ctrl = load_map(f"results/e1a_tailfill/raw_base/linksign/{ds}/SignDyGFormer_seed*.NN-{cnn}.LF-{clf}*.json")
    else:
        ctrl = load_map(f"results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.NN-{cnn}.LF-{clf}*.json")
    seeds = sorted(set(cand) & set(ctrl))
    say(f"===== {label}====")
    if not seeds:
        say("  数据缺失（待跑/待取）")
        say()
        continue
    say(f"  （n={len(seeds)} 配对种子：{seeds}；候选 {len(cand)} 件 / 对照 {len(ctrl)} 件）")
    for key, lbl in METRICS[task]:
        if key not in cand[seeds[0]] or key not in ctrl[seeds[0]]:
            continue
        d = [(float(cand[s][key]) - float(ctrl[s][key])) * 1000 for s in seeds]
        mu = st.mean(d)
        try:
            t, p = stats.ttest_rel([float(cand[s][key]) for s in seeds],
                                   [float(ctrl[s][key]) for s in seeds])
        except Exception:
            t, p = float("nan"), float("nan")
        pos = sum(1 for x in d if x > 0)
        detail = " ".join(f"{x:+.1f}" for x in d)
        star = " ★" if key == PRIMARY[task] else ""
        say(f"  [{lbl:>9s}]{star} 均值 {mu:+6.1f}‰  ({pos}/{len(d)} 正)  p={p:.4f}  逐种子: {detail}")
        if key == PRIMARY[task]:
            gate2 = (mu >= 5 and p < 0.05) or (pos >= len(d) - 1 and mu > 0)
            gate_rows.append((label, mu, pos, len(d), p, gate2))
    say()

say("===== Gate 2 机械汇总（主指标）=====")
for (label, mu, pos, n, p, ok) in gate_rows:
    say(f"  {label}: Δ={mu:+.1f}‰ ({pos}/{n}) p={p:.4f} → {'通过' if ok else '未通过'}")
say()
say("注：FX 固定阈值伴行（linksign 两点）待训练出参→组行→补表；Gate 1 = val 侧另行交付。")

OUT = ROOT / "results/gconfirm_verdict_20260929.txt"
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\n[ok] 写出 {OUT.relative_to(ROOT)}")
