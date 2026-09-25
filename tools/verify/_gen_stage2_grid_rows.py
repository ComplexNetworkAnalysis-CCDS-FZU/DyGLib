# -*- coding: utf-8 -*-
"""Paper 0656/78f6 排程：BTE 阶段二（B2/B4/B5 linksign 各 25 runs）+ sign-RT noBTE 扩种子 + 网格方案 B（150 runs）。

行序：
  1) B2 linksign × 5 ds（每行 5 种子；RB 行提前）——25 runs
  2) B4 linksign × 5 ds ——25 runs
  3) B5 linksign × 5 ds ——25 runs（探索性）
  4) sign RT noBTE 扩 4 种子（123/456/789/1024）——4 runs
  5) 网格方案 B：RT/RB/WV × {sign, linksign} × 25 格（NN∈{15,40,60,80,100}×LF∈{1,3,5,10,15}）——150 runs
输出：tools/queue/insert_stage2_grid_20260925.txt
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools/queue/insert_stage2_grid_20260925.txt"

LINK = {"WikiVote": (15, 10), "RedditHyperlinkTitle": (60, 1), "RedditHyperlinkBody": (80, 3),
        "BitcoinAlpha": (40, 15), "BitcoinOTC": (80, 5)}
SIGN = {"WikiVote": (40, 15), "RedditHyperlinkTitle": (100, 1), "RedditHyperlinkBody": (60, 1),
        "BitcoinAlpha": (40, 15), "BitcoinOTC": (40, 15)}
TAIL = {"WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"}
PREFIX = ("@cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
          "&& conda activate gc && python ")
SEEDS5 = "42 123 456 789 1024"


def tail(ds: str) -> str:
    return " --tail-num 20000" if ds in TAIL else ""


rows = []

# 1–3) BTE 阶段二（linksign；RB 优先）
VARIANTS = [
    ("B2", "--module-bte-b2-density-norm"),
    ("B4", "--module-bte-b4-channel-gate"),
    ("B5", "--module-bte-b5-continuous-gate"),
]
DS_ORDER = ["RedditHyperlinkBody", "WikiVote", "RedditHyperlinkTitle", "BitcoinAlpha", "BitcoinOTC"]
for tag, flag in VARIANTS:
    for ds in DS_ORDER:
        nn, lf = LINK[ds]
        rows.append(
            PREFIX + "train_sign_link_3class_prediction.py"
            + f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds {SEEDS5}"
            + " --early-stop-notice f1_wt f1_mic ap f1_mac auc"
            + " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
            + f" {flag}"
            + f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail(ds)}"
        )

# 4) sign RT noBTE 扩 4 种子
nn, lf = SIGN["RedditHyperlinkTitle"]
rows.append(
    PREFIX + "train_link_sign_prediction.py"
    + " --dataset-name RedditHyperlinkTitle --model SignDyGFormer --gpu @GPU@ --seeds 123 456 789 1024"
    + " --early-stop-notice f1_binary auc f1_weighted"
    + " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
    + " --no-module-balance-theory-encoder"
    + f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail('RedditHyperlinkTitle')}"
)

# 5) 网格方案 B（150 runs；seed42 单种子；轴 NN{15,40,60,80,100} × LF{1,3,5,10,15}）
GRID_DS = ["RedditHyperlinkTitle", "RedditHyperlinkBody", "WikiVote"]
for ds in GRID_DS:
    for nn in [15, 40, 60, 80, 100]:
        for lf in [1, 3, 5, 10, 15]:
            rows.append(
                PREFIX + "train_sign_link_3class_prediction.py"
                + f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds 42"
                + " --early-stop-notice f1_wt f1_mic ap f1_mac auc"
                + " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
                + f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail(ds)}"
            )
for ds in GRID_DS:
    for nn in [15, 40, 60, 80, 100]:
        for lf in [1, 3, 5, 10, 15]:
            rows.append(
                PREFIX + "train_link_sign_prediction.py"
                + f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds 42"
                + " --early-stop-notice f1_binary auc f1_weighted"
                + " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
                + f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail(ds)}"
            )

OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"写出 {len(rows)} 行（15 阶段二 + 1 signRT + 150 网格）-> {OUT}")
