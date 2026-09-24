# -*- coding: utf-8 -*-
"""生成 BTE 四变体第一段筛查批（40 runs）：4 变体 × {linksign 5 ds, sign 5 ds} × seed42。

约束（Paper d350 + 用户已批）：默认关、代际标记 .B2/.B3/.B4/.B5；主对照 = full；
本轮不含 G1（B5/B3 为 G1 的替代变体，单独跑）。
输出：tools/queue/insert_bte_stage1_20260924.txt
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools/queue/insert_bte_stage1_20260924.txt"

LINK = {"WikiVote": (15, 10), "RedditHyperlinkTitle": (60, 1), "RedditHyperlinkBody": (80, 3),
        "BitcoinAlpha": (40, 15), "BitcoinOTC": (80, 5)}
SIGN = {"WikiVote": (40, 15), "RedditHyperlinkTitle": (100, 1), "RedditHyperlinkBody": (60, 1),
        "BitcoinAlpha": (40, 15), "BitcoinOTC": (40, 15)}
TAIL = {"WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"}
PREFIX = ("@cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
          "&& conda activate gc && python ")

VARIANTS = [
    ("B2", "--module-bte-b2-density-norm"),
    ("B3", "--module-bte-b3-default-marker"),
    ("B4", "--module-bte-b4-channel-gate"),
    ("B5", "--module-bte-b5-continuous-gate"),
]

rows = []
for vname, vflag in VARIANTS:
    # linksign（3class）
    for ds, (nn, lf) in LINK.items():
        tail = " --tail-num 20000" if ds in TAIL else ""
        rows.append(
            PREFIX + "train_sign_link_3class_prediction.py"
            + f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds 42"
            + " --early-stop-notice f1_wt f1_mic ap f1_mac auc"
            + " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
            + f" {vflag}"
            + f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail}"
        )
    # sign（binary）
    for ds, (nn, lf) in SIGN.items():
        tail = " --tail-num 20000" if ds in TAIL else ""
        rows.append(
            PREFIX + "train_link_sign_prediction.py"
            + f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds 42"
            + " --early-stop-notice f1_binary auc f1_weighted"
            + " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
            + f" {vflag}"
            + f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail}"
        )

OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"写出 {len(rows)} 行 -> {OUT}")
