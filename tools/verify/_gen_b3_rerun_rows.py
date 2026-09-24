# -*- coding: utf-8 -*-
"""B3 重跑批（10 行）：修复构造期设备 bug 后重发 B3（linksign 5 + sign 5，seed42）。

背景：2026-09-25 凌晨发现 B3 行（队列 375–384）全部在构造器抛设备不匹配
（layer 仍在 CPU，而 init 前向用 device='cuda:0' 张量）。修复后重跑。
输出：tools/queue/insert_bte_b3_rerun_20260925.txt
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools/queue/insert_bte_b3_rerun_20260925.txt"

LINK = {"WikiVote": (15, 10), "RedditHyperlinkTitle": (60, 1), "RedditHyperlinkBody": (80, 3),
        "BitcoinAlpha": (40, 15), "BitcoinOTC": (80, 5)}
SIGN = {"WikiVote": (40, 15), "RedditHyperlinkTitle": (100, 1), "RedditHyperlinkBody": (60, 1),
        "BitcoinAlpha": (40, 15), "BitcoinOTC": (40, 15)}
TAIL = {"WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"}
PREFIX = ("@cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
          "&& conda activate gc && python ")

rows = []
for ds, (nn, lf) in LINK.items():
    tail = " --tail-num 20000" if ds in TAIL else ""
    rows.append(
        PREFIX + "train_sign_link_3class_prediction.py"
        + f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds 42"
        + " --early-stop-notice f1_wt f1_mic ap f1_mac auc"
        + " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
        + " --module-bte-b3-default-marker"
        + f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail}"
    )
for ds, (nn, lf) in SIGN.items():
    tail = " --tail-num 20000" if ds in TAIL else ""
    rows.append(
        PREFIX + "train_link_sign_prediction.py"
        + f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds 42"
        + " --early-stop-notice f1_binary auc f1_weighted"
        + " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
        + " --module-bte-b3-default-marker"
        + f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail}"
    )

OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"写出 {len(rows)} 行 -> {OUT}")
