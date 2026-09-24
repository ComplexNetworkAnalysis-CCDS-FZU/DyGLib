# -*- coding: utf-8 -*-
"""生成 sign × w/o BTE（BTE-D）屏幕批：5 数据集 × seed42（非 TF 代，对齐主表 full）。

Paper ebcf §二：sign 侧 BTE LOO 缺口；先 5 runs 屏，方向为正再补 5 种子。
输出：tools/queue/insert_sign_nobte_20260924.txt
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools/queue/insert_sign_nobte_20260924.txt"

SIGN = {"WikiVote": (40, 15), "RedditHyperlinkTitle": (100, 1), "RedditHyperlinkBody": (60, 1),
        "BitcoinAlpha": (40, 15), "BitcoinOTC": (40, 15)}
TAIL = {"WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"}
PREFIX = ("@cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
          "&& conda activate gc && python train_link_sign_prediction.py ")

rows = []
for ds, (nn, lf) in SIGN.items():
    tail = " --tail-num 20000" if ds in TAIL else ""
    rows.append(
        PREFIX
        + f"--dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds 42"
        " --early-stop-notice f1_binary auc f1_weighted"
        " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
        " --no-module-balance-theory-encoder"
        f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail}"
    )

OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"写出 {len(rows)} 行 -> {OUT}")
