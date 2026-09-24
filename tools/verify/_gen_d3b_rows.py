# -*- coding: utf-8 -*-
"""生成 D3b/T15 逐样本转储行（2026-09-24）。

15 行 = 5 数据集 × {full, noBTE(LOO idx8), G1}，linksign，seed42，eval-only + --dump-samples。
说明：三系 ckpt 均为**非 TF 代（.TE）**，与 results/bte_sparsity/*.npz（full 配置掩码）对齐；
      full/noBTE 用 LOO 批的 NN-Best.LF-Best 名；G1 用具体 NN/LF 名。
输出：tools/queue/insert_d3b_20260924.txt"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools/queue/insert_d3b_20260924.txt"

LINK = {"WikiVote": (15, 10), "RedditHyperlinkTitle": (60, 1), "RedditHyperlinkBody": (80, 3),
        "BitcoinAlpha": (40, 15), "BitcoinOTC": (80, 5)}
TAIL = {"WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"}
PREFIX = ("@cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
          "&& conda activate gc && python train_sign_link_3class_prediction.py ")

rows = []
for ds, (nn, lf) in LINK.items():
    tail = " --tail-num 20000" if ds in TAIL else ""
    common = (
        f"--dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds 42"
        " --early-stop-notice f1_wt f1_mic ap f1_mac auc"
        " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
    )
    # full
    rows.append(
        PREFIX + common
        + " --eval-only"
        + " --eval-ckpt-name SignDyGFormer_seed42.NN-Best.LF-Best.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE"
        + f" --dump-samples results/samples_dump/{ds}"
        + f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail}"
    )
    # noBTE（module idx8：--no-module-balance-theory-encoder）
    rows.append(
        PREFIX + common
        + " --no-module-balance-theory-encoder --eval-only"
        + " --eval-ckpt-name SignDyGFormer_seed42.NN-Best.LF-Best.RAS-E.RASE-E.BTE-D.CNAS-E.P1.TE"
        + f" --dump-samples results/samples_dump/{ds}"
        + f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail}"
    )
    # G1
    rows.append(
        PREFIX + common
        + " --module-bte-evidence-gate --eval-only"
        + f" --eval-ckpt-name SignDyGFormer_seed42.NN-{nn}.LF-{lf}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.G1"
        + f" --dump-samples results/samples_dump/{ds}"
        + f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail}"
    )

OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"写出 {len(rows)} 行 -> {OUT}")
