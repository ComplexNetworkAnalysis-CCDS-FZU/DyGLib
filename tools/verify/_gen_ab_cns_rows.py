# -*- coding: utf-8 -*-
"""90d1（BTE×RB 结构交叉复核 A/B）+ f8c2（CNS 15 runs）行生成（2026-09-29 深夜）。

A = 组合窗（TF-E + RK-80，80/3）+ 删 BTE（--no-module-balance-theory-encoder）
    → 结果名 …RAS-E.RASE-E.BTE-D.CNAS-E.P1.TE.TF-E.RK-80（对照=既有 .BTE-E.TF-E.RK-80）
B = 采纳配置（NN60/LF1）+ 删 BTE + .G2 标记
    → …BTE-D.CNAS-E.P1.TE.G2（对照=既有 .G2）
CNS 3 行（f8c2 §三-1）= 采纳组合（RT 15/3、RB 60/1、sign RT 60/3）+ 采样器替换
    （--no-module-common-neighbor-aware-sampler；与主表 CNS 行同款）→ …CNAS-D….G2
标记说明：用既有 .BTE-D / .CNAS-D 代际机制（与 LOO/noBTE、主表 CNS 批同款；文件名必带、防覆盖）。
输出：tools/queue/ab_cns_20260929.txt（5 行 = A + B + 3×CNS；每行 5 种子）
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools/queue/ab_cns_20260929.txt"

PREFIX = ("@cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
          "&& conda activate gc && python ")
SEEDS5 = "42 123 456 789 1024"
LINK_S = "--early-stop-notice f1_wt f1_mic ap f1_mac auc --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
SIGN_S = "--early-stop-notice f1_binary auc f1_weighted --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"

rows = []

# A：组合窗 + 删 BTE（linksign RB 80/3）
rows.append(
    PREFIX + "train_sign_link_3class_prediction.py"
    " --dataset-name RedditHyperlinkBody --model SignDyGFormer --gpu @GPU@"
    f" --seeds {SEEDS5} {LINK_S} --no-module-balance-theory-encoder"
    " --cnas-tail-fill --recent-block 80"
    " --batch-size 200 --num-neighbors 80 --common-neighbors-look-forward 3 --tail-num 20000"
)

# B：采纳配置 + 删 BTE（linksign RB 60/1 + G2 标记）
rows.append(
    PREFIX + "train_sign_link_3class_prediction.py"
    " --dataset-name RedditHyperlinkBody --model SignDyGFormer --gpu @GPU@"
    f" --seeds {SEEDS5} {LINK_S} --no-module-balance-theory-encoder --grid-confirm-g2"
    " --batch-size 200 --num-neighbors 60 --common-neighbors-look-forward 1 --tail-num 20000"
)

# CNS 3 行：采纳组合 + 采样器替换（linksign RT 15/3、linksign RB 60/1、sign RT 60/3）
rows.append(
    PREFIX + "train_sign_link_3class_prediction.py"
    " --dataset-name RedditHyperlinkTitle --model SignDyGFormer --gpu @GPU@"
    f" --seeds {SEEDS5} {LINK_S} --no-module-common-neighbor-aware-sampler --grid-confirm-g2"
    " --batch-size 200 --num-neighbors 15 --common-neighbors-look-forward 3 --tail-num 20000"
)
rows.append(
    PREFIX + "train_sign_link_3class_prediction.py"
    " --dataset-name RedditHyperlinkBody --model SignDyGFormer --gpu @GPU@"
    f" --seeds {SEEDS5} {LINK_S} --no-module-common-neighbor-aware-sampler --grid-confirm-g2"
    " --batch-size 200 --num-neighbors 60 --common-neighbors-look-forward 1 --tail-num 20000"
)
rows.append(
    PREFIX + "train_link_sign_prediction.py"
    " --dataset-name RedditHyperlinkTitle --model SignDyGFormer --gpu @GPU@"
    f" --seeds {SEEDS5} {SIGN_S} --no-module-common-neighbor-aware-sampler --grid-confirm-g2"
    " --batch-size 200 --num-neighbors 60 --common-neighbors-look-forward 3 --tail-num 20000"
)

OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"写出 {len(rows)} 行 -> {OUT}")
for r in rows:
    print(" ", r[:150], "...")
