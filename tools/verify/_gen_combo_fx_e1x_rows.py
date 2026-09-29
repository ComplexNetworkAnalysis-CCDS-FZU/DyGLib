# -*- coding: utf-8 -*-
"""E1a×E1c 组合批（RB）固定阈值 FX 伴行行生成（2026-09-29）。

阈值来源（服务器采集，2026-09-29 08:30）：RB `.TF-E.RK-80` 各 seed param.json 的
best_sign_thr / best_exist_thr：
  seed42: 0.35000000000000003 / 0.01；123: 0.27 / 0.01；456: 0.31 / 0.01；
  789: 0.4 / 0.01；1024: 0.29000000000000004 / 0.01
行格式 = 训练行同 flags（RAS-E/RASE-E + TF-E + RK-80，无 G1）+ `--eval-only
--eval-ckpt-name …TF-E.RK-80 --sign-thr <t> --exist-thr <e>` → 结果名 …TF-E.RK-80.EVT.FX.json
输出：tools/queue/combo_fx_e1x_20260929.txt（5 行）
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools/queue/combo_fx_e1x_20260929.txt"

PREFIX = ("@cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
          "&& conda activate gc && python ")

# 服务器 param.json 采集值（repr 保真）
THR = {
    42: (0.35000000000000003, 0.01),
    123: (0.27, 0.01),
    456: (0.31, 0.01),
    789: (0.4, 0.01),
    1024: (0.29000000000000004, 0.01),
}

rows = []
for s, (st, et) in THR.items():
    base = f"SignDyGFormer_seed{s}.NN-80.LF-3.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.TF-E.RK-80"
    rows.append(
        PREFIX + "train_sign_link_3class_prediction.py"
        " --dataset-name RedditHyperlinkBody --model SignDyGFormer --gpu @GPU@"
        f" --seeds {s}"
        " --early-stop-notice f1_wt f1_mic ap f1_mac auc"
        " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
        " --cnas-tail-fill --recent-block 80"
        f" --eval-only --eval-ckpt-name {base}"
        f" --sign-thr {repr(st)} --exist-thr {repr(et)}"
        " --batch-size 200 --num-neighbors 80 --common-neighbors-look-forward 3 --tail-num 20000"
    )

OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"写出 {len(rows)} 行 -> {OUT}")
