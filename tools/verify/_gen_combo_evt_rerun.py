# -*- coding: utf-8 -*-
"""T16 组合固定阈值伴行补跑（RB；seeds 123/456/789/1024；seed42 已在批）。

原因：原伴行行（284–288）在组合训练完成前即派发，除 seed42 外 4 行加载 ckpt 失败。
输出：tools/queue/insert_combo_evt_rerun_20260925.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THR = json.load(open(ROOT / "results/thr_from_ckpt.json", encoding="utf-8"))
OUT = ROOT / "tools/queue/insert_combo_evt_rerun_20260925.txt"

PREFIX = ("@cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
          "&& conda activate gc && python ")

# 默认 123/456/789/1024（4 行）；`--with-42` 追加 seed42（竞态污染补跑）
SEEDS = [123, 456, 789, 1024]
if "--with-42" in sys.argv:
    SEEDS = [42] + SEEDS

rows = []
for s in SEEDS:
    key = f"linksign|RedditHyperlinkBody|{s}|full"
    st = float(THR[key]["best_sign_thr"])
    et = float(THR[key]["best_exist_thr"])
    base = f"SignDyGFormer_seed{s}.NN-80.LF-3.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE"
    rows.append(
        PREFIX + "train_sign_link_3class_prediction.py"
        " --dataset-name RedditHyperlinkBody --model SignDyGFormer --gpu @GPU@"
        f" --seeds {s}"
        " --early-stop-notice f1_wt f1_mic ap f1_mac auc"
        " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
        " --recent-block 80 --module-bte-evidence-gate"
        f" --eval-only --eval-ckpt-name {base}.RK-80.G1"
        f" --sign-thr {repr(st)} --exist-thr {repr(et)}"
        " --batch-size 200 --num-neighbors 80 --common-neighbors-look-forward 3 --tail-num 20000"
    )

OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"写出 {len(rows)} 行 -> {OUT}")
