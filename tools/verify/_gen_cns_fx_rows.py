# -*- coding: utf-8 -*-
"""CNS 行 FX 固定阈值伴行行生成（f8c2 §三-2；linksign RT/RB 的 CNS 行）。

阈值来源：CNS 行各 seed param.json（服务器采集，写入 results/cns_thr.json；结构
{"RedditHyperlinkTitle|15|3|42": {"best_sign_thr": x, "best_exist_thr": y}, ...}）。
输出：tools/queue/cns_fx_20260930.txt（10 行 = 2 点 × 5 种子）
用法：python tools/verify/_gen_cns_fx_rows.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THR_FILE = ROOT / "results/cns_thr.json"
OUT = ROOT / "tools/queue/cns_fx_20260930.txt"
PREFIX = ("@cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
          "&& conda activate gc && python ")

POINTS = [("RedditHyperlinkTitle", 15, 3), ("RedditHyperlinkBody", 60, 1)]
SEEDS = (42, 123, 456, 789, 1024)

thr = json.load(open(THR_FILE, encoding="utf-8"))
rows = []
for (ds, nn, lf) in POINTS:
    for s in SEEDS:
        key = f"{ds}|{nn}|{lf}|{s}"
        st_, et = float(thr[key]["best_sign_thr"]), float(thr[key]["best_exist_thr"])
        base = (f"SignDyGFormer_seed{s}.NN-{nn}.LF-{lf}"
                f".RAS-E.RASE-E.BTE-E.CNAS-D.P1.TE.G2")
        rows.append(
            PREFIX + "train_sign_link_3class_prediction.py"
            f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds {s}"
            " --early-stop-notice f1_wt f1_mic ap f1_mac auc"
            " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
            " --no-module-common-neighbor-aware-sampler --grid-confirm-g2"
            f" --eval-only --eval-ckpt-name {base}"
            f" --sign-thr {repr(st_)} --exist-thr {repr(et)}"
            f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf} --tail-num 20000"
        )
OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"写出 {len(rows)} 行 -> {OUT}")
