# -*- coding: utf-8 -*-
"""网格确认批（G2）linksign 两点 FX 固定阈值伴行行生成（2026-09-29）。

阈值来源（服务器 param.json 采集）：各 seed 的 best_sign_thr / best_exist_thr。
行 = 训练行同 flags + `--grid-confirm-g2 --eval-only --eval-ckpt-name …G2 --sign-thr --exist-thr`
→ 结果名 `…P1.TE.G2.EVT.FX.json`。
输出：tools/queue/gconfirm_fx_20260929.txt（10 行 = 2 点 × 5 种子）
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools/queue/gconfirm_fx_20260929.txt"

PREFIX = ("@cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
          "&& conda activate gc && python ")

# (ds, nn, lf, {seed: (sign_thr, exist_thr)})
POINTS = {
    ("RedditHyperlinkTitle", 15, 3): {
        42: (0.47000000000000003, 0.08),
        123: (0.5700000000000001, 0.01),
        456: (0.49, 0.13),
        789: (0.56, 0.12),
        1024: (0.48000000000000004, 0.13),
    },
    ("RedditHyperlinkBody", 60, 1): {
        42: (0.49, 0.01),
        123: (0.52, 0.01),
        456: (0.5800000000000001, 0.03),
        789: (0.5, 0.01),
        1024: (0.45, 0.01),
    },
}

rows = []
for (ds, nn, lf), thr_map in POINTS.items():
    for s, (st, et) in thr_map.items():
        base = (f"SignDyGFormer_seed{s}.NN-{nn}.LF-{lf}"
                f".RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.G2")
        rows.append(
            PREFIX + "train_sign_link_3class_prediction.py"
            f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds {s}"
            " --early-stop-notice f1_wt f1_mic ap f1_mac auc"
            " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
            " --grid-confirm-g2"
            f" --eval-only --eval-ckpt-name {base}"
            f" --sign-thr {repr(st)} --exist-thr {repr(et)}"
            f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf} --tail-num 20000"
        )

OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"写出 {len(rows)} 行 -> {OUT}")
