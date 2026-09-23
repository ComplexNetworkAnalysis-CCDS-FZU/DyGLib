# -*- coding: utf-8 -*-
"""生成"固定阈值评测（eval-only）"队列行：E1a(`.TF-E`) checkpoint × full 侧记录的阈值。

输入：results/thr_from_ckpt.json（服务器 checkpoint param.json 采集；key=task|ds|seed|full|TF-E）
输出：tools/queue/insert_evt_20260923.txt（仅命令行行；linksign=3class、sign=binary）
说明：WV/RT/RB 的 linksign full seed42 检查点缺失（历史遗留）→ 该 3×1 点跳过（4 种子配对）。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THR = json.load(open(ROOT / "results/thr_from_ckpt.json", encoding="utf-8"))
OUT = ROOT / "tools/queue/insert_evt_20260923.txt"

LINK_PARAMS = {"WikiVote": (15, 10), "RedditHyperlinkTitle": (60, 1), "RedditHyperlinkBody": (80, 3),
               "BitcoinAlpha": (40, 15), "BitcoinOTC": (80, 5)}
SIGN_PARAMS = {"WikiVote": (40, 15), "RedditHyperlinkTitle": (100, 1), "RedditHyperlinkBody": (60, 1),
               "BitcoinAlpha": (40, 15), "BitcoinOTC": (40, 15)}
TAIL = {"WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"}
PREFIX = ("@cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
          "&& conda activate gc && python ")
SEEDS = [42, 123, 456, 789, 1024]

rows, skipped = [], []


def num(x) -> str:
    return repr(float(x))


# ---- linksign（3class）----
for ds, (nn, lf) in LINK_PARAMS.items():
    for s in SEEDS:
        full = f"linksign|{ds}|{s}|full"
        tf = f"linksign|{ds}|{s}|TF-E"
        if full not in THR:
            skipped.append(full)
            continue
        st = THR[full]["best_sign_thr"]
        et = THR[full]["best_exist_thr"]
        base = f"SignDyGFormer_seed{s}.NN-{nn}.LF-{lf}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE"
        tail = " --tail-num 20000" if ds in TAIL else ""
        rows.append(
            PREFIX + "train_sign_link_3class_prediction.py"
            f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds {s}"
            " --early-stop-notice f1_wt f1_mic ap f1_mac auc"
            " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder --cnas-tail-fill"
            f" --eval-only --eval-ckpt-name {base}.TF-E"
            f" --sign-thr {num(st)} --exist-thr {num(et)}"
            f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail}"
        )

# ---- sign（binary；seed42）----
for ds, (nn, lf) in SIGN_PARAMS.items():
    full = f"sign|{ds}|42|full"
    if full not in THR:
        skipped.append(full)
        continue
    th = THR[full]["thr"]
    base = f"SignDyGFormer_seed42.NN-{nn}.LF-{lf}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE"
    tail = " --tail-num 20000" if ds in TAIL else ""
    rows.append(
        PREFIX + "train_link_sign_prediction.py"
        f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds 42"
        " --early-stop-notice f1_binary auc f1_weighted"
        " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder --cnas-tail-fill"
        f" --eval-only --eval-ckpt-name {base}.TF-E"
        f" --test-thr {num(th)}"
        f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail}"
    )

OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"写出 {len(rows)} 行 -> {OUT}")
if skipped:
    print("跳过（full 检查点/阈值缺失）：")
    for k in skipped:
        print("  ", k)
