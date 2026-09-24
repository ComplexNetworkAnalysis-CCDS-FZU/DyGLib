# -*- coding: utf-8 -*-
"""修正批：G1 固定阈值对照（去掉 --cnas-tail-fill 的代际混用）。

背景（2026-09-24 深夜核查）：原 G1-EVT 批（254–283）命令行含 `--cnas-tail-fill`
但装载非 TF 代 `.G1` ckpt（如 WV auc 0.9673 vs 训练批 0.9615、RB 0.9327 vs 0.9304）
⇒ 特征代际混用，判定失真。本批 = 同 ckpt、同固定阈值、**无 TF**（与 G1 训练批
及 full 基线（main_tables）同代），仅做阈值口径对照。
输出：tools/queue/insert_g1evt_fix_20260924.txt（30 行）
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THR = json.load(open(ROOT / "results/thr_from_ckpt.json", encoding="utf-8"))
OUT = ROOT / "tools/queue/insert_g1evt_fix_20260924.txt"

RLF_SEED42 = {
    "WikiVote": (0.61, 0.01),
    "RedditHyperlinkTitle": (0.53, 0.01),
    "RedditHyperlinkBody": (0.48, 0.01),
}

LINK = {"WikiVote": (15, 10), "RedditHyperlinkTitle": (60, 1), "RedditHyperlinkBody": (80, 3),
        "BitcoinAlpha": (40, 15), "BitcoinOTC": (80, 5)}
SIGN = {"WikiVote": (40, 15), "RedditHyperlinkTitle": (100, 1), "RedditHyperlinkBody": (60, 1),
        "BitcoinAlpha": (40, 15), "BitcoinOTC": (40, 15)}
TAIL = {"WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"}
SEEDS = [42, 123, 456, 789, 1024]
PREFIX = ("@cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
          "&& conda activate gc && python ")


def tail(ds: str) -> str:
    return " --tail-num 20000" if ds in TAIL else ""


def link_thr(ds: str, s: int):
    key = f"linksign|{ds}|{s}|full"
    if key in THR:
        return float(THR[key]["best_sign_thr"]), float(THR[key]["best_exist_thr"])
    assert s == 42 and ds in RLF_SEED42, f"missing thr: {key}"
    return RLF_SEED42[ds]


def sign_thr(ds: str, s: int) -> float:
    return float(THR[f"sign|{ds}|{s}|full"]["thr"])


rows = []
for ds, (nn, lf) in LINK.items():
    for s in SEEDS:
        st, et = link_thr(ds, s)
        base = f"SignDyGFormer_seed{s}.NN-{nn}.LF-{lf}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE"
        rows.append(
            PREFIX + "train_sign_link_3class_prediction.py"
            f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds {s}"
            " --early-stop-notice f1_wt f1_mic ap f1_mac auc"
            " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
            f" --module-bte-evidence-gate --eval-only --eval-ckpt-name {base}.G1"
            f" --sign-thr {repr(st)} --exist-thr {repr(et)}"
            f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail(ds)}"
        )
for ds, (nn, lf) in SIGN.items():
    st = sign_thr(ds, 42)
    base = f"SignDyGFormer_seed42.NN-{nn}.LF-{lf}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE"
    rows.append(
        PREFIX + "train_link_sign_prediction.py"
        f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds 42"
        " --early-stop-notice f1_binary auc f1_weighted"
        " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
        f" --module-bte-evidence-gate --eval-only --eval-ckpt-name {base}.G1"
        f" --test-thr {repr(st)}"
        f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail(ds)}"
    )

OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"写出 {len(rows)} 行 -> {OUT}")
