# -*- coding: utf-8 -*-
"""生成 2026-09-24 插批：G1-EVT(30) + T16 combo(1) + combo-EVT(5) + signB-EVT(25)。

thr 来源：results/thr_from_ckpt.json（full 侧 ckpt 记录）；
WV/RT/RB 的 linksign seed42 full ckpt 缺失 → 用 RLF 对角等价记录（已验证逐位一致）。
输出：tools/queue/insert_evt2_combo_20260924.txt
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THR = json.load(open(ROOT / "results/thr_from_ckpt.json", encoding="utf-8"))
OUT = ROOT / "tools/queue/insert_evt2_combo_20260924.txt"

# RLF 对角（k_r=k_c）等价 full 的 seed42 阈值（WV/RT/RB linksign）
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
# 1) T16 combo 训练行（先行；RB linksign NN-80/LF-3 + recent-block 80 + G1）
rows.append(
    PREFIX + "train_sign_link_3class_prediction.py"
    " --dataset-name RedditHyperlinkBody --model SignDyGFormer --gpu @GPU@ --seeds 42 123 456 789 1024"
    " --early-stop-notice f1_wt f1_mic ap f1_mac auc"
    " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
    " --recent-block 80 --module-bte-evidence-gate"
    " --batch-size 200 --num-neighbors 80 --common-neighbors-look-forward 3 --tail-num 20000"
)

# 2) G1 固定阈值对照（linksign 25 + sign 5）
for ds, (nn, lf) in LINK.items():
    for s in SEEDS:
        st, et = link_thr(ds, s)
        base = f"SignDyGFormer_seed{s}.NN-{nn}.LF-{lf}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE"
        rows.append(
            PREFIX + "train_sign_link_3class_prediction.py"
            f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds {s}"
            " --early-stop-notice f1_wt f1_mic ap f1_mac auc"
            " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder --cnas-tail-fill"
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
        " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder --cnas-tail-fill"
        f" --module-bte-evidence-gate --eval-only --eval-ckpt-name {base}.G1"
        f" --test-thr {repr(st)}"
        f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail(ds)}"
    )

# 3) T16 组合固定阈值伴行（5 行，逐 seed 装载组合 ckpt）
for s in SEEDS:
    st, et = link_thr("RedditHyperlinkBody", s)
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

# 4) sign 扩批固定阈值对照（25 行，逐 (ds,seed) 装载 TF-E ckpt）
for ds, (nn, lf) in SIGN.items():
    for s in SEEDS:
        th = sign_thr(ds, s)
        base = f"SignDyGFormer_seed{s}.NN-{nn}.LF-{lf}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE"
        rows.append(
            PREFIX + "train_link_sign_prediction.py"
            f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds {s}"
            " --early-stop-notice f1_binary auc f1_weighted"
            " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder --cnas-tail-fill"
            f" --eval-only --eval-ckpt-name {base}.TF-E"
            f" --test-thr {repr(th)}"
            f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail(ds)}"
        )

OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"写出 {len(rows)} 行 -> {OUT}")

# 5) 网格替补行（沿用原 251/252 命令；放最后，插批后恢复网格）
grid_rows = [
    "-s linksign -t parameter -m SignDyGFormer -r RedditHyperlinkTitle RedditHyperlinkBody",
    "-s linksign -t parameter -m SignDyGFormer -r BitcoinAlpha BitcoinOTC WikiVote",
]
OUT.write_text("\n".join(rows + grid_rows) + "\n", encoding="utf-8")
print(f"含网格替补共 {len(rows) + len(grid_rows)} 行")
