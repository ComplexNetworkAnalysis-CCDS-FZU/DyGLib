# -*- coding: utf-8 -*-
"""网格候选点 5 种子确认批行生成（Paper a43d，2026-09-28 下单/用户已批"批"）。

范围（5 点 × 5 种子 = 25 runs；对照=同代际 Full 同种子配对，已有文件）：
  1. linksign RT 15/3（当前 60/1）
  2. linksign RB 60/1（当前 80/3）
  3. sign RT 60/3（当前 100/1）
  4. sign RB 40/1（当前 60/1）
  5. sign WV 15/10（当前 40/15）
统一加 `--grid-confirm-g2`（仅命名；结果名带 .G2，防与屏幕批 seed42 件混算）。
行格式与网格批（insert_stage2_grid_20260925.txt）逐字对齐（除 G2 标记与种子数）。
输出：tools/queue/grid_confirm_20260929.txt（5 行）
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools/queue/grid_confirm_20260929.txt"

TAIL = {"WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"}
SEEDS5 = "42 123 456 789 1024"
PREFIX = ("@cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
          "&& conda activate gc && python ")


def tail(ds: str) -> str:
    return " --tail-num 20000" if ds in TAIL else ""


def linksign_row(ds: str, nn: int, lf: int) -> str:
    return (
        PREFIX + "train_sign_link_3class_prediction.py"
        + f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds {SEEDS5}"
        + " --early-stop-notice f1_wt f1_mic ap f1_mac auc"
        + " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
        + " --grid-confirm-g2"
        + f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail(ds)}"
    )


def sign_row(ds: str, nn: int, lf: int) -> str:
    return (
        PREFIX + "train_link_sign_prediction.py"
        + f" --dataset-name {ds} --model SignDyGFormer --gpu @GPU@ --seeds {SEEDS5}"
        + " --early-stop-notice f1_binary auc f1_weighted"
        + " --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
        + " --grid-confirm-g2"
        + f" --batch-size 200 --num-neighbors {nn} --common-neighbors-look-forward {lf}{tail(ds)}"
    )


rows = [
    linksign_row("RedditHyperlinkTitle", 15, 3),   # 1. linksign RT 15/3
    linksign_row("RedditHyperlinkBody", 60, 1),    # 2. linksign RB 60/1
    sign_row("RedditHyperlinkTitle", 60, 3),       # 3. sign RT 60/3
    sign_row("RedditHyperlinkBody", 40, 1),        # 4. sign RB 40/1
    sign_row("WikiVote", 15, 10),                  # 5. sign WV 15/10
]

OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"写出 {len(rows)} 行 -> {OUT}")
for r in rows:
    print("  ", r[:130], "...")
