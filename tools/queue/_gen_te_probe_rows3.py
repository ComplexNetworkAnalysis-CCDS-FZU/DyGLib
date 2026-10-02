"""_gen_te_probe_rows3.py — 三策略批次行（带**等待式** smoke 守卫；2026-10-03 00:15）。

背景：`#746/#747` 立即式守卫（`[ -f ] || exit 1`）在 smoke 未落盘时按设计提前退出、把行消耗掉。
本版改为**等待式守卫**：最多等 90 min（每 60 s 检查一次），smoke 两产物齐才开跑；超时也开跑（留痕）。
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / "te_probe3_1003.txt"

RES = ("/home/fedsa/DyGLib/saved_results/LinkSign/SignDyGFormer/BitcoinAlpha/"
       "SignDyGFormer_seed42.NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.{tag}.json")
WAIT = (f'for _i in $(seq 1 90); do [ -f "{RES.format(tag="TD")}" ] && [ -f "{RES.format(tag="TD-LIN")}" ] '
        '&& break; sleep 60; done; ')

PREFIX = "cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh && conda activate gc && "
BASE = ("python train_link_sign_prediction.py --dataset-name {ds} --model SignDyGFormer "
        "--gpu @GPU@ --seeds $s --early-stop-notice f1_binary auc f1_weighted "
        "--module-repeat-aware-sampler --module-repeat-aware-sign-encoder "
        "--batch-size 200 --num-neighbors 40 --common-neighbors-look-forward 15{extra}{decay}")
LIN = " --time-decay-lambda 0.1 --time-decay-form LINEAR"


def run(ds: str, extra: str, decay: str) -> str:
    return BASE.format(ds=ds, extra=extra, decay=decay)


def block(ds: str, extra: str, tag: str) -> str:
    return (f"for s in 42 123 456 789 1024; do "
            f"{run(ds, extra, '')} || echo \"TE_FAIL {ds} $s\"; "
            f"{run(ds, extra, ' --time-decay-lambda 0.1')} || echo \"TD_FAIL {ds} $s\"; "
            f"{run(ds, extra, ' --time-decay-lambda 0.1' + LIN)} || echo \"LIN_FAIL {ds} $s\"; "
            f"done; echo {tag}")


rows = [
    "@" + PREFIX + WAIT + block("BitcoinAlpha", "", "TE5_A_OK") + "; " + block("BitcoinOTC", "", "TE5_B_OK"),
    "@" + PREFIX + WAIT + block("WikiVote", " --tail-num 20000", "TE5_C_OK"),
]
OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
for i, r in enumerate(rows, 1):
    print(f"--- row{i} ({len(r)} chars): {r[:120]} ... {r[-60:]}")
print(f"[ok] {OUT}")
