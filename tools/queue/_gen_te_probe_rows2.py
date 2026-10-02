"""_gen_te_probe_rows2.py — 生成 §4.5 三策略队列行（修复内核签名后重排；2026-10-02 18:30）。

结构（3 行）：
  #smoke : BA seed42 跑 **TD**（exp）与 **TD-LIN**（linear）各一次 —— 二者文件名与主归档不同，不会覆盖；
           同时验证 ①exp 臂在内核路径可用 ②linear 臂自动回落纯 Python 路径可用。
  #batchA: BA + OTC × 3 策略 × 5 种子（30 runs）
  #batchB: WV × 3 策略 × 5 种子（15 runs）
  两个 batch 行**带 guard**：smoke 的两个产物不存在则直接退出（不白跑）。
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / "te_probe2_1002.txt"

RES = ("/home/fedsa/DyGLib/saved_results/LinkSign/SignDyGFormer/BitcoinAlpha/"
       "SignDyGFormer_seed42.NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.{tag}.json")
GUARD = (f'[ -f "{RES.format(tag="TD")}" ] && [ -f "{RES.format(tag="TD-LIN")}" ] || exit 1; ')

PREFIX = "cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh && conda activate gc && "
BASE = ("python train_link_sign_prediction.py --dataset-name {ds} --model SignDyGFormer "
        "--gpu @GPU@ --seeds $s --early-stop-notice f1_binary auc f1_weighted "
        "--module-repeat-aware-sampler --module-repeat-aware-sign-encoder "
        "--batch-size 200 --num-neighbors 40 --common-neighbors-look-forward 15{extra}{decay}")


LINEAR_ARG = " --time-decay-form LINEAR"


def run(ds: str, extra: str, decay: str) -> str:
    return BASE.format(ds=ds, extra=extra, decay=decay)


def block(ds: str, extra: str, tag: str) -> str:
    return (f"for s in 42 123 456 789 1024; do "
            f"{run(ds, extra, '')} || echo \"TE_FAIL {ds} $s\"; "
            f"{run(ds, extra, ' --time-decay-lambda 0.1')} || echo \"TD_FAIL {ds} $s\"; "
            f"{run(ds, extra, ' --time-decay-lambda 0.1' + LINEAR_ARG)} || echo \"LIN_FAIL {ds} $s\"; "
            f"done; echo {tag}")


smoke = ("@" + PREFIX
         + f"{run('BitcoinAlpha', '', ' --time-decay-lambda 0.1').replace('--seeds $s', '--seeds 42')} || echo TD_SMOKE_FAIL; "
         + f"{run('BitcoinAlpha', '', ' --time-decay-lambda 0.1' + LINEAR_ARG).replace('--seeds $s', '--seeds 42')} || echo LIN_SMOKE_FAIL; "
         + "echo TE_SMOKE_DONE")

rows = [
    smoke,
    "@" + PREFIX + GUARD + block("BitcoinAlpha", "", "TE5_A_OK") + "; " + block("BitcoinOTC", "", "TE5_B_OK"),
    "@" + PREFIX + GUARD + block("WikiVote", " --tail-num 20000", "TE5_C_OK"),
]
OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
for i, r in enumerate(rows, 1):
    print(f"--- row{i} ({len(r)} chars): {r[:110]} ... {r[-70:]}")
print(f"[ok] {OUT}（{len(rows)} 行）")
