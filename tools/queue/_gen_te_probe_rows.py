"""_gen_te_probe_rows.py — 生成 §4.5 三策略对照（TE / TD / TD-LIN）队列行。

口径（Paper 8bd4 批准；口径 B）：
  3 策略 × {BitcoinAlpha, BitcoinOTC, WikiVote} × 5 种子 = 45 runs（sign 任务，AUC + F1_bin）
  · TE 臂  : 不传 --time-decay-lambda（= cosine 时间编码，主口径）→ 结果名尾 .TE
  · TD 臂  : --time-decay-lambda 0.1（exp(-λ·Δt)；Δt 口径 A=staleness，默认）→ .TD
  · TD-LIN : 同 λ + --time-decay-form linear（max(0,1-γ·Δt)，γ 自动按 Δt 中位数与 EXP 等权）→ .TD-LIN
配置 = 各数据集 sign 采纳点（NN-40 / LF-15；WV 另加 --tail-num 20000），与主口径逐字一致。
行的 @GPU@ 由队列调度器替换；输出 = 结果 JSON（同名 TE 臂会覆盖主归档 ⇒ 已留档 sha256）。
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / "te_probe_1002.txt"

BASE = ("python train_link_sign_prediction.py --dataset-name {ds} --model SignDyGFormer "
        "--gpu @GPU@ --seeds $s --early-stop-notice f1_binary auc f1_weighted "
        "--module-repeat-aware-sampler --module-repeat-aware-sign-encoder "
        "--batch-size 200 --num-neighbors 40 --common-neighbors-look-forward 15{extra}")


def ds_block(ds: str, tail: str, tag: str) -> str:
    te = BASE.format(ds=ds, extra=tail)
    td = BASE.format(ds=ds, extra=tail) + " --time-decay-lambda 0.1"
    lin = BASE.format(ds=ds, extra=tail) + " --time-decay-lambda 0.1 --time-decay-form linear"
    return (f"for s in 42 123 456 789 1024; do "
            f"{te} || echo \"TE_FAIL {ds} $s\"; "
            f"{td} || echo \"TD_FAIL {ds} $s\"; "
            f"{lin} || echo \"LIN_FAIL {ds} $s\"; "
            f"done; echo {tag}")


PREFIX = "cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh && conda activate gc && "
rows = [
    "@" + PREFIX + ds_block("BitcoinAlpha", "", "TE5_A_OK") + "; " + ds_block("BitcoinOTC", "", "TE5_B_OK"),
    "@" + PREFIX + ds_block("WikiVote", " --tail-num 20000", "TE5_C_OK"),
]
OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
for i, r in enumerate(rows, 1):
    print(f"--- row{i} ({len(r)} chars) 头 150: {r[:150]}")
    print(f"    ... 尾 90: {r[-90:]}")
print(f"[ok] {OUT}（{len(rows)} 行；共 3 策略 × 3 数据集 × 5 种子 = 45 runs）")
