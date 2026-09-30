# -*- coding: utf-8 -*-
"""AB/CNS 失败行重排（2026-09-30）：B + CNS×3 加"显存就绪护栏"后重新入队。

背景：9-30 03:05–03:19，#700(B)/#701/#702/#703(CNS×3) 全部 CUDA OOM（GPU 8GB，
守护阈值 500MiB；#700 首 batch 崩、#701–703 评测阶段崩）——疑派发竞态/残存显存。
护栏：先等目标 GPU 显存 used ≤ 800MiB（最多 5 分钟），再 sleep 45s 沉降，然后执行原命令。
其余 flags 与训练行逐字一致（accel 默认开启，与对照件同代）。
输出：tools/queue/ab_cns_requeue_20260930.txt（4 行：B → CNS RT → CNS RB → CNS sign RT）
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools/queue/ab_cns_requeue_20260930.txt"

GUARD = ('@for _i in 1 2 3 4 5 6 7 8 9 10; do '
         '_u=$(nvidia-smi -i @GPU@ --query-gpu=memory.used --format=csv,noheader,nounits); '
         '[ "${_u:-9999}" -le 800 ] && break; sleep 30; done; sleep 45; ')
PREFIX = (GUARD + "cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
          "&& conda activate gc && python ")
SEEDS5 = "42 123 456 789 1024"
LINK_S = "--early-stop-notice f1_wt f1_mic ap f1_mac auc --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"
SIGN_S = "--early-stop-notice f1_binary auc f1_weighted --module-repeat-aware-sampler --module-repeat-aware-sign-encoder"

rows = [
    # B：采纳配置 + 删 BTE（RB 60/1 + G2）
    PREFIX + "train_sign_link_3class_prediction.py"
    " --dataset-name RedditHyperlinkBody --model SignDyGFormer --gpu @GPU@"
    f" --seeds {SEEDS5} {LINK_S} --no-module-balance-theory-encoder --grid-confirm-g2"
    " --batch-size 200 --num-neighbors 60 --common-neighbors-look-forward 1 --tail-num 20000",
    # CNS RT 15/3
    PREFIX + "train_sign_link_3class_prediction.py"
    " --dataset-name RedditHyperlinkTitle --model SignDyGFormer --gpu @GPU@"
    f" --seeds {SEEDS5} {LINK_S} --no-module-common-neighbor-aware-sampler --grid-confirm-g2"
    " --batch-size 200 --num-neighbors 15 --common-neighbors-look-forward 3 --tail-num 20000",
    # CNS RB 60/1
    PREFIX + "train_sign_link_3class_prediction.py"
    " --dataset-name RedditHyperlinkBody --model SignDyGFormer --gpu @GPU@"
    f" --seeds {SEEDS5} {LINK_S} --no-module-common-neighbor-aware-sampler --grid-confirm-g2"
    " --batch-size 200 --num-neighbors 60 --common-neighbors-look-forward 1 --tail-num 20000",
    # CNS sign RT 60/3
    PREFIX + "train_link_sign_prediction.py"
    " --dataset-name RedditHyperlinkTitle --model SignDyGFormer --gpu @GPU@"
    f" --seeds {SEEDS5} {SIGN_S} --no-module-common-neighbor-aware-sampler --grid-confirm-g2"
    " --batch-size 200 --num-neighbors 60 --common-neighbors-look-forward 3 --tail-num 20000",
]

OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"写出 {len(rows)} 行 -> {OUT}")
