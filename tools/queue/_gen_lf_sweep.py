# -*- coding: utf-8 -*-
"""生成「固定 NN、单变量扫 LF」30 行队列（2026-09-25，用户设想：网格瘦身）。

设计：NN 固定为各数据集现行 best（linksign: WV15/RT60/RB80；sign: WV40/RT100/RB60），
      m=0（默认），LF ∈ {1,3,5,10,15}；RT/RB/WV × {linksign, sign} = 30 runs（单种子 42）。
顺序：RT×2任务 → RB×2任务 → WV×2任务（首块 RT 10 runs 最早出曲线）。
输出：tools/queue/lf_sweep_30_20260925.txt
"""
from pathlib import Path

HEAD = (
    "@cd /home/fedsa/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
    "&& conda activate gc && python {script} --dataset-name {ds} --model SignDyGFormer "
    "--gpu @GPU@ --seeds 42 --early-stop-notice {notice} --module-repeat-aware-sampler "
    "--module-repeat-aware-sign-encoder --batch-size 200 --num-neighbors {nn} "
    "--common-neighbors-look-forward {lf} --tail-num 20000"
)

LINK = dict(script="train_sign_link_3class_prediction.py", notice="f1_wt f1_mic ap f1_mac auc")
SIGN = dict(script="train_link_sign_prediction.py", notice="f1_binary auc f1_weighted")

# (task, dataset, NN) 顺序 = 输出顺序
cells = [
    (LINK, "RedditHyperlinkTitle", 60),
    (SIGN, "RedditHyperlinkTitle", 100),
    (LINK, "RedditHyperlinkBody", 80),
    (SIGN, "RedditHyperlinkBody", 60),
    (LINK, "WikiVote", 15),
    (SIGN, "WikiVote", 40),
]
LFS = (1, 3, 5, 10, 15)

lines = []
for task, ds, nn in cells:
    for lf in LFS:
        lines.append(HEAD.format(ds=ds, nn=nn, lf=lf, **task))

out = Path(__file__).parent / "lf_sweep_30_20260925.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"写出 {len(lines)} 行 → {out}")
for i in (0, 4, 5, 9, 10, 29):
    print(f"{i + 1}: ...{lines[i][-120:]}")
assert len(lines) == 30
