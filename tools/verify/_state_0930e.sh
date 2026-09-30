#!/bin/bash
cd ~/DyGLib
echo "== linksign RB 目录内 CNS-D 文件 =="
ls -l --time-style=+%m-%d_%H:%M saved_results/SignLinkPrediction/RedditHyperlinkBody/ 2>/dev/null | grep -E "CNAS-[DE]" | tail -20
echo "== 目标护栏文件逐个判定 =="
for s in 42 123 456 789 1024; do
  f="saved_results/SignLinkPrediction/RedditHyperlinkBody/SignDyGFormer_seed${s}.NN-60.LF-1.RAS-E.RASE-E.BTE-E.CNAS-D.P1.TE.G2.json"
  if [ -f "$f" ]; then echo "  seed$s: EXISTS"; else echo "  seed$s: MISSING"; fi
done
echo "== GPU / 进程 =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ps -ef | grep -E "train_sign_link_3class|train_sign_dygmamba" | grep -v grep | cut -c1-190
echo "== task 719/720 log 大小 =="
ls -l --time-style=+%H:%M tools/queue/logs/task_719.log tools/queue/logs/task_720.log 2>/dev/null
echo "== queue.log 尾 6 =="
tail -6 tools/queue/queue.log | cut -c1-160
