#!/bin/bash
cd ~/DyGLib
echo "== 时间 =="; date
echo "== tasks.txt 行数 =="; wc -l < tools/queue/tasks.txt
echo "== running.txt =="; cat tools/queue/running.txt 2>/dev/null; echo "-"
echo "== tasks 尾部 8 行 =="; tail -n 8 tools/queue/tasks.txt
echo "== GPU =="; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "== 近 30 分钟新结果文件 =="; find results/expm-2026-09-25-logs -name "*.json" -mmin -30 2>/dev/null | head -20
echo "== B2 各数据集文件数 =="; for d in WikiVote RedditHyperlinkTitle RedditHyperlinkBody BitcoinAlpha BitcoinOTC; do
  n=$(ls results/expm-2026-09-25-logs/train_link_sign_prediction/$d 2>/dev/null | grep -c "B2" || true)
  echo "$d: $n"
done
echo "== B4 各数据集 =="; for d in WikiVote RedditHyperlinkTitle RedditHyperlinkBody BitcoinAlpha BitcoinOTC; do
  n=$(ls results/expm-2026-09-25-logs/train_link_sign_prediction/$d 2>/dev/null | grep -c "B4" || true)
  echo "$d: $n"
done
echo "== B5 =="; ls results/expm-2026-09-25-logs/train_link_sign_prediction/*/ 2>/dev/null | grep -c "B5" || true
