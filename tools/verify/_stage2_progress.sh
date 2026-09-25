#!/bin/bash
cd /home/fedsa/DyGLib
echo "== #422 (B2-RB 5seeds) 日志尾部 =="
tail -4 tools/queue/logs/task_422.log | cut -c1-140
grep -c "^Epoch: 100" tools/queue/logs/task_422.log 2>/dev/null || true
grep -oE "train for the [0-9]+-th batch" tools/queue/logs/task_422.log | tail -1
echo "== #423 (B2-WV 5seeds) 日志尾部 =="
tail -4 tools/queue/logs/task_423.log | cut -c1-140
grep -oE "train for the [0-9]+-th batch" tools/queue/logs/task_423.log | tail -1
echo "== 已产出 B2 种子（重跑覆盖检测） =="
ls -lt saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/*.B2.json 2>/dev/null | head -5
ls -lt saved_results/SignLinkPrediction/SignDyGFormer/WikiVote/*.B2.json 2>/dev/null | head -5
echo "== GPU =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
