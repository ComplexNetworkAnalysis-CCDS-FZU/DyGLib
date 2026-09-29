#!/bin/bash
cd ~/DyGLib || exit 1
date
echo '=== RB combo ==='
ls saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/ | grep -c 'TF-E.RK-80' || true
ls -l --time-style=+%m-%d_%H:%M saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/*TF-E.RK-80* 2>/dev/null | awk '{print $6, $7}'
echo '=== mamba test / outputs ==='
ps -o pid,etime,stat -p 616212 2>/dev/null | tail -1
ls -la /tmp/mamba_wrap_test.log
ls -la /home/fedsa/DynamiSE_DySDGNN_repro/outputs/DyG-Mamba/ 2>/dev/null
echo '=== queue ==='
echo "tasks=$(wc -l < tools/queue/tasks.txt) running=$(wc -l < tools/queue/running.txt)"
tail -4 tools/queue/queue.log
echo '=== GPU ==='
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
