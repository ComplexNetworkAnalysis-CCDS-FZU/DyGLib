#!/bin/bash
cd ~/DyGLib || exit 1
date
echo '=== mamba test status ==='
ls -la /tmp/mamba_wrap_test.log
ps -o pid,etime,stat,cmd -p 616212 2>/dev/null | tail -1 | cut -c1-120
echo '--- outputs/DyG-Mamba ---'
ls -la /home/fedsa/DynamiSE_DySDGNN_repro/outputs/DyG-Mamba/ 2>/dev/null
echo '=== RB combo progress ==='
ls saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/ | grep -c 'TF-E.RK-80' || true
ls -l --time-style=+%m-%d_%H:%M saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/*TF-E.RK-80* 2>/dev/null | awk '{print $6, $7}'
echo '=== queue log tail ==='
tail -5 tools/queue/queue.log
echo '=== running.txt / GPU ==='
wc -l < tools/queue/running.txt
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
