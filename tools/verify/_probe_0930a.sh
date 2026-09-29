#!/bin/bash
cd ~/DyGLib || exit 1
date
echo '=== queue ==='
echo "tasks=$(wc -l < tools/queue/tasks.txt) running=$(wc -l < tools/queue/running.txt)"
tail -10 tools/queue/queue.log | cut -c1-125
echo '=== A/B files (BTE-D) ==='
ls -l --time-style=+%m-%d_%H:%M saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/ 2>/dev/null | grep 'BTE-D' | awk '{print $6, $7}'
echo '=== CNS files (CNAS-D) ==='
find saved_results -name '*CNAS-D*P1.TE.G2.json' ! -name '*profiler*' -newermt '2026-09-29 21:00' 2>/dev/null | sort
echo '=== mamba outputs ==='
ls -l --time-style=+%m-%d_%H:%M /home/fedsa/DynamiSE_DySDGNN_repro/outputs/DyG-Mamba/ 2>/dev/null | tail -12
echo '=== GPU ==='
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
