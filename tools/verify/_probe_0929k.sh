#!/bin/bash
cd ~/DyGLib || exit 1
date
echo '=== queue ==='
echo "tasks=$(wc -l < tools/queue/tasks.txt) running=$(wc -l < tools/queue/running.txt)"
tail -8 tools/queue/queue.log | cut -c1-130
echo '=== FX files (EVT.FX) ==='
ls saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/ | grep -c 'EVT.FX' || true
ls -l --time-style=+%m-%d_%H:%M saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/*EVT.FX* 2>/dev/null | awk '{print $6, $7, $8}'
echo '=== G2 files (confirm batch) ==='
for task in SignLinkPrediction LinkSign; do
  echo "-- $task"
  find saved_results/$task -name '*P1.TE.G2*.json' -newermt '2026-09-29' 2>/dev/null | head -10
done
echo '=== mamba outputs ==='
ls -l --time-style=+%m-%d_%H:%M /home/fedsa/DynamiSE_DySDGNN_repro/outputs/DyG-Mamba/ 2>/dev/null
echo '=== GPU ==='
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
