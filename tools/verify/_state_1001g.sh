#!/bin/bash
cd ~/DyGLib
date "+%m-%d %H:%M"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
wc -l < tools/queue/tasks.txt; wc -l < tools/queue/running.txt
tail -6 tools/queue/queue.log | cut -c1-105
echo "== C1/C2 产物 =="
ls -l --time-style=+%H:%M ~/DynamiSE_DySDGNN_repro/outputs/DySDGNN_visibility/ 2>/dev/null | tail -18
echo "-- 计数: $(ls ~/DynamiSE_DySDGNN_repro/outputs/DySDGNN_visibility/*_C012.json 2>/dev/null | wc -l)"
echo "== task 730/731/732 日志尾 =="
for i in 730 731 732; do
  echo "--- $i: $(wc -c < tools/queue/logs/task_$i.log 2>/dev/null) bytes"
  tail -3 tools/queue/logs/task_$i.log 2>/dev/null | cut -c1-150
done
echo "== scadyg 相关脚本 =="
ls ~/DynamiSE_DySDGNN_repro/ext_baselines/scadyg/ 2>/dev/null | head -12
echo "== repro 工作树改动 =="
cd ~/DynamiSE_DySDGNN_repro && git status --porcelain | grep -vE "^ M outputs/|^ D outputs/" | head -8
git log -1 --oneline | cut -c1-50
