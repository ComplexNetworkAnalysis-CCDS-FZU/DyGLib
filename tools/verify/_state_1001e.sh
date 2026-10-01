#!/bin/bash
cd ~/DyGLib
date "+%m-%d %H:%M"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
wc -l < tools/queue/tasks.txt; wc -l < tools/queue/running.txt
tail -4 tools/queue/queue.log | cut -c1-110
echo "== C1/C2 产物 =="
ls -l --time-style=+%H:%M ~/DynamiSE_DySDGNN_repro/outputs/DySDGNN_visibility/ 2>/dev/null | tail -6
ls ~/DynamiSE_DySDGNN_repro/outputs/_smoke/DySDGNN_visibility_gpu/ 2>/dev/null
echo "== task 729/730 日志尾 =="
for i in 729 730 731; do echo "--- $i: $(wc -c < tools/queue/logs/task_$i.log 2>/dev/null)"; tail -2 tools/queue/logs/task_$i.log 2>/dev/null | cut -c1-140; done
