#!/bin/bash
cd ~/DyGLib
echo "== 时间/GPU =="
date "+%m-%d %H:%M"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "== 队列 =="
wc -l < tools/queue/tasks.txt; wc -l < tools/queue/running.txt
tail -5 tools/queue/queue.log | cut -c1-120
echo "== 进程 =="
ps -ef | grep -E "m5_run|train_sign|run_experiments" | grep -v grep | cut -c1-140
echo "== C1/C2 产物 =="
ls -l --time-style=+%m-%d_%H:%M ~/DynamiSE_DySDGNN_repro/outputs/DySDGNN_visibility/ 2>/dev/null | tail -18
echo "-- 计数: $(ls ~/DynamiSE_DySDGNN_repro/outputs/DySDGNN_visibility/*_C012.json 2>/dev/null | wc -l)"
echo "== 冒烟目录 =="
ls ~/DynamiSE_DySDGNN_repro/outputs/_smoke/DySDGNN_visibility/ 2>/dev/null
echo "== 任务日志 723-725 =="
for i in 723 724 725; do
  f=tools/queue/logs/task_$i.log
  echo "--- task_$i: $(wc -c <"$f" 2>/dev/null) bytes; 尾 3 行:"
  tail -3 "$f" 2>/dev/null | cut -c1-160
done
echo "== wave2/mamba =="
ls ~/DynamiSE_DySDGNN_repro/outputs/DyG-Mamba/*.json 2>/dev/null | wc -l
ls ~/DynamiSE_DySDGNN_repro/outputs/ScaDyG/*.json 2>/dev/null | wc -l
