#!/bin/bash
cd /home/fedsa/DyGLib
echo "== 当前时间 =="; date '+%m-%d %H:%M:%S'
echo "== 运行中训练进程 =="
ps -u fedsa -o pid,etime,cmd | grep -E "train_sign|train_link|run_experiments" | grep -v grep | cut -c1-160
echo "== task_322 日志尾部 =="
tail -20 tools/queue/logs/task_322.log 2>/dev/null
echo "== GPU =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "== running.txt 行数 =="
wc -l tools/queue/running.txt
echo "== 最后完成 npz 时间 =="
ls -lt results/samples_dump/*/*.npz 2>/dev/null | head -3
