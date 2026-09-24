#!/bin/bash
# 2026-09-24: 终止 GPU1 上的网格 run_experiments(pid 待定) 进程树，为 D3b 转储行让路
cd /home/fedsa/DyGLib
PID=$(pgrep -f "run_experiments.py -s linksign -t parameter" | head -1)
echo "== 目标 run_experiments pid: $PID"
if [ -z "$PID" ]; then echo "无运行中的参数网格"; exit 0; fi
echo "== 进程树（直接子进程） =="
pgrep -P "$PID" -a
echo "== 杀子进程 =="
for c in $(pgrep -P "$PID"); do kill -TERM "$c" 2>/dev/null && echo "killed child $c"; done
echo "== 杀父进程 =="
kill -TERM "$PID" 2>/dev/null && echo "killed parent $PID"
for i in $(seq 1 15); do
  kill -0 "$PID" 2>/dev/null || break
  sleep 1
done
kill -0 "$PID" 2>/dev/null && { echo "父进程仍存活，强杀"; kill -9 "$PID" 2>/dev/null; }
echo "== 残余 GPU1 训练进程（应为空） =="
ps aux | grep "train_" | grep -v grep | grep -v "gpu 0" | head -5
echo "== GPU 状态 =="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "== GPU0 组合训练仍在（应为 1 条） =="
ps aux | grep "recent-block 80" | grep -v grep | wc -l
