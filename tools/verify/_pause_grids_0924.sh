#!/bin/bash
# 暂停网格（先子后父），为插批让路（2026-09-24；Paper 批准"先例：暂停网格→插批→恢复"）
cd /home/fedsa/DyGLib
echo "=== 杀前 ==="
ps -eo pid,ppid,etime,args | grep -E "run_experiments|train_(link_sign|sign_link)" | grep -v grep | awk '{print $1, $2, $3}'
for pp in $(pgrep -f "run_experiments.py -s linksign -t parameter"); do
    for ch in $(pgrep -P "$pp"); do
        kill "$ch" 2>/dev/null && echo "killed child $ch (parent $pp)"
    done
    sleep 1
    kill "$pp" 2>/dev/null && echo "killed parent $pp"
done
sleep 3
echo "=== 杀后 ==="
pgrep -af "run_experiments|train_" || echo "grids stopped, no training procs"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
