#!/bin/bash
# 清理所有残留 train_* 进程（守护已 idle；确保网格完全停止后再插批）
cd /home/fedsa/DyGLib
for round in 1 2 3; do
    pids=$(pgrep -f "train_(link_sign|sign_link)")
    if [ -z "$pids" ]; then echo "round $round: clean"; break; fi
    echo "round $round kill: $pids"
    kill $pids 2>/dev/null
    sleep 2
done
echo "=== 最终检查 ==="
pgrep -af "run_experiments|train_" || echo "ALL STOPPED"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
