#!/bin/bash
cd /home/fedsa/DyGLib
date '+%m-%d %H:%M:%S'
echo "== queue.log 尾部 6 行 =="
tail -6 tools/queue/queue.log
echo "== 训练/重算进程 =="
ps -u fedsa -o pid,etime,cmd | grep -E "train_|bte_sparsity" | grep -v grep | cut -c1-150
echo "== 掩码重算日志尾部 =="
tail -6 results/bte_sparsity/rerun_linksign_20260924.log 2>/dev/null
echo "== GPU =="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
