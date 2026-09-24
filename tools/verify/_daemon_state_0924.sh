#!/bin/bash
cd /home/fedsa/DyGLib
echo "== daemon 进程 =="
ps aux | grep -E "queue|daemon" | grep -v grep | head -5
echo "== daemon 日志尾部 =="
ls tools/queue/*.log tools/queue/daemon.nohup 2>/dev/null
tail -12 tools/queue/daemon.nohup 2>/dev/null || tail -12 tools/queue/daemon.log 2>/dev/null
echo "== nvidia-smi 概览 =="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "== 运行中 train_ 进程（gpu 参数） =="
ps aux | grep "train_" | grep -v grep | sed -E "s/.*(train_[a-z_0-9]+\.py).*--gpu ([0-9]+).*--seeds ([0-9 ]+).*(--eval-ckpt-name [^ ]+)?/\1 gpu=\2 seeds=\3/" | head -12
