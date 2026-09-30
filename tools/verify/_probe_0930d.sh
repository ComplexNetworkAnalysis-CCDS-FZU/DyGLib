#!/bin/bash
cd ~/DyGLib || exit 1
echo '=== GPU 型号/显存总量 ==='
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo
echo '=== queue_daemon.sh（busy/dispatch 逻辑）==='
grep -n "gpulock\|nvidia-smi\|MiB\|free\|dispatch\|sleep" tools/queue/queue_daemon.sh | head -40
echo
echo '=== task_702 尾部（崩溃点）==='
tail -25 tools/queue/logs/task_702.log | cut -c1-180
echo
echo '=== task_703 尾部 ==='
tail -20 tools/queue/logs/task_703.log | cut -c1-180
