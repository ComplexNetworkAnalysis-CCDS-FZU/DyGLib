#!/bin/bash
cd /home/fedsa/DyGLib
echo "=== 调度尾 ==="
tail -5 tools/queue/queue.log
echo "=== G1 文件计数 ==="
find saved_results -name "*G1.json" | wc -l
echo "=== EVT 文件计数 ==="
find saved_results -name "*EVT*.json" | grep -v profiler | wc -l
echo "=== 其余新档（18:00 后非 G1/EVT）==="
find saved_results -name "*.json" -newermt "2026-09-23 18:00" ! -name "*profiler*" ! -name "*G1*" ! -name "*EVT*" -printf '%TH:%TM %p\n' | sort | tail -8
echo "=== 运行中 ==="
ps aux | grep -E "train_(link_sign|sign_link)" | grep -v grep | awk '{print $2, $10}' | head -4
