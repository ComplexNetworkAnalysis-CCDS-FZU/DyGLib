#!/bin/bash
cd /home/fedsa/DyGLib
echo "=== 调度尾 8 ==="
tail -8 tools/queue/queue.log
echo "=== 运行中 ==="
ps aux | grep -E "train_(link_sign|sign_link)" | grep -v grep | awk '{print $2, $10}' | head -4
echo "=== 计数 ==="
echo -n "G1: "; find saved_results -name "*G1.json" ! -name "*profiler*" | wc -l
echo -n "EVT: "; find saved_results -name "*EVT*.json" ! -name "*profiler*" | wc -l
echo -n "TF-E(linksign): "; find saved_results/SignLinkPrediction -name "*TF-E.json" ! -name "*profiler*" | wc -l
echo -n "TF-E(sign): "; find saved_results/LinkSign -name "*TF-E.json" ! -name "*profiler*" | wc -l
echo -n "RK-80(RB 5seed): "; find saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody -name "*RK-80.json" ! -name "*profiler*" | wc -l
echo "=== 最新 12 结果 ==="
find saved_results -name "*.json" ! -name "*profiler*" -newermt "2026-09-23 18:00" -printf '%TH:%TM %p\n' | sort | tail -12
echo "=== 队列行数 ==="
wc -l tools/queue/tasks.txt
