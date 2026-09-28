#!/bin/bash
# 9-29 晨探测：param.json 结构 / RB 组合进度 / mamba 进度 / 队列
cd ~/DyGLib || exit 1
date
echo '=== param.json sample (grid RT 15/3 seed42) ==='
F=$(find saved_models -path '*RedditHyperlinkTitle*' -name '*NN-15.LF-3*param.json' 2>/dev/null | head -1)
echo "FILE=$F"
cat "$F" 2>/dev/null | head -50
echo '=== RB combo: newest log dirs (RedditHyperlinkBody) ==='
ls -lt logs/SignDyGFormer/RedditHyperlinkBody/ 2>/dev/null | head -10
echo '=== RB combo result files (TF-E.RK-80) ==='
ls saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/ | grep -c 'TF-E.RK-80' || true
ls -lt saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/ | grep 'TF-E.RK-80' | head -8 || true
echo '=== RB combo log tail ==='
L=$(ls -t logs/SignDyGFormer/RedditHyperlinkBody/*/*.log 2>/dev/null | head -1)
echo "LOG=$L"
tail -25 "$L" 2>/dev/null
echo '=== mamba test ==='
ls -la /tmp/mamba_wrap_test.log
ls /home/fedsa/DynamiSE_DySDGNN_repro/outputs/DyG-Mamba/ 2>/dev/null | head
echo '=== queue ==='
wc -l < tools/queue/running.txt
tail -6 tools/queue/queue.log
echo '=== GPU ==='
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ps -o pid,etime,stat,cmd -p 614899,616212 2>/dev/null | cut -c1-160
