#!/bin/bash
cd ~/DyGLib || exit 1
date
echo '=== G2 counts per confirm point (exclude profiler) ==='
cnt() { ls saved_results/$1/SignDyGFormer/$2/ 2>/dev/null | grep -v profiler | grep -c "P1.TE.G2.json"; }
echo "linksign RT 15/3 : $(cnt SignLinkPrediction RedditHyperlinkTitle)"
echo "linksign RB 60/1 : $(cnt SignLinkPrediction RedditHyperlinkBody)"
echo "sign RT 60/3     : $(cnt LinkSign RedditHyperlinkTitle)"
echo "sign RB 40/1     : $(cnt LinkSign RedditHyperlinkBody)"
echo "sign WV 15/10    : $(cnt LinkSign WikiVote)"
echo
echo '=== G2 detail (all, newest 12) ==='
find saved_results -name '*P1.TE.G2.json' ! -name '*profiler*' -newermt '2026-09-29' 2>/dev/null | sort | tail -14
echo
echo '=== mamba outputs ==='
ls -l --time-style=+%m-%d_%H:%M /home/fedsa/DynamiSE_DySDGNN_repro/outputs/DyG-Mamba/ 2>/dev/null
echo '=== queue log tail 12 ==='
tail -12 tools/queue/queue.log | cut -c1-140
echo "tasks=$(wc -l < tools/queue/tasks.txt) running=$(wc -l < tools/queue/running.txt)"
echo '=== GPU ==='
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
