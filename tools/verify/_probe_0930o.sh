#!/bin/bash
cd ~/DyGLib || exit 1
date
echo '=== CNS-FX 计数（CNAS-D…G2.EVT.FX）==='
for spec in "RedditHyperlinkTitle 15 3" "RedditHyperlinkBody 60 1"; do
  set -- $spec
  n=$(ls saved_results/SignLinkPrediction/SignDyGFormer/$1/SignDyGFormer_seed*.NN-$2.LF-$3.RAS-E.RASE-E.BTE-E.CNAS-D.P1.TE.G2.EVT.FX.json 2>/dev/null | wc -l)
  echo "$1 NN-$2.LF-$3: $n/5"
  ls -l --time-style=+%H:%M saved_results/SignLinkPrediction/SignDyGFormer/$1/SignDyGFormer_seed*.NN-$2.LF-$3.RAS-E.RASE-E.BTE-E.CNAS-D.P1.TE.G2.EVT.FX.json 2>/dev/null | awk '{print $6, $7}' | head -6
done
echo '=== 指针/队列 ==='
echo "ptr=$(wc -l < tools/queue/running.txt) total=$(grep -c . tools/queue/tasks.txt)"
tail -4 tools/queue/queue.log | cut -c1-115
echo '=== mamba ==='
ls /home/fedsa/DynamiSE_DySDGNN_repro/outputs/DyG-Mamba/*.json 2>/dev/null | wc -l
ls -l --time-style=+%m-%d_%H:%M /home/fedsa/DynamiSE_DySDGNN_repro/outputs/DyG-Mamba/*.json 2>/dev/null | tail -4 | awk '{print $6, $7}'
echo '=== GPU ==='
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
