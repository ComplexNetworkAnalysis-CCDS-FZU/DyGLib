#!/bin/bash
cd ~/DyGLib
for ds in BitcoinAlpha BitcoinOTC WikiVote; do
  echo "=== $ds (tasks.txt 最新命中) ==="
  grep -m1 "train_link_sign_prediction.py --dataset-name $ds" tools/queue/tasks.txt | cut -c1-430
done
echo "=== 备份补充（若无命中） ==="
for b in $(ls -t tools/queue/tasks.txt.bak-* 2>/dev/null | head -12); do
  for ds in BitcoinAlpha BitcoinOTC WikiVote; do
    grep -m1 "train_link_sign_prediction.py --dataset-name $ds" "$b" | cut -c1-430
  done
done | head -8
