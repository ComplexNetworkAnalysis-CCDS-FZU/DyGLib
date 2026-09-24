#!/bin/bash
cd /home/fedsa/DyGLib
echo "=== 网格进度（linksign：每数据集唯一 (NN,LF) 组合数 / 42）==="
for ds in RedditHyperlinkTitle RedditHyperlinkBody BitcoinAlpha BitcoinOTC WikiVote; do
  n=$(ls saved_results/SignLinkPrediction/SignDyGFormer/$ds/ 2>/dev/null | grep -E "seed42\.NN-(10|15|20|40|60|80|100)\.LF-(1|3|5|10|15|20)\.RAS-E\.RASE-E\.BTE-E\.CNAS-E\.P1\.TE\.json$" | wc -l)
  echo "$ds: $n/42"
done
echo "=== 运行进程（含父级）==="
ps -eo pid,ppid,etime,args | grep -E "run_experiments|train_(link_sign|sign_link)" | grep -v grep | awk '{print $1, $2, $3, $5, $6, $7, $8, $9}' | head -8
