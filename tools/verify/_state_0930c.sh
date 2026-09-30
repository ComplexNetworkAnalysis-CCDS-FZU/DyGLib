#!/bin/bash
cd ~/DyGLib
echo "== git =="
git log -1 --oneline | cut -c1-60
echo "== 队列指针 =="
wc -l < tools/queue/tasks.txt
wc -l < tools/queue/running.txt 2>/dev/null
tail -4 tools/queue/queue.log | cut -c1-110
echo "== GPU =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ps -ef | grep -E "run_experiments|python" | grep -v grep | tail -5 | cut -c1-150
echo "== CNS-FX 计数（.CNAS-D....G2.EVT.FX）=="
for t in SignLinkPrediction LinkSign; do
  for d in BitcoinAlpha BitcoinOTC RedditHyperlinkBody WikiVote Epinions; do
    n=$(ls saved_results/$t/$d/ 2>/dev/null | grep -c "CNAS-D.*G2\.EVT\.FX\.json")
    [ "$n" != "0" ] && echo "  CNS-FX $t/$d: $n"
  done
done
echo "== mamba 计数 (pure) =="
for t in SignLinkPrediction LinkSign; do
  for d in BitcoinAlpha BitcoinOTC; do
    n=$(ls saved_results/$t/$d/ 2>/dev/null | grep -ciE "mamb")
    [ "$n" != "0" ] && echo "  mamba $t/$d: $n"
  done
done
ls saved_results/SignLinkPrediction/BitcoinAlpha/ 2>/dev/null | grep -iE "mamb" | tail -6
echo "== running.txt 尾 8 =="
tail -8 tools/queue/running.txt | cut -c1-150
echo "== 队列 712-726 =="
sed -n '712,726p' tools/queue/tasks.txt | cut -c1-170
