#!/bin/bash
cd /home/fedsa/DyGLib
echo "== task_383/384/385 日志尾部 =="
for t in 383 384 385; do
  echo "--- task_$t"
  tail -3 tools/queue/logs/task_$t.log 2>/dev/null | cut -c1-160
  grep -E "Traceback|Error" tools/queue/logs/task_$t.log | head -2
done
echo "== BTE 变体产物计数（linksign） =="
for tag in B2 B3 B4 B5; do
  n=$(ls saved_results/SignLinkPrediction/SignDyGFormer/*/*seed42*.$tag.P1.TE.json 2>/dev/null | wc -l)
  echo "linksign .$tag: $n/5"
done
echo "== BTE 变体产物计数（sign） =="
for tag in B2 B3 B4 B5; do
  n=$(ls saved_results/LinkSign/SignDyGFormer/*/*seed42*.$tag.P1.TE.json 2>/dev/null | wc -l)
  echo "sign .$tag: $n/5"
done
echo "== sign noBTE 产物 =="
n=$(ls saved_results/LinkSign/SignDyGFormer/*/*seed42*BTE-D.CNAS-E.P1.TE.json 2>/dev/null | wc -l)
echo "sign w/o BTE: $n/5"
echo "== running.txt =="
wc -l tools/queue/running.txt
