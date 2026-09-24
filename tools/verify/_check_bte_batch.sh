#!/bin/bash
cd /home/fedsa/DyGLib
echo "== linksign BTE 变体产物（正确 glob） =="
for tag in B2 B3 B4 B5; do
  n=$(ls saved_results/SignLinkPrediction/SignDyGFormer/*/*seed42*.$tag.json 2>/dev/null | wc -l)
  echo "linksign .$tag: $n/5"
done
echo "== sign BTE 变体产物 =="
for tag in B2 B3 B4 B5; do
  n=$(ls saved_results/LinkSign/SignDyGFormer/*/*seed42*.$tag.json 2>/dev/null | wc -l)
  echo "sign .$tag: $n/5"
done
echo "== 各变体行日志分类（365-404） =="
for t in $(seq 365 404); do
  L="tools/queue/logs/task_$t.log"
  [ -f "$L" ] || continue
  if grep -q "Traceback" "$L" 2>/dev/null; then
    err=$(grep -m1 "Error" "$L" | cut -c1-60)
    echo "task_$t: CRASH ($err)"
  else
    ok=$(grep -c "dump\|final\|best" "$L" 2>/dev/null | head -1)
    echo "task_$t: ok?"
  fi
done | head -45
