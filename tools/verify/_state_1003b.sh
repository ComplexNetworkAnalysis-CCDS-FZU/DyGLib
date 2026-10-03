#!/bin/bash
cd ~/DyGLib
. /home/fedsa/anaconda3/etc/profile.d/conda.sh
conda activate gc
date "+%m-%d %H:%M"
echo "== 队列 =="
wc -l < tools/queue/tasks.txt; wc -l < tools/queue/running.txt
tail -5 tools/queue/queue.log | cut -c1-92
echo "== 失败标记（745-749）=="
for i in 745 746 747 748 749; do
  n=$(grep -c -E "TE_FAIL|TD_FAIL|LIN_FAIL|Traceback|error:" tools/queue/logs/task_$i.log 2>/dev/null)
  echo "  task_$i 失败行=$n  (log $(wc -c < tools/queue/logs/task_$i.log 2>/dev/null) B)"
done
echo "== 三策略产物计数 =="
python - <<'PY'
import glob
for ds in ("BitcoinAlpha", "BitcoinOTC", "WikiVote"):
    base = f"saved_results/LinkSign/SignDyGFormer/{ds}"
    row = []
    for tag in ("P1.TE.json", "P1.TD.json", "P1.TD-LIN.json"):
        row.append(f"{tag.split('.')[-2]}={len(glob.glob(f'{base}/*NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.{tag}'))}")
    print(f"  {ds:<14} " + "  ".join(row))
PY
echo "== smoke 产物（BA seed42）=="
python - <<'PY'
import json, glob, os
for f in sorted(glob.glob("saved_results/LinkSign/SignDyGFormer/BitcoinAlpha/*seed42*P1.TD*.json")):
    d = json.load(open(f, encoding="utf-8")); m = d.get("test metrics", {})
    print("  ", os.path.basename(f)[-38:], "auc=", m.get("auc"), "f1_macro=", m.get("f1_macro"))
PY
echo "== 队列尾 3 行任务号 =="
awk 'NR>=748 && NR<=749 {print NR": "substr($0,1,60)}' tools/queue/tasks.txt
