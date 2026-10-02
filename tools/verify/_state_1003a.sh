#!/bin/bash
cd ~/DyGLib
. /home/fedsa/anaconda3/etc/profile.d/conda.sh
conda activate gc
date "+%m-%d %H:%M"
echo "== 队列 =="
wc -l < tools/queue/tasks.txt; wc -l < tools/queue/running.txt
tail -6 tools/queue/queue.log | cut -c1-95
echo "== 任务日志尾（742/743/744）=="
for i in 742 743 744; do
  echo "--- task_$i ($(wc -c < tools/queue/logs/task_$i.log 2>/dev/null) bytes) 尾 6:"
  tail -6 tools/queue/logs/task_$i.log 2>/dev/null | cut -c1-150
done
echo "== 失败标记 grep =="
for i in 742 743 744; do
  n=$(grep -c -E "TE_FAIL|TD_FAIL|LIN_FAIL|GUARD_FAIL|Traceback" tools/queue/logs/task_$i.log 2>/dev/null)
  echo "  task_$i 失败行数=$n"
done
grep -h -E "TE_FAIL|TD_FAIL|LIN_FAIL|Traceback" tools/queue/logs/task_74*.log 2>/dev/null | head -5 | cut -c1-140
echo "== 三策略产物计数（LinkSign/SignDyGFormer）=="
python - <<'PY'
import glob, os
for ds in ("BitcoinAlpha", "BitcoinOTC", "WikiVote"):
    base = f"saved_results/LinkSign/SignDyGFormer/{ds}"
    for pat in ("NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json",
                "NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TD.json",
                "NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TD-LIN.json"):
        n = len(glob.glob(f"{base}/*{pat}"))
        print(f"  {ds:<14} {pat.split('.P1.')[1]:<16} {n} 件")
PY
echo "== 速览（TE vs 主口径核对用：BA seed42）=="
python - <<'PY'
import json, glob, os
for f in sorted(glob.glob("saved_results/LinkSign/SignDyGFormer/BitcoinAlpha/*seed42*P1.TD*.json")):
    d = json.load(open(f, encoding="utf-8")); m = d.get("test metrics", {})
    print("  ", os.path.basename(f)[-40:], "auc=", m.get("auc"), "f1_macro=", m.get("f1_macro"))
PY
