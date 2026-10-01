#!/bin/bash
cd ~/DyGLib
. /home/fedsa/anaconda3/etc/profile.d/conda.sh
conda activate gc
echo "== 队列/日志 =="
wc -l < tools/queue/tasks.txt; wc -l < tools/queue/running.txt
tail -3 tools/queue/queue.log | cut -c1-95
for i in 738 739; do echo "--- $i tail:"; tail -3 tools/queue/logs/task_$i.log 2>/dev/null | cut -c1-140; done
echo "== ScaDyG 结果（repro）=="
ls -l --time-style=+%m-%d_%H:%M ~/DynamiSE_DySDGNN_repro/outputs/ScaDyG/ | tail -16
echo "== ScaDyG AUC 速览 =="
python - <<'PY'
import json, glob, os
fs = sorted(glob.glob(os.path.expanduser("~/DynamiSE_DySDGNN_repro/outputs/ScaDyG/*.json")))
for f in fs:
    d = json.load(open(f, encoding="utf-8"))
    m = d.get("metrics", d.get("test metrics", {}))
    print("  ", os.path.basename(f).ljust(30), "AUC=", m.get("AUC", m.get("auc")), "val=", d.get("val_AUC"))
PY
echo "== grid_new 覆盖（是否有 Bitcoin）=="
ls results/grid_new/raw/ 2>/dev/null
for t in linksign sign; do echo "  $t: $(ls results/grid_new/raw/$t 2>/dev/null | tr '\n' ' ')"; done
ls results/grid_new/raw/linksign/ 2>/dev/null | while read -r d; do echo "    linksign/$d: $(ls results/grid_new/raw/linksign/$d | wc -l) files"; done
echo "== 单种子确认（抽样文件名）=="
ls results/grid_new/raw/linksign/RedditHyperlinkTitle 2>/dev/null | head -3
ls results/grid_new/raw/sign/WikiVote 2>/dev/null | head -3
