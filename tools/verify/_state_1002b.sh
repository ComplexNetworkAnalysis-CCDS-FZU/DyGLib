#!/bin/bash
cd ~/DyGLib
. /home/fedsa/anaconda3/etc/profile.d/conda.sh
conda activate gc
date "+%m-%d %H:%M"
echo "== 队列 =="
wc -l < tools/queue/tasks.txt; wc -l < tools/queue/running.txt
tail -3 tools/queue/queue.log | cut -c1-95
ps -ef | grep -c "train_link_sign_prediction"
echo "== TE 臂结果（sign 任务 → saved_results/LinkSign/SignDyGFormer/<ds>）=="
for ds in BitcoinAlpha BitcoinOTC WikiVote; do
  echo "-- $ds"
  ls -l --time-style=+%m-%d_%H:%M saved_results/LinkSign/SignDyGFormer/$ds/ 2>/dev/null | grep -E "NN-40.LF-15" | tail -20
done
echo "== 内容速览（TE/TD/TD-LIN 各取最新）=="
python - <<'PY'
import json, glob, os
for ds in ("BitcoinAlpha", "BitcoinOTC", "WikiVote"):
    fs = sorted(glob.glob(f"saved_results/LinkSign/SignDyGFormer/{ds}/*NN-40.LF-15*.json"))
    print(f"-- {ds}: {len(fs)} 件")
    for f in fs[-6:]:
        d = json.load(open(f, encoding="utf-8"))
        m = d.get("test metrics", {})
        print("   ", os.path.basename(f)[:78].ljust(78),
              "auc=", m.get("auc"), "f1_macro=", m.get("f1_macro"))
PY
echo "== 主口径对照（sign_valthr 归档，服务器侧）=="
ls saved_results/LinkSign/SignDyGFormer/BitcoinAlpha/*seed42.NN-40.LF-15*.json 2>/dev/null | tail -3
