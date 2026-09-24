#!/bin/bash
cd /home/fedsa/DyGLib
echo "== 变体产物（正确路径计数） =="
for tag in B2 B3 B4 B5; do
  nl=$(ls saved_results/SignLinkPrediction/SignDyGFormer/*/SignDyGFormer_seed42.*.$tag.json 2>/dev/null | wc -l)
  ns=$(ls saved_results/LinkSign/SignDyGFormer/*/SignDyGFormer_seed42.*.$tag.json 2>/dev/null | wc -l)
  echo ".$tag  linksign=$nl/5  sign=$ns/5"
done
echo "== B2 训练时长与指标（样例） =="
python3 - <<'PY'
import json, glob
for task, d in [("linksign","SignLinkPrediction"), ("sign","LinkSign")]:
    for f in sorted(glob.glob(f"saved_results/{d}/SignDyGFormer/*/SignDyGFormer_seed42.*.B2.json"))[:3]:
        j=json.load(open(f)); tm=j["test metrics"]
        key = "f1_wt" if task=="linksign" else "f1_macro"
        print(f"{task}: {f.split('/')[2]:<20} train={j.get('training time (s)')}  auc={tm['auc']}  {key}={tm[key]}")
PY
echo "== queue.log 尾部 3 行 =="
tail -3 tools/queue/queue.log | cut -c1-120
