#!/bin/bash
. /home/fedsa/anaconda3/etc/profile.d/conda.sh
conda activate gc
cd /home/fedsa/DynamiSE_DySDGNN_repro
echo "== 目录树 =="
find outputs/_smoke outputs/DySDGNN_visibility -name "*.json" -printf "%TY-%Tm-%Td %TH:%TM  %p\n" 2>/dev/null | sort
echo "== 内容核对 =="
python - <<'PY'
import json, glob
for f in sorted(glob.glob("outputs/_smoke/*/BitcoinAlpha_seed42_C012.json")) + sorted(glob.glob("outputs/DySDGNN_visibility/BitcoinAlpha_seed42_C012.json")):
    d = json.load(open(f, encoding="utf-8"))
    pm = d.get("protocol_metrics", {})
    print(f)
    print("   device=%s epochs=%s runtime=%.1f" % (d.get("device"), d.get("config", {}).get("epochs"), d.get("runtime_s", -1)))
    print("   metrics(C0)=%s" % d.get("metrics"))
    for t in ("C0", "C1", "C2"):
        if t in pm:
            print("   %s AUC=%.6f F1=%.6f" % (t, pm[t]["AUC"], pm[t]["F1_bin"]))
PY
echo "== 队列 =="
cd ~/DyGLib && wc -l < tools/queue/running.txt && tail -2 tools/queue/queue.log | cut -c1-100
