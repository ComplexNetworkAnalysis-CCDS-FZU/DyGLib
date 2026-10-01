. /home/fedsa/anaconda3/etc/profile.d/conda.sh
conda activate gc
#!/bin/bash
cd /home/fedsa/DynamiSE_DySDGNN_repro
echo "== 冒烟 JSON（CPU 2ep）=="
python - <<'PY'
import json
d = json.load(open("outputs/_smoke/DySDGNN_visibility/BitcoinAlpha_seed42_C012.json", encoding="utf-8"))
print("顶层键:", sorted(d.keys()))
print("metrics(=C0):", d.get("metrics"))
print("protocol_metrics:", json.dumps(d.get("protocol_metrics"), ensure_ascii=False))
for k in ("eval_protocol", "approx", "approx_kind", "n_eval_edges", "runtime_s"):
    if k in d:
        print(f"  {k} = {d[k]}")
PY
echo "== 现有主表 BA/seed42 对账（outputs/DySDGNN/）=="
python - <<'PY'
import json
d = json.load(open("outputs/DySDGNN/BitcoinAlpha_seed42.json", encoding="utf-8"))
print("主表:", d.get("metrics"))
PY
