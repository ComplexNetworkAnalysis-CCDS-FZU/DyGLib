#!/bin/bash
cd ~/DynamiSE_DySDGNN_repro
git pull --ff-only 2>&1 | tail -3
git log -1 --oneline | cut -c1-60
echo "== 新解析校验 =="
grep -n "DYSDGNN_DATA_ROOT\|def expand_path\|^DATA_ROOT" data/snapshot.py | head -8
grep -n "csv_path" configs/datasets.yaml | head -4
. /home/fedsa/anaconda3/etc/profile.d/conda.sh
conda activate gc
python - <<'PY'
import sys, importlib
sys.path.insert(0, ".")
m = importlib.import_module("data.snapshot")
print("DATA_ROOT =", getattr(m, "DATA_ROOT", None))
import yaml, os
cfg = yaml.safe_load(open("configs/datasets.yaml", encoding="utf-8"))
for ds, v in cfg["datasets"].items():
    p = v["csv_path"]
    if hasattr(m, "expand_path"):
        p2 = m.expand_path(p)
    else:
        p2 = p
    print(f"  {ds:<14} exists={os.path.exists(p2)}  -> {p2}")
PY
echo "== scadyg 修复文件是否到位 =="
ls -l ext_baselines/scadyg/csv2npz.py ext_baselines/scadyg/edge_order.py ext_baselines/scadyg/train_sign_scadyg.py 2>&1 | cut -c1-100
ls scripts/verify_scadyg_fix.py 2>&1
echo "== 现有 ScaDyG 数据分片 =="
ls ext_baselines/scadyg/*.npz ext_baselines/scadyg/**/*.npz 2>/dev/null | head -5
find ext_baselines/scadyg -name "*.npz" 2>/dev/null | head -8
