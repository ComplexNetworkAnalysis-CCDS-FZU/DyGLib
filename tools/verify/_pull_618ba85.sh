#!/bin/bash
cd ~/DyGLib
echo "== task_732（semba a735 方案①）日志 =="
cat tools/queue/logs/task_732.log | tail -40
echo
echo "== repro：还原热补丁 → ff 到 618ba85 =="
cd ~/DynamiSE_DySDGNN_repro
git checkout -- configs/datasets.yaml data/snapshot.py
echo "-- 还原后 status（应无 configs/data 改动）--"
git status --porcelain | grep -E "configs/|data/" | head -5
echo "(空=干净)"
git fetch --all -q 2>&1 | tail -2
echo "-- 拉取 --"
git pull --ff-only 2>&1 | tail -4
git log -1 --oneline | cut -c1-64
echo "== 校验新解析 =="
grep -n "DYSDGNN_DATA_ROOT\|def expand_path\|DATA_ROOT =" data/snapshot.py | head -8
grep -n "csv_path" configs/datasets.yaml | head -4
. /home/fedsa/anaconda3/etc/profile.d/conda.sh
conda activate gc
python - <<'PY'
import sys
sys.path.insert(0, ".")
import importlib
m = importlib.import_module("data.snapshot")
print("DATA_ROOT resolved =", getattr(m, "DATA_ROOT", None))
PY
