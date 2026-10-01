#!/bin/bash
# 服务器本地路径修补（仅工作副本，未提交；可 git checkout 还原）
set -e
. /home/fedsa/anaconda3/etc/profile.d/conda.sh
conda activate gc
cd /home/fedsa/DynamiSE_DySDGNN_repro
echo "== 备份 diff =="
git diff > /tmp/repro_prepath_$(date +%Y%m%d-%H%M%S).diff || true

python - <<'PY'
import pathlib
y = pathlib.Path("configs/datasets.yaml")
t = y.read_text(encoding="utf-8")
n = t.replace("D:/codes/DyGLib/processed_data", "/home/fedsa/DyGLib/processed_data")
y.write_text(n, encoding="utf-8")
print("datasets.yaml 替换次数:", t.count("D:/codes/DyGLib/processed_data"))

s = pathlib.Path("data/snapshot.py")
ts = s.read_text(encoding="utf-8")
ns = ts.replace('DATA_ROOT = "D:/codes/DyGLib/processed_data"',
                'DATA_ROOT = "/home/fedsa/DyGLib/processed_data"')
s.write_text(ns, encoding="utf-8")
print("snapshot.py 替换次数:", ts.count('DATA_ROOT = "D:/codes/DyGLib/processed_data"'))
PY

echo "== 校验 =="
grep -n "processed_data" configs/datasets.yaml data/snapshot.py | head -6
echo "== 快速冒烟（CPU，2 epoch，仅验证数据路径可读）=="
timeout 900 python scripts/m5_run.py --model DySDGNN --datasets BitcoinAlpha --seeds 42 --epochs 2 \
  --device cpu --eval-protocol ALL --out-subdir _smoke/DySDGNN_visibility > /tmp/smoke_cpu.log 2>&1
echo "exit=$?"
tail -6 /tmp/smoke_cpu.log
ls -l outputs/_smoke/DySDGNN_visibility/ 2>/dev/null | tail -3
