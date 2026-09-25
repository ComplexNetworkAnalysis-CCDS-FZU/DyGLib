#!/bin/bash
set -e
cd ~/DyGLib
source /home/fedsa/anaconda3/etc/profile.d/conda.sh
conda activate gc
echo "== git pull =="
git pull --ff-only
echo "== git log -1 =="
git log -1 --oneline
echo "== 关键标记 grep（应为 4/2/2/2 左右） =="
grep -c "e2_self_recent" utils/direct_neighbor_sampler.py utils/load_configs.py train_sign_link_3class_prediction.py train_link_sign_prediction.py
echo "== E2 引擎冒烟（真实数据；CPU） =="
python tools/verify/_probe_e2_smoke.py 2>&1 | tail -8
echo "== E2 单测 =="
python tools/verify/test_e2_guard.py 2>&1 | tail -6
echo "== CLI 旗标存在性 =="
python train_link_sign_prediction.py --help 2>&1 | grep -c "e2-self-recent"
