#!/bin/bash
cd /home/fedsa/DyGLib
source /home/fedsa/anaconda3/etc/profile.d/conda.sh && conda activate gc
echo "== pull =="
git fetch origin && git pull --ff-only | tail -2
git log --oneline -1
echo "== 修复标记核对 =="
grep -n "构造期 layer 仍在 CPU" models/NeighborInteractEncoder.py | head -2
echo "== 设备回归（GPU1） =="
CUDA_VISIBLE_DEVICES=1 python tools/verify/_probe_b3_device_regression.py
echo "== B2 单行训练时长样例 =="
python3 -c "
import json, glob
for f in sorted(glob.glob('saved_results/SignLinkPrediction/SignDyGFormer/WikiVote/*seed42.B2.json'))[:2]:
    j=json.load(open(f)); print(f.split('/')[-1], 'train(s)=', j.get('training time (s)'))
for f in sorted(glob.glob('saved_results/LinkSign/SignDyGFormer/WikiVote/*seed42.B2.json'))[:2]:
    j=json.load(open(f)); print(f.split('/')[-1], 'train(s)=', j.get('training time (s)'))
"
