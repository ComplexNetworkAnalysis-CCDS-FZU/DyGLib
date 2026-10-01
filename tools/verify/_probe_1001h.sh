#!/bin/bash
cd ~/DynamiSE_DySDGNN_repro
echo "== third_party ScaDyG =="
ls ext_baselines/third_party/ 2>/dev/null
ls ext_baselines/third_party/ScaDyG/ 2>/dev/null | head -12
echo "== 分片位置 =="
find ext_baselines -name "*.npz" 2>/dev/null | head -10
echo "== scadyg README 运行段 =="
grep -n -A3 -B1 "csv2npz\|train_sign_scadyg" ext_baselines/scadyg/README.md 2>/dev/null | head -40
