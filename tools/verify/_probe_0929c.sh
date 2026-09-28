#!/bin/bash
# 9-29 探测 C：mamba trainer 源码了解运行时间/进度信号
cd /home/fedsa/DynamiSE_DySDGNN_repro/ext_baselines || exit 1
echo '=== trainer head (args/defaults) ==='
sed -n '1,90p' dyg_mamba/train_sign_dygmamba.py
echo '=== training loop core ==='
sed -n '150,290p' dyg_mamba/train_sign_dygmamba.py
