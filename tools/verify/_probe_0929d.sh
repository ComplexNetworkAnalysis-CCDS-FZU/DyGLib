#!/bin/bash
cd /home/fedsa/DynamiSE_DySDGNN_repro/ext_baselines || exit 1
echo '=== trainer args default (tail) ==='
sed -n '290,395p' dyg_mamba/train_sign_dygmamba.py
