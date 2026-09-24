#!/bin/bash
cd /home/fedsa/DyGLib
source /home/fedsa/anaconda3/etc/profile.d/conda.sh && conda activate gc
echo "== 拉取 =="
git fetch origin && git pull --ff-only | tail -3
echo "== 后台重算 linksign 掩码（含极性） =="
nohup python tools/verify/bte_sparsity_stats.py --task linksign --quiet > results/bte_sparsity/rerun_linksign_20260924.log 2>&1 &
echo "started pid=$!"
echo "== 日志开头 =="
head -5 results/bte_sparsity/rerun_linksign_20260924.log 2>/dev/null || echo "(日志尚未生成)"
