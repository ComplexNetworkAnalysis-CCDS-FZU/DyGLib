#!/bin/bash
. /home/fedsa/anaconda3/etc/profile.d/conda.sh
conda activate gc
cd /home/fedsa/DynamiSE_DySDGNN_repro
echo "== dry-run（打印将要执行的命令）=="
python ext_baselines/semba/run_semba_queue.py --mode A --datasets RedditHyperlinkTitle --tasks linksign --seeds 1024 \
  --device cuda:0 --processed-root /home/fedsa/DyGLib/processed_data --force --dry-run 2>&1 | tail -20
echo "== 现有 semba 产物位置 =="
ls outputs/semba_aligned/semba/ 2>/dev/null | grep -i "RedditHyperlinkTitle" | head -5
ls outputs/semba_aligned/ 2>/dev/null | head -5
echo "== 现有 249 件总数 =="
find outputs/semba_aligned -name "*.json" 2>/dev/null | wc -l
