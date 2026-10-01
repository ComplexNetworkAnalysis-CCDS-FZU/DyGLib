#!/bin/bash
echo "== 服务器上可能的 CSV 位置 =="
for p in /home/fedsa/DyGLib/processed_data /home/fedsa/DynamiSE_DySDGNN_repro/processed_data /home/fedsa/DyGLib/DG_data; do
  echo "--- $p"
  ls "$p" 2>/dev/null | head -8
done
echo "== 关键 CSV =="
ls -l /home/fedsa/DyGLib/processed_data/BitcoinAlpha/ml_BitcoinAlpha.csv /home/fedsa/DyGLib/processed_data/BitcoinOTC/ml_BitcoinOTC.csv /home/fedsa/DyGLib/processed_data/WikiVote/ml_WikiVote_tail20000.csv 2>&1 | cut -c1-120
echo "== repro configs/datasets.yaml 现状 =="
sed -n '1,25p' /home/fedsa/DynamiSE_DySDGNN_repro/configs/datasets.yaml
echo "== 工作树是否被改过 =="
cd /home/fedsa/DynamiSE_DySDGNN_repro && git status --porcelain | grep -E "configs/|data/" | head -5
echo "(空=未改)"
