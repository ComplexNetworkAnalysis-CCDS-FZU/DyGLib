#!/bin/bash
cd /home/fedsa/DyGLib
source /home/fedsa/anaconda3/etc/profile.d/conda.sh && conda activate gc
echo "== CLI 旗标解析（应 ≥4） =="
python train_sign_link_3class_prediction.py --help 2>/dev/null | grep -c "module-bte-b"
python train_sign_link_3class_prediction.py --help 2>/dev/null | grep -o "module-bte-b[2345][a-z-]*" | sort | head -8
echo "== 服务器 CPU 冒烟（四变体） =="
python tools/verify/_probe_b2345_smoke.py 2>&1 | tail -10
echo "== D3b 转储进度 =="
ls results/samples_dump/*/ 2>/dev/null | wc -l
ls -la results/samples_dump/*/*.npz 2>/dev/null | awk '{print $NF, $5}' | tail -20
echo "== queue.log 尾部 4 行 =="
tail -4 tools/queue/queue.log
