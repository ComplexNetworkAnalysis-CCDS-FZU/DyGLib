#!/bin/bash
# eval-only GPU 冒烟（卡1；occupies small footprint）
cd /home/fedsa/DyGLib
source /home/fedsa/anaconda3/etc/profile.d/conda.sh && conda activate gc
echo "=== 冒烟：WikiVote seed123（linksign eval-only @ full thr 0.65/0.01，GPU1）==="
CUDA_VISIBLE_DEVICES=1 timeout 1500 python train_sign_link_3class_prediction.py \
  --dataset-name WikiVote --model SignDyGFormer --gpu 0 --seeds 123 \
  --early-stop-notice f1_wt f1_mic ap f1_mac auc \
  --module-repeat-aware-sampler --module-repeat-aware-sign-encoder --cnas-tail-fill \
  --eval-only --eval-ckpt-name SignDyGFormer_seed123.NN-15.LF-10.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.TF-E \
  --sign-thr 0.65 --exist-thr 0.01 \
  --batch-size 200 --num-neighbors 15 --common-neighbors-look-forward 10 --tail-num 20000 2>&1 | tail -26
echo "=== 产物检查 ==="
ls -la saved_results/SignLinkPrediction/SignDyGFormer/WikiVote/ 2>/dev/null | grep -E "EVT" || echo "（未发现 EVT 结果文件）"
