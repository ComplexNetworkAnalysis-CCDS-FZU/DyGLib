#!/bin/bash
cd ~/DyGLib || exit 1
echo '=== RB combo per-seed param.json (FX thresholds) ==='
for s in 42 123 456 789 1024; do
  F="saved_models/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/SignDyGFormer_seed${s}/SignDyGFormer_seed${s}.NN-80.LF-3.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.TF-E.RK-80.param.json"
  echo "-- seed${s}:"
  cat "$F" 2>/dev/null || echo "MISSING: $F"
  echo
done
echo '=== ckpt pkl present? ==='
ls -la saved_models/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/SignDyGFormer_seed42/*TF-E.RK-80* 2>/dev/null | awk '{print $5, $9}'
echo '=== GPU1 process / mamba progress ==='
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
ps -o pid,etime,cmd -p 628362 2>/dev/null | tail -1 | cut -c1-100
ls -la /home/fedsa/DynamiSE_DySDGNN_repro/outputs/DyG-Mamba/
echo '=== queue log tail ==='
tail -3 tools/queue/queue.log
