#!/bin/bash
# 2026-09-24: D3b 转储行的 ckpt 名核对（linksign seed42，5 数据集）
cd /home/fedsa/DyGLib
for ds in WikiVote RedditHyperlinkTitle RedditHyperlinkBody BitcoinAlpha BitcoinOTC; do
  echo "===== $ds"
  d="saved_models/SignLinkPrediction/SignDyGFormer/$ds/SignDyGFormer_seed42"
  echo "-- 候选总数: $(ls $d/*.pkl 2>/dev/null | wc -l)"
  echo "-- full 系（RAS-E.RASE-E.BTE-E.CNAS-E，非 .G1/.TF-E/RLF/REV）:"
  ls $d/ 2>/dev/null | grep -E "RAS-E\.RASE-E\.BTE-E\.CNAS-E\.P1\.TE(\.pkl|\.json)" | grep -v -E "G1|TF-E|RLF|REV" | head -8
  echo "-- noBTE（BTE-D）:"
  ls $d/ 2>/dev/null | grep -E "BTE-D" | grep -v -E "G1|TF-E" | head -8
  echo "-- G1 系:"
  ls $d/ 2>/dev/null | grep -E "\.G1" | head -8
done
