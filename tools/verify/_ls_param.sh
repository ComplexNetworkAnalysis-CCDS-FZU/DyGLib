#!/bin/bash
cd /home/fedsa/DyGLib
source /home/fedsa/anaconda3/etc/profile.d/conda.sh
conda activate gc
for ds in WikiVote RedditHyperlinkTitle RedditHyperlinkBody; do
  echo "=== $ds seed42 ==="
  ls saved_models/SignLinkPrediction/SignDyGFormer/$ds/SignDyGFormer_seed42/ | grep -E "seed42.*(P1\.TE|NN-15|NN-60|NN-80)[^/]*\.param\.json$" | head -20
done
