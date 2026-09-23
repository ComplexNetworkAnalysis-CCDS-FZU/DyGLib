#!/bin/bash
cd /home/fedsa/DyGLib
for ds in WikiVote RedditHyperlinkTitle RedditHyperlinkBody; do
  echo "=== $ds seed42: plain pkl? ==="
  case $ds in
    WikiVote) NN=15; LF=10;;
    RedditHyperlinkTitle) NN=60; LF=1;;
    RedditHyperlinkBody) NN=80; LF=3;;
  esac
  ls -la saved_models/SignLinkPrediction/SignDyGFormer/$ds/SignDyGFormer_seed42/SignDyGFormer_seed42.NN-$NN.LF-$LF.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.pkl 2>&1 | tail -1
  ls saved_models/SignLinkPrediction/SignDyGFormer/$ds/SignDyGFormer_seed42/ | grep -c "\.pkl$"
done
