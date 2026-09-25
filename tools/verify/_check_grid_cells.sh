#!/bin/bash
cd /home/fedsa/DyGLib
echo "== 目标格删除校验（应全部 MISSING） =="
miss=0; exist=0
for ds in RedditHyperlinkTitle RedditHyperlinkBody WikiVote; do
  for nn in 15 40 60 80 100; do
    for lf in 1 3 5 10 15; do
      for d in SignLinkPrediction LinkSign; do
        f="saved_results/$d/SignDyGFormer/$ds/SignDyGFormer_seed42.NN-$nn.LF-$lf.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
        if [ -f "$f" ]; then exist=$((exist+1)); echo "STILL EXISTS: $f"; else miss=$((miss+1)); fi
      done
    done
  done
done
echo "缺失(=已删) $miss / 存在 $exist （期望 150/0）"
echo "== 六配置点抽查 =="
for f in \
 "saved_results/SignLinkPrediction/SignDyGFormer/WikiVote/SignDyGFormer_seed42.NN-15.LF-10.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json" \
 "saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed42.NN-60.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json" \
 "saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/SignDyGFormer_seed42.NN-80.LF-3.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json" \
 "saved_results/LinkSign/SignDyGFormer/WikiVote/SignDyGFormer_seed42.NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json" \
 "saved_results/LinkSign/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed42.NN-100.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json" \
 "saved_results/LinkSign/SignDyGFormer/RedditHyperlinkBody/SignDyGFormer_seed42.NN-60.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json" ; do
  [ -f "$f" ] && echo "EXISTS: $(basename $(dirname $f))/$(basename $f)" || echo "deleted(ok): $(basename $(dirname $f))"
done
