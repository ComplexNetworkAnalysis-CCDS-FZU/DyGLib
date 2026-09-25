#!/bin/bash
# 2026-09-25: 执行 Paper 0656/78f6：停旧网格 → 备份旧件 → 删除待重跑格（代际隔离）
cd /home/fedsa/DyGLib
echo "== 停网格进程树 =="
for P in $(pgrep -f "run_experiments.py -s linksign -t parameter"); do
  for c in $(pgrep -P "$P"); do kill -TERM "$c" 2>/dev/null && echo "killed child $c"; done
  kill -TERM "$P" 2>/dev/null && echo "killed parent $P"
done
for i in $(seq 1 20); do
  pgrep -f "run_experiments.py -s linksign -t parameter" >/dev/null || break
  sleep 1
done
echo "== 残留检查 =="
pgrep -f "run_experiments.py -s linksign -t parameter" || echo "(网格已停)"
ps -u fedsa -o pid,cmd | grep "train_sign_link" | grep -v grep | head -3

echo "== 备份旧网格件（linksign RT/RB/WV 全量现存） =="
mkdir -p results/grid_oldgen_backup/linksign
for ds in RedditHyperlinkTitle RedditHyperlinkBody WikiVote; do
  mkdir -p results/grid_oldgen_backup/linksign/$ds
  cp -a saved_results/SignLinkPrediction/SignDyGFormer/$ds/SignDyGFormer_seed42.NN-*.LF-*.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json results/grid_oldgen_backup/linksign/$ds/ 2>/dev/null
  echo "$ds: $(ls results/grid_oldgen_backup/linksign/$ds/ 2>/dev/null | wc -l) 件"
done

echo "== 删除待重跑格（3 ds × 2 任务 × 25 点；seed42） =="
for ds in RedditHyperlinkTitle RedditHyperlinkBody WikiVote; do
  for nn in 15 40 60 80 100; do
    for lf in 1 3 5 10 15; do
      rm -f saved_results/SignLinkPrediction/SignDyGFormer/$ds/SignDyGFormer_seed42.NN-$nn.LF-$lf.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json
      rm -f saved_results/SignLinkPrediction/SignDyGFormer/$ds/SignDyGFormer_seed42.NN-$nn.LF-$lf.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE-profiler.json
      rm -f saved_results/LinkSign/SignDyGFormer/$ds/SignDyGFormer_seed42.NN-$nn.LF-$lf.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json
      rm -f saved_results/LinkSign/SignDyGFormer/$ds/SignDyGFormer_seed42.NN-$nn.LF-$lf.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE-profiler.json
    done
  done
done
echo "== 删除后剩余抽查（RT linksign） =="
ls saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkTitle/ | grep "seed42.NN-" | head -8
echo "剩余 42 格中应保留 17 格（42-25）：$(ls saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkTitle/ | grep -c 'seed42.NN-.*LF-.*RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json')"
