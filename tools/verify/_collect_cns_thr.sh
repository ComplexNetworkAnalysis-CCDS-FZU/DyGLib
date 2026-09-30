#!/bin/bash
# CNS 行（linksign RT 15/3、RB 60/1）逐种子 param.json 采集（供 _gen_cns_fx_rows.py）
cd ~/DyGLib || exit 1
for spec in "RedditHyperlinkTitle:15:3" "RedditHyperlinkBody:60:1"; do
  ds=${spec%%:*}; rest=${spec#*:}; nn=${rest%%:*}; lf=${rest##*:}
  for s in 42 123 456 789 1024; do
    F="saved_models/SignLinkPrediction/SignDyGFormer/${ds}/SignDyGFormer_seed${s}/SignDyGFormer_seed${s}.NN-${nn}.LF-${lf}.RAS-E.RASE-E.BTE-E.CNAS-D.P1.TE.G2.param.json"
    echo -n "KEY=${ds}|${nn}|${lf}|${s} "
    python3 -c "
import json,sys
p='$F'
try:
    d=json.load(open(p)); print('SIGN=%r EXIST=%r' % (d['best_sign_thr'], d['best_exist_thr']))
except Exception as e:
    print('MISSING')
"
  done
done
echo '=== CNS 件计数 ==='
for spec in "SignLinkPrediction RedditHyperlinkTitle 15 3" "SignLinkPrediction RedditHyperlinkBody 60 1" "LinkSign RedditHyperlinkTitle 60 3"; do
  set -- $spec
  echo "$1/$2 NN-$3.LF-$4: $(ls saved_results/$1/SignDyGFormer/$2/SignDyGFormer_seed*.NN-$3.LF-$4.RAS-E.RASE-E.BTE-E.CNAS-D.P1.TE.G2.json 2>/dev/null | wc -l)/5"
done
