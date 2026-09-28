#!/bin/bash
# 9-29 探测 B：grid 日志可用性（Gate1 val 提取源）+ mamba/combo 进度
cd ~/DyGLib || exit 1
echo '=== grid run logs availability (seed42) ==='
for spec in "RedditHyperlinkTitle NN-15.LF-3" "RedditHyperlinkTitle NN-60.LF-1" "RedditHyperlinkBody NN-60.LF-1" "RedditHyperlinkBody NN-80.LF-3"; do
  set -- $spec
  ds=$1; pat=$2
  d="logs/SignDyGFormer/$ds/SignDyGFormer_seed42"
  echo "-- $spec : $(ls -t $d/*.log 2>/dev/null | wc -l) logs total"
  ls -lt $d/*.log 2>/dev/null | head -3
done
echo
echo '=== one grid log structure check (RT 15/3): find save model + validate block ==='
F=$(ls -t logs/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed42/*.log 2>/dev/null | head -1)
echo "FILE=$F"
grep -n "save model\|Early stop\|num_neighbors=15" "$F" 2>/dev/null | head -12
echo '--- last validate block before last save ---'
grep -n "validate f1_wt\|validate f1_mac\|validate auc" "$F" 2>/dev/null | tail -9
echo
echo '=== mamba test: proc stats ==='
ps -o pid,etime,time,rss,stat -p 616212 2>/dev/null
cat /proc/616212/stat 2>/dev/null | awk '{print "state="$3" utime="$14" stime="$15}'
ls -la /tmp/mamba_wrap_test.log
echo '--- outputs dir ---'
find /home/fedsa/DynamiSE_DySDGNN_repro/outputs -newermt '2026-09-28 21:30' -type f 2>/dev/null | head -10
echo '--- mamba pkg print statements (buffering hints) ---'
grep -n "print(" /home/fedsa/DynamiSE_DySDGNN_repro/ext_baselines/third_party/DyG-Mamba/../../dyg_mamba/train_sign_dygmamba.py 2>/dev/null | head -8
echo
echo '=== RB combo now ==='
ls saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/ | grep -c 'TF-E.RK-80' || true
date
