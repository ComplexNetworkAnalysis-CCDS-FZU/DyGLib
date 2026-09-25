#!/bin/bash
cd ~/DyGLib/tools/queue
echo "== 415-421 原始（截断显示） =="
sed -n '415,421p' tasks.txt | cut -c1-260
echo "== 437 原始 =="
sed -n '437p' tasks.txt | cut -c1-260
echo "== 422-436 seeds 字段抽查 =="
sed -n '422p;427p;432p;437p' tasks.txt | grep -o -- "--seeds [0-9,]*"
echo "== B2/B4/B5 各数据集 5 种子完整性 =="
cd ~/DyGLib
for tag in B2 B4 B5; do
  for d in WikiVote RedditHyperlinkTitle RedditHyperlinkBody BitcoinAlpha BitcoinOTC; do
    n=$(ls saved_results/SignLinkPrediction/SignDyGFormer/$d 2>/dev/null | grep -c "\.$tag\.json" || true)
    printf "%s %s: %s\n" "$tag" "$d" "$n"
  done
done
echo "== 最新 B2-OTC / B4-RB 文件时间 =="
ls -lt saved_results/SignLinkPrediction/SignDyGFormer/BitcoinOTC/*.B2.json 2>/dev/null | head -6
ls -lt saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/*.B4.json 2>/dev/null | head -6
