#!/bin/bash
cd /home/fedsa/DyGLib
echo "== B2-OTC 种子文件 =="
ls -lt saved_results/SignLinkPrediction/SignDyGFormer/BitcoinOTC/*B2.json 2>/dev/null | awk '{print $6,$7,$8,$NF}' | head -8
echo "== 数量 =="
ls saved_results/SignLinkPrediction/SignDyGFormer/BitcoinOTC/*B2.json 2>/dev/null | wc -l
echo "== B4-RB 种子文件 =="
ls -lt saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/*B4.json 2>/dev/null | awk '{print $6,$7,$8,$NF}' | head -4
echo "== 进程 =="
ps -u fedsa -o pid,etime,cmd | grep -E "train_(sign|link)" | grep -v grep | cut -c1-130
