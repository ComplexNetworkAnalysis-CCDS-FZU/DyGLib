#!/bin/bash
cd ~/DyGLib
echo "== CNS-D 训练件全路径（全库）=="
find . -name "*CNAS-D*.G2.json" 2>/dev/null | sed 's#^\./##' | sort | head -30
echo "== CNS-FX 全路径 =="
find . -name "*CNAS-D*G2.EVT.FX.json" 2>/dev/null | sed 's#^\./##' | sort
echo "== saved_results 结构 =="
ls saved_results/SignLinkPrediction/ 
echo "-- RB 目录文件数: $(ls saved_results/SignLinkPrediction/RedditHyperlinkBody/ 2>/dev/null | wc -l)"
echo "== RB 目录内 .G2 相关 =="
ls saved_results/SignLinkPrediction/RedditHyperlinkBody/ 2>/dev/null | grep -E "G1|G2" | head -20
echo "== RB 目录最近 8 个文件 =="
ls -lt --time-style=+%m-%d_%H:%M saved_results/SignLinkPrediction/RedditHyperlinkBody/ 2>/dev/null | head -9
