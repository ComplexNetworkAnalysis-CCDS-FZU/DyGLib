#!/bin/bash
cd ~/DyGLib/tools/queue
echo "== 关键批次行号（>400 部分）=="
awk 'NR>=400 { if (/b2-density/) print NR": B2"; else if (/b4-channel/) print NR": B4"; else if (/b5-continuous/) print NR": B5"; else if (/no-module-bte/) print NR": sign-noBTE"; else if (/no-accelerate|@/) {} }' tasks.txt | awk -F: '{c[$2]++; if(!(($2) in first)) first[$2]=$1; last[$2]=$1} END {for (k in c) print k": lines "first[k]"-"last[k]" ("c[k]")"}'
echo "== grid 行（LF + NN 组合）在哪些行 =="
grep -n "num-neighbors 15 --common-neighbors-look-forward 1 " tasks.txt | head -3
grep -n "num-neighbors 100 --common-neighbors-look-forward 15 " tasks.txt | tail -3
echo "== 400-428 行内容摘要 =="
sed -n '400,428p' tasks.txt | sed 's/.*--dataset-name \([A-Za-z]*\).*--seeds \([0-9,]*\).*--num-neighbors \([0-9]*\).*--common-neighbors-look-forward \([0-9]*\).*/\1 seed\2 NN\3 LF\4/; s/.*--module-bte-\([a-z0-9]*\).*/TAG bte-\1/' | sed 's/@cd.*/RAW/' | head -40
echo "== B2/B4/B5 结果（正确目录）=="
for tag in B2 B4 B5; do
  echo "--- $tag"
  ls expm-2026-09-25-logs/train_link_sign_prediction/*/ 2>/dev/null | grep "$tag" | sort | tail -30
done
echo "== 近 2 小时新 json =="; find expm-2026-09-25-logs -name "*.json" -mmin -120 2>/dev/null | sort | tail -30
