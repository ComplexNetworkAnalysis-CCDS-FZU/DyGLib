#!/bin/bash
cd /home/fedsa/DyGLib
echo "== G1-EVT 修正批（非 TF）产物计数 =="
for ds in WikiVote RedditHyperlinkTitle RedditHyperlinkBody BitcoinAlpha BitcoinOTC; do
  n=$(ls saved_results/SignLinkPrediction/SignDyGFormer/$ds/*P1.TE.G1.EVT.FX.json 2>/dev/null | grep -v "TF-E" | wc -l)
  echo "$ds: $n/5"
done
echo "== G1-EVT 修正批（sign 侧） =="
for ds in WikiVote RedditHyperlinkTitle RedditHyperlinkBody BitcoinAlpha BitcoinOTC; do
  n=$(ls saved_results/LinkSign/SignDyGFormer/$ds/*P1.TE.G1.EVT.FX.json 2>/dev/null | grep -v "TF-E" | wc -l)
  echo "$ds: $n/1"
done
echo "== queue.log 尾部 4 行 =="
tail -4 tools/queue/queue.log | cut -c1-110
