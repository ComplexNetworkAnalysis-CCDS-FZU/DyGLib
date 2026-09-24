#!/bin/bash
cd /home/fedsa/DyGLib
echo "== T16 组合行产物（RB linksign） =="
ls saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/ | grep "RK-80.G1" | head -20
echo ""
echo "== 数量 =="
echo "train: $(ls saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/*seed*.RK-80.G1.json 2>/dev/null | grep -v EVT | wc -l)/5"
echo "evt  : $(ls saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/*seed*.RK-80.G1.EVT.FX.json 2>/dev/null | wc -l)/5"
echo "== queue.log 尾部 3 行 =="
tail -3 tools/queue/queue.log | cut -c1-110
