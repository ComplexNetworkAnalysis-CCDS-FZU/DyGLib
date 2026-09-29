#!/bin/bash
cd ~/DyGLib || exit 1
echo '=== G2 EVT.FX files ==='
find saved_results/SignLinkPrediction -name '*P1.TE.G2.EVT.FX.json' 2>/dev/null | sort
echo
echo '=== counts ==='
echo "RT: $(ls saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkTitle/ | grep -c 'P1.TE.G2.EVT.FX.json')"
echo "RB: $(ls saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/ | grep -c 'P1.TE.G2.EVT.FX.json')"
echo
echo '=== queue log: FX dispatch lines ==='
grep -n "task#68[7-9]\|task#69[0-6]" tools/queue/queue.log | tail -14 | cut -c1-120
