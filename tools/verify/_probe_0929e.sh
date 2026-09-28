#!/bin/bash
cd ~/DyGLib || exit 1
echo '=== exact-name save lines: RT seed42 NN-15.LF-3 (grid cand) ==='
grep -l "save model .*NN-15.LF-3\.RAS-E\.RASE-E\.BTE-E\.CNAS-E\.P1\.TE\.pkl" logs/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed42/*.log 2>/dev/null | while read f; do ls -l --time-style=+%m-%d_%H:%M "$f"; done
echo '=== exact-name save lines: RT seed42 NN-60.LF-1 (ctrl) ==='
grep -l "save model .*NN-60.LF-1\.RAS-E\.RASE-E\.BTE-E\.CNAS-E\.P1\.TE\.pkl" logs/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed42/*.log 2>/dev/null | while read f; do ls -l --time-style=+%m-%d_%H:%M "$f"; done
echo '=== exact-name save lines: sign RT seed42 NN-60.LF-3 (cand) ==='
grep -l "save model .*NN-60.LF-3\.RAS-E\.RASE-E\.BTE-E\.CNAS-E\.P1\.TE\.pkl" logs/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed42/*.log 2>/dev/null | while read f; do ls -l --time-style=+%m-%d_%H:%M "$f"; done
echo '=== exact-name save lines: RB seed42 NN-60.LF-1 (linksign cand) ==='
grep -l "save model .*NN-60.LF-1\.RAS-E\.RASE-E\.BTE-E\.CNAS-E\.P1\.TE\.pkl" logs/SignDyGFormer/RedditHyperlinkBody/SignDyGFormer_seed42/*.log 2>/dev/null | while read f; do ls -l --time-style=+%m-%d_%H:%M "$f"; done
echo '=== exact-name save lines: RB seed42 NN-80.LF-3 (linksign ctrl) ==='
grep -l "save model .*NN-80.LF-3\.RAS-E\.RASE-E\.BTE-E\.CNAS-E\.P1\.TE\.pkl" logs/SignDyGFormer/RedditHyperlinkBody/SignDyGFormer_seed42/*.log 2>/dev/null | while read f; do ls -l --time-style=+%m-%d_%H:%M "$f"; done
echo '=== 检查误配日志（前一轮输出里被选中的 1790483328.2023282.log） ==='
grep -c "save model" logs/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed42/1790483328.2023282.log 2>/dev/null
grep "save model" logs/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed42/1790483328.2023282.log 2>/dev/null | head -4
wc -l logs/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed42/1790483328.2023282.log 2>/dev/null
