#!/bin/bash
cd ~/DyGLib || exit 1
F=logs/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed42/1790369281.6263242.log
echo "=== sign log: first validate-ish lines ==="
grep -n "validate" "$F" 2>/dev/null | head -30
echo
echo "=== sign log: save model lines ==="
grep -n "save model" "$F" 2>/dev/null | head -6
echo
echo "=== sign log: sample around a save ==="
S=$(grep -n "save model " "$F" | head -1 | cut -d: -f1)
echo "first save at line $S"
sed -n "$((S-24)),$((S+2))p" "$F"
