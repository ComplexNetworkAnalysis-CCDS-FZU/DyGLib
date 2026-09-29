#!/bin/bash
cd ~/DyGLib || exit 1
F=logs/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed123/1789627982.2108428.log
echo "=== size:"
ls -la "$F" 2>/dev/null
echo "=== validate lines (first block) ==="
grep -n "validate" "$F" 2>/dev/null | head -16
echo "=== save model lines ==="
grep -n "save model" "$F" 2>/dev/null | head -6
echo "=== around first save ==="
S=$(grep -n "save model " "$F" | head -1 | cut -d: -f1)
sed -n "$((S-16)),$((S+2))p" "$F" 2>/dev/null
