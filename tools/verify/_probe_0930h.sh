#!/bin/bash
echo "== 服务器 repro remote =="
cd ~/DynamiSE_DySDGNN_repro && git remote -v
echo "-- 裸库目录 --"
ls ~/git/ 2>/dev/null
echo "== 服务器 repro 是否含 648c3fe =="
cd ~/DynamiSE_DySDGNN_repro && git cat-file -t 648c3fe 2>&1 | head -2
echo "== 本地 DyGLib remote（参照）=="
cd ~/DyGLib && git remote -v
