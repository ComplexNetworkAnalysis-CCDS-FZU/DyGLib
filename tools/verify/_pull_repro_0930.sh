#!/bin/bash
cd ~/DynamiSE_DySDGNN_repro || exit 1
echo "== pull =="
git pull --ff-only 2>&1 | tail -6
echo "== HEAD =="
git log -1 --oneline | cut -c1-70
echo "== 关键文件 =="
ls -l scripts/m5_run.py scripts/verify_visibility.py 2>&1
echo "== eval-protocol 支持 =="
grep -n "eval-protocol\|eval_protocol" scripts/m5_run.py | head -8
echo "== 工作树状态（前 6）=="
git status --porcelain | head -6
